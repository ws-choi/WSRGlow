import torch
import librosa
import numpy as np
from src.models.components.submodules import MuLawEmbedding, AngleEmbedding, Invertible1x1Conv, WN


def remove(conv_list):
    new_conv_list = torch.nn.ModuleList()
    for old_conv in conv_list:
        old_conv = torch.nn.utils.remove_weight_norm(old_conv)
        new_conv_list.append(old_conv)
    return new_conv_list


class WaveGlowMelHF(torch.nn.Module):
    def __init__(self, mu, embed_num, embed_dim, n_flows, n_group, n_early_every,
                 n_early_size, WN_config):
        super(WaveGlowMelHF, self).__init__()
        self.muembed = MuLawEmbedding(mu, embed_num, embed_dim)
        #print(f'embed_dim * n_group = {embed_dim * n_group}', flush=True)

        assert (n_group % 2 == 0)
        self.n_flows = n_flows
        self.n_group = n_group
        self.n_early_every = n_early_every
        self.n_early_size = n_early_size
        self.WN = torch.nn.ModuleList()
        self.convinv = torch.nn.ModuleList()
        self.phase_embedding = AngleEmbedding(embed_num=120, hidden_dim=50)

        n_half = n_group

        # Set up layers with the right sizes based on how many dimensions
        # have been output already
        n_remaining_channels = 2 * n_group
        #print(f'n_half = {n_half}', flush=True)
        for k in range(n_flows):
            if k % self.n_early_every == 0 and k > 0:
                n_half = n_half - int(self.n_early_size / 2)
                n_remaining_channels = n_remaining_channels - self.n_early_size
            self.convinv.append(Invertible1x1Conv(n_remaining_channels))
            self.WN.append(WN(n_half, embed_dim * n_group + 50 * (n_group + 1) + n_group + 1, **WN_config))  # ??
        self.n_remaining_channels = n_remaining_channels  # Useful during inference

    def forward(self, lr, hr):
        """
        audio: batch x time
        """
        #print(lr.shape)
        T = lr.shape[1]
        lr = lr

        n_group = self.n_group
        Ds = [librosa.stft(x, n_fft=n_group * 2, hop_length=n_group) for x in lr.cpu().numpy()]
        spect = torch.tensor([np.abs(d) for d in Ds], dtype=lr.dtype, device=lr.device)  # (B, n_group + 1, T / 2 / n_group)
        phase = torch.tensor([np.angle(d) for d in Ds], dtype=lr.dtype, device=lr.device)  # (B, n_group + 1, T / 2 / n_group)
        phaseemb = self.phase_embedding(phase.permute(0, 2, 1))  # (B, n_group + 1, T / 2 / n_group, H)
        #print(f'spect.shape = {spect.shape}')
        phaseemb = phaseemb.reshape(phaseemb.shape[0], phaseemb.shape[1], -1).permute(0, 2, 1)
        # (B, (n_group + 1) * H, T / 2 / n_group)
        #print(f'phaseemb.shape = {phaseemb.shape}')

        #  use mu-law embedding to depict low res audio
        lr = self.muembed(lr).permute(0, 2, 1)  # (B, H, T / 2)


        #print(f'lr_shape after muembed = {lr.shape}')
        lr = lr.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)
        # (B, T / 2 / n_group, H, n_group)
        lr = lr.contiguous().view(lr.size(0), lr.size(1), -1).permute(0, 2, 1)
        #print(f'lr.shape = {lr.shape}', lr.shape, flush=True)
        # (B, H x n_group, T / 2 / n_group)

        min_dim2 = min([lr.shape[2], spect.shape[2], phaseemb.shape[2]])
        lr = lr[:, :, :min_dim2]
        spect = spect[:, :, :min_dim2]
        phaseemb = phaseemb[:, :, :min_dim2]

        lr = torch.cat((lr, spect, phaseemb), dim=1)
        # H1 = embed_dim for phase
        # H2 = embed_dim for waveform
        # (B, H1 x (n_group + 1) + H2 x n_group + n_group + 1, T / 2 / n_group)
        #print(f'lr.shape = {lr.shape}', flush=True)

        audio = hr.reshape(hr.shape[0], -1)  # (B, T)

        # TODO : 2 means what?
        # audio = audio.unfold(1, self.n_group * 2, self.n_group * 2).permute(0, 2, 1)
        audio = audio.unfold(1, self.n_group * 2, self.n_group * 2).permute(0, 2, 1)
        #print(f'lr.shape = {lr.shape}, audio.shape = {audio.shape}', flush=True)
        #print(self.n_group, audio.shape, flush=True)
        # batch x (n_group * 2) x (time / n_group / 2)
        #print(f'audio.shape = {audio.shape}', flush=True)
        output_audio = []
        log_s_list = []
        log_det_W_list = []

        #print(f'lr.shape = {lr.shape}')
        #print(f'audio.shape = {audio.shape}')

        for k in range(self.n_flows):
            #print(f'k = {k}', flush=True)
            if k % self.n_early_every == 0 and k > 0:
                output_audio.append(audio[:, :self.n_early_size, :])
                audio = audio[:, self.n_early_size:, :]

            audio, log_det_W = self.convinv[k](audio)
            log_det_W_list.append(log_det_W)

            n_half = int(audio.size(1) / 2)
            audio_0 = audio[:, :n_half, :]
            audio_1 = audio[:, n_half:, :]

            output = self.WN[k]((audio_0, lr))
            log_s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            audio_1 = torch.exp(log_s) * audio_1 + b
            log_s_list.append(log_s)

            audio = torch.cat([audio_0, audio_1], 1)

        output_audio.append(audio)
        return torch.cat(output_audio, 1), log_s_list, log_det_W_list

    def infer(self, lr, sigma=1.0):
        n_group = self.n_group
        Ds = [librosa.stft(x, n_fft=n_group * 2, hop_length=n_group) for x in lr.cpu().numpy()]
        spect = torch.tensor([np.abs(d) for d in Ds], dtype=lr.dtype,
                             device=lr.device)  # (B, n_group + 1, T / 2 / n_group)
        phase = torch.tensor([np.angle(d) for d in Ds], dtype=lr.dtype,
                             device=lr.device)  # (B, n_group + 1, T / 2 / n_group)
        phaseemb = self.phase_embedding(phase.permute(0, 2, 1))  # (B, n_group + 1, T / 2 / n_group, H)
        phaseemb = phaseemb.reshape(phaseemb.shape[0], phaseemb.shape[1], -1).permute(0, 2, 1)
        lr = self.muembed(lr).permute(0, 2, 1)  # (B, H, T / 2)
        lr = lr.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)
        lr = lr.contiguous().view(lr.size(0), lr.size(1), -1).permute(0, 2, 1)

        min_dim2 = min([lr.shape[2], spect.shape[2], phaseemb.shape[2]])
        lr = lr[:, :, :min_dim2]
        spect = spect[:, :, :min_dim2]
        phaseemb = phaseemb[:, :, :min_dim2]
        lr = torch.cat((lr, spect, phaseemb), dim=1)

        audio = torch.randn(lr.size(0), self.n_remaining_channels, lr.size(2),
                            dtype=lr.dtype,
                            device=lr.device)

        # print(f'sigma = {sigma}')
        # print(f'audio.shape = {audio.shape}', flush=True)
        audio = torch.autograd.Variable(sigma * audio)

        for k in reversed(range(self.n_flows)):
            n_half = int(audio.size(1) / 2)
            audio_0 = audio[:, :n_half, :]
            audio_1 = audio[:, n_half:, :]

            output = self.WN[k]((audio_0, lr))

            s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            audio_1 = (audio_1 - b) / torch.exp(s)
            audio = torch.cat([audio_0, audio_1], 1)

            audio = self.convinv[k](audio, reverse=True)

            if k % self.n_early_every == 0 and k > 0:
                z = torch.randn(lr.size(0), self.n_early_size, lr.size(2),
                                dtype=lr.dtype,
                                device=lr.device)
                audio = torch.cat((sigma * z, audio), 1)

        audio = audio.permute(0, 2, 1).contiguous().view(audio.size(0), -1)
        audio = audio.reshape(audio.shape[0], 1, -1)
        #print(audio.shape, lr__.shape)
        return audio

    @staticmethod
    def remove_weightnorm(model):
        waveglow = model
        for WN in waveglow.WN:
            WN.start = torch.nn.utils.remove_weight_norm(WN.start)
            WN.in_layers = remove(WN.in_layers)
            WN.cond_layer = torch.nn.utils.remove_weight_norm(WN.cond_layer)
            WN.res_skip_layers = remove(WN.res_skip_layers)
        return waveglow
