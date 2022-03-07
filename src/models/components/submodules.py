import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


#@torch.jit.script #TODO
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


class MuLawEmbedding(torch.nn.Module):
    def __init__(self, mu, embed_num, hidden_dim):
        super(MuLawEmbedding, self).__init__()
        self.mu = mu
        self.embed_num = embed_num
        self.embed = Embedding(num_embeddings=embed_num,
                               embedding_dim=hidden_dim)

    def forward(self, index):
        # forward_input: batch x (time / 2)
        # print(index.device, flush=True)
        index = index.sign()
        index = index * torch.log(1 + self.mu *
                                  torch.abs(index)) / np.log(1 + self.mu)
        # (-1, 1)
        embed_num = self.embed_num
        index = ((index + 1) * (self.embed_num // 2)).floor().long()
        index = (index < 0) * 0 + (index >= 0) * (index < embed_num) * \
                index + (index >= embed_num) * (embed_num - 1)
        # [0, 256)
        assert torch.min(index).item() >= 0 and torch.max(
            index).item() < embed_num
        return self.embed(index)


class AngleEmbedding(torch.nn.Module):
    def __init__(self, embed_num, hidden_dim):
        super(AngleEmbedding, self).__init__()
        self.embed_num = embed_num
        self.embed = Embedding(num_embeddings=embed_num,
                               embedding_dim=hidden_dim)

    def forward(self, index):
        embed_num = self.embed_num
        index = ((index / np.pi + 1) * (embed_num // 2)).floor().long()
        index = (index < 0) * 0 + (index >= 0) * (index < embed_num) * \
                index + (index >= embed_num) * (embed_num - 1)
        assert torch.min(index).item() >= 0 and torch.max(
            index).item() < embed_num
        return self.embed(index)


class Invertible1x1Conv(torch.nn.Module):
    """
    The layer outputs both the convolution, and the log determinant
    of its weight matrix.  If reverse=True it does convolution with
    inverse
    """

    def __init__(self, c):
        super(Invertible1x1Conv, self).__init__()
        self.conv = torch.nn.Conv1d(c, c, kernel_size=1, stride=1, padding=0,
                                    bias=False)

        # Sample a random orthonormal matrix to initialize weights
        W = torch.empty(c, c)
        torch.nn.init.orthogonal_(W)
        while torch.isnan(torch.logdet(W)):
            torch.nn.init.orthogonal_(W)
        # print(f'W={W}')
        # print(f'detW={torch.det(W)}', flush=True)
        # print(f'WTW={torch.mm(W.t(), W)}', flush=True)

        # Ensure determinant is 1.0 not -1.0
        if torch.det(W) < 0:
            W[:, 0] = -1 * W[:, 0]
        W = W.view(c, c, 1)
        self.conv.weight.data = W

    def forward(self, z, reverse=False):
        # shape
        batch_size, group_size, n_of_groups = z.size()

        W = self.conv.weight.squeeze()

        if reverse:
            if not hasattr(self, 'W_inverse'):
                # Reverse computation
                W_inverse = W.float().inverse()
                W_inverse = Variable(W_inverse[..., None])
                if z.type() == 'torch.cuda.HalfTensor':
                    W_inverse = W_inverse.half()
                self.W_inverse = W_inverse
            z = F.conv1d(z, self.W_inverse, bias=None, stride=1, padding=0)
            return z
        else:
            # Forward computation
            log_det_W = batch_size * n_of_groups * torch.logdet(W)
            z = self.conv(z)
            return z, log_det_W


class WN(torch.nn.Module):
    """
    This is the WaveNet like layer for the affine coupling.  The primary difference
    from WaveNet is the convolutions need not be causal.  There is also no dilation
    size reset.  The dilation only doubles on each layer
    """

    def __init__(self, n_in_channels, n_mel_channels, n_layers, n_channels,
                 kernel_size):
        super(WN, self).__init__()
        assert (kernel_size % 2 == 1)
        assert (n_channels % 2 == 0)
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()

        start = torch.nn.Conv1d(n_in_channels, n_channels, 1)
        start = torch.nn.utils.weight_norm(start, name='weight')
        self.start = start

        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        end = torch.nn.Conv1d(n_channels, 2 * n_in_channels, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end
        # print(f'n_channels = {n_channels}, n_channels = {n_layers}')
        cond_layer = torch.nn.Conv1d(
            n_mel_channels, 2 * n_channels * n_layers, 1)
        self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')

        for i in range(n_layers):
            dilation = 2 ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = torch.nn.Conv1d(n_channels, 2 * n_channels, kernel_size,
                                       dilation=dilation, padding=padding)
            in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2 * n_channels
            else:
                res_skip_channels = n_channels
            res_skip_layer = torch.nn.Conv1d(n_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(
                res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, forward_input):
        audio, spect = forward_input
        # print(self.start, flush=True)
        audio = self.start(audio)
        output = torch.zeros_like(audio)
        n_channels_tensor = torch.IntTensor([self.n_channels])

        spect = self.cond_layer(spect)

        # print(f'n_channels * 2 = {self.n_channels * 2}')
        for i in range(self.n_layers):
            spect_offset = i * 2 * self.n_channels
            # print(f'i = {i}')
            # print(f'shape1 = {self.in_layers[i](audio).shape}')
            # print(f'shape2 = {spect[:, spect_offset:spect_offset + 2 * self.n_channels, :].shape}', flush=True)
            acts = fused_add_tanh_sigmoid_multiply(
                self.in_layers[i](audio),
                spect[:, spect_offset:spect_offset + 2 * self.n_channels, :],
                n_channels_tensor)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                audio = audio + res_skip_acts[:, :self.n_channels, :]
                output = output + res_skip_acts[:, self.n_channels:, :]
            else:
                output = output + res_skip_acts
            # print(f'WN: i={i}, output={output}')

        return self.end(output)
