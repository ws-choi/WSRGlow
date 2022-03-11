import librosa
import numpy as np
import pyloudnorm as pyln
import torch


def run(model, wav, sigma=1.0):
    wav = torch.Tensor(wav).reshape(1, -1)
    wav = wav.to(model.device)
    output = np.array(model.infer(wav, sigma=sigma)[0].cpu().detach())
    output = output.reshape(-1)
    return output


def librosa_pad_lr(x, fsize, fshift, pad_sides=1):
    '''compute right padding (final frame) or both sides padding (first and final frames)
    '''
    assert pad_sides in (1, 2)
    # return int(fsize // 2)
    pad = (x.shape[0] // fshift + 1) * fshift - x.shape[0]
    if pad_sides == 1:
        return 0, pad
    else:
        return pad // 2, pad // 2 + pad % 2


def load_wav(wav_fn, sr_rates, wav2spec_args):
    audio_sample_rate = sr_rates['audio_sample_rate']
    target_rate = sr_rates['target_rate']
    resampled_rate = sr_rates['resampled_rate']

    wav, sr = librosa.load(wav_fn, audio_sample_rate)
    assert sr == audio_sample_rate

    if wav2spec_args['loud_norm']:
        print('LOUD NORM!', flush=True)
        meter = pyln.Meter(sr)  # create BS.1770 meter
        loudness = meter.integrated_loudness(wav)
        wav = pyln.normalize.loudness(wav, loudness, -22.0)
        if np.abs(wav).max() > 1:
            wav = wav / np.abs(wav).max()

    real_target = librosa.resample(wav, sr, target_rate)
    print('real target rate: ', real_target.shape, sr, target_rate)

    resample_target = librosa.resample(real_target, target_rate, resampled_rate)
    print('resampled rate: ', resample_target.shape, target_rate, resampled_rate)
    return resample_target, resampled_rate

    # Below are not used
    wav = resample_target

    # get amplitude spectrogram
    fft_size = wav2spec_args['fft_size']
    hop_size = wav2spec_args['hop_size']
    win_length = wav2spec_args['win_size']
    fmin = wav2spec_args['fmin']
    fmax = wav2spec_args['fmax']
    sample_rate = wav2spec_args['audio_sample_rate']
    num_mels = wav2spec_args['audio_num_mel_bins']
    x_stft = librosa.stft(wav, n_fft=fft_size, hop_length=hop_size,
                          win_length=win_length, window='hann', pad_mode="constant")
    spc = np.abs(x_stft)  # (n_bins, T)

    # get mel basis
    fmin = 0 if fmin == -1 else fmin
    fmax = sample_rate / 2 if fmax == -1 else fmax
    mel_basis = librosa.filters.mel(sample_rate, fft_size, num_mels, fmin, fmax)
    mel = mel_basis @ spc
    eps = 1e-10
    mel = np.log10(np.maximum(eps, mel))  # (n_mel_bins, T)

    l_pad, r_pad = librosa_pad_lr(wav, fft_size, hop_size, 1)
    wav = np.pad(wav, (l_pad, r_pad), mode='constant', constant_values=0.0)
    wav = wav[:mel.shape[1] * hop_size]
    return wav, resampled_rate
