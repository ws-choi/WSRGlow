import pickle
from copy import deepcopy
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils import utils


class IndexedDataset:
    def __init__(self, path, num_cache=1):
        super().__init__()
        self.path = path
        self.data_file = None
        self.data_offsets = np.load(f"{path}.idx", allow_pickle=True).item()['offsets']
        self.data_file = open(f"{path}.data", 'rb', buffering=-1)
        self.cache = []
        self.num_cache = num_cache

    def check_index(self, i):
        if i < 0 or i >= len(self.data_offsets) - 1:
            raise IndexError('index out of range')

    def __del__(self):
        if self.data_file:
            self.data_file.close()

    def __getitem__(self, i):
        self.check_index(i)
        if self.num_cache > 0:
            for c in self.cache:
                if c[0] == i:
                    return c[1]
        self.data_file.seek(self.data_offsets[i])
        b = self.data_file.read(self.data_offsets[i + 1] - self.data_offsets[i])
        item = pickle.loads(b)
        if self.num_cache > 0:
            self.cache = [(i, deepcopy(item))] + self.cache[:-1]
        return item

    def __len__(self):
        return len(self.data_offsets) - 1


class BaseDataset(Dataset):
    def __init__(self, shuffle, sort_by_len, max_frames):
        super().__init__()
        self.shuffle = shuffle
        self.sort_by_len = sort_by_len
        self.max_frames = max_frames
        self.sizes = None

    @property
    def _sizes(self):
        return self.sizes

    def __getitem__(self, index):
        raise NotImplementedError

    def collater(self, samples):
        raise NotImplementedError

    def __len__(self):
        return len(self._sizes)

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        size = min(self._sizes[index], self.max_frames)
        return size

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
            if self.sort_by_len:
                indices = indices[np.argsort(
                    np.array(self._sizes)[indices], kind='mergesort')]
        else:
            indices = np.arange(len(self))
        return indices


class SRDataset(BaseDataset):
    def __init__(self, shuffle, audio_sample_rate, resampled_rate, sort_by_len, max_frames,
                 prefix, binary_data_dir, max_samples, n_fft):
        super().__init__(shuffle, sort_by_len, max_frames)
        self.data_dir = binary_data_dir
        self.prefix = prefix
        self.is_infer = prefix == 'test'
        self.batch_max_samples = 0 if self.is_infer else max_samples
        self.resampled_rate = resampled_rate
        self.sizes = np.load(f'{self.data_dir}/{resampled_rate}/{self.prefix}_lengths.npy')
        self.indexed_ds = None
        self.n_fft = n_fft
        self.resample_ratio = float(resampled_rate) / audio_sample_rate

    def _get_item(self, index):
        if hasattr(self, 'avail_idxs') and self.avail_idxs is not None:
            index = self.avail_idxs[index]
        if self.indexed_ds is None:
            self.indexed_ds = IndexedDataset(f'{self.data_dir}/{self.resampled_rate}/{self.prefix}')
        return self.indexed_ds[index]

    def __getitem__(self, index):
        item = self._get_item(index)

        sample = {
            'id': index,
            'item_name': item['item_name'],
            'wav': item['wav'],
            'lr_wav': item['resampled_wav']
        }

        return sample

    def collater(self, samples):
        if len(samples) == 0:
            return {}

        id = []
        item_name = []
        hr_batch = []
        lr_batch = []
        lr_mag_batch = []
        lr_pha_batch = []
        hr_mag_batch = []
        hr_pha_batch = []
        n_fft = self.n_fft
        for (idx, s) in enumerate(samples):
            id.append(s['id'])
            item_name.append(s['item_name'])
            hr, lr = s['wav'], s['lr_wav']
            if len(hr) > self.batch_max_samples:
                batch_max_samples = len(hr) if self.is_infer else self.batch_max_samples
                lr_max_samples = int(self.resample_ratio * batch_max_samples)
                lr_start_step = np.random.randint(0, len(lr) - lr_max_samples + 1)
                # print(hr.shape)
                hr_ = hr[int(lr_start_step / self.resample_ratio): int(
                    (lr_start_step + lr_max_samples) / self.resample_ratio)]
                lr_ = lr[lr_start_step: lr_start_step + lr_max_samples]
                Dlow = librosa.stft(lr_, n_fft=n_fft // 2)
                lr_mag = np.abs(Dlow)
                lr_pha = np.angle(Dlow)
                D = librosa.stft(hr_, n_fft=n_fft)
                hr_mag = np.abs(D)
                hr_pha = np.angle(D)
            else:
                print(f'Removed short sample from batch (length={len(hr)}).')
                continue
            hr_batch += [torch.FloatTensor(hr_)]
            lr_batch += [torch.FloatTensor(lr_)]
            lr_mag_batch += [torch.FloatTensor(lr_mag).t()]
            lr_pha_batch += [torch.FloatTensor(lr_pha).t()]
            hr_mag_batch += [torch.FloatTensor(hr_mag).t()]
            hr_pha_batch += [torch.FloatTensor(hr_pha).t()]

        hr_batch = utils.collate_1d(hr_batch, 0)
        lr_batch = utils.collate_1d(lr_batch, 0)
        lr_mag_batch = utils.collate_2d(lr_mag_batch, 0).permute(0, 2, 1)
        lr_pha_batch = utils.collate_2d(lr_pha_batch, 0).permute(0, 2, 1)
        hr_mag_batch = utils.collate_2d(hr_mag_batch, 0).permute(0, 2, 1)
        hr_pha_batch = utils.collate_2d(hr_pha_batch, 0).permute(0, 2, 1)

        return {
            'wavs': hr_batch,
            'nsamples': len(samples),
            'resampled_wavs': lr_batch,
            'item_name': item_name,
            'lr_mags': lr_mag_batch,
            'lr_phas': lr_pha_batch,
            'hr_mags': hr_mag_batch,
            'hr_phas': hr_pha_batch
        }


class SRDataset_2PR(SRDataset):
    def __init__(self, shuffle, audio_sample_rate, resampled_rate, target_rate,
                 sort_by_len, max_frames, prefix, binary_data_dir,
                 max_samples, n_fft):

        super().__init__(shuffle, audio_sample_rate, resampled_rate, sort_by_len, max_frames, prefix, binary_data_dir,
                         max_samples, n_fft)
        self.target_rate = target_rate
        self.sizes = np.load(f'{self.data_dir}/{target_rate}_to_{resampled_rate}/{self.prefix}_lengths.npy')

    def _get_item(self, index):
        if hasattr(self, 'avail_idxs') and self.avail_idxs is not None:
            index = self.avail_idxs[index]
        if self.indexed_ds is None:
            self.indexed_ds = IndexedDataset(f'{self.data_dir}/{self.target_rate}_to_{self.resampled_rate}/{self.prefix}')
        return self.indexed_ds[index]

    def __getitem__(self, index):
        item = self._get_item(index)

        sample = {
            'id': index,
            'item_name': item['item_name'],
            'wav': item['wav'],
            'lr_wav': item['resampled_wav']
        }

        return sample

    def collater(self, samples):
        if len(samples) == 0:
            return {}

        id = []
        item_name = []
        hr_batch = []
        lr_batch = []
        lr_mag_batch = []
        lr_pha_batch = []
        hr_mag_batch = []
        hr_pha_batch = []
        n_fft = self.n_fft
        for (idx, s) in enumerate(samples):
            id.append(s['id'])
            item_name.append(s['item_name'])
            hr, lr = s['wav'], s['lr_wav']
            if len(hr) > self.batch_max_samples:
                batch_max_samples = len(hr) if self.is_infer else self.batch_max_samples
                lr_max_samples = int(self.resample_ratio * batch_max_samples)
                lr_start_step = np.random.randint(0, len(lr) - lr_max_samples + 1)
                # print(hr.shape)
                hr_ = hr[int(lr_start_step / self.resample_ratio): int(
                    (lr_start_step + lr_max_samples) / self.resample_ratio)]
                lr_ = lr[lr_start_step: lr_start_step + lr_max_samples]
                Dlow = librosa.stft(lr_, n_fft=n_fft // 2)
                lr_mag = np.abs(Dlow)
                lr_pha = np.angle(Dlow)
                D = librosa.stft(hr_, n_fft=n_fft)
                hr_mag = np.abs(D)
                hr_pha = np.angle(D)
            else:
                print(f'Removed short sample from batch (length={len(hr)}).')
                continue
            hr_batch += [torch.FloatTensor(hr_)]
            lr_batch += [torch.FloatTensor(lr_)]
            lr_mag_batch += [torch.FloatTensor(lr_mag).t()]
            lr_pha_batch += [torch.FloatTensor(lr_pha).t()]
            hr_mag_batch += [torch.FloatTensor(hr_mag).t()]
            hr_pha_batch += [torch.FloatTensor(hr_pha).t()]

        hr_batch = utils.collate_1d(hr_batch, 0)
        lr_batch = utils.collate_1d(lr_batch, 0)
        lr_mag_batch = utils.collate_2d(lr_mag_batch, 0).permute(0, 2, 1)
        lr_pha_batch = utils.collate_2d(lr_pha_batch, 0).permute(0, 2, 1)
        hr_mag_batch = utils.collate_2d(hr_mag_batch, 0).permute(0, 2, 1)
        hr_pha_batch = utils.collate_2d(hr_pha_batch, 0).permute(0, 2, 1)

        return {
            'wavs': hr_batch,
            'nsamples': len(samples),
            'resampled_wavs': lr_batch,
            'item_name': item_name,
            'lr_mags': lr_mag_batch,
            'lr_phas': lr_pha_batch,
            'hr_mags': hr_mag_batch,
            'hr_phas': hr_pha_batch
        }
