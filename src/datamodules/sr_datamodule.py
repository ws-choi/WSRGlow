import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from src.datamodules.components.sr_bianarizer import DapsSRBinarizer
from src.datamodules.components.sr_dataset import SRDataset


class SRDataModule(LightningDataModule):

    def __init__(
            self,
            data_dir,
            batch_size,
            num_workers,
            pin_memory,
            sort_by_len,
            max_frames,
            max_samples,
            n_fft,
            test_spkers,
            audio_sample_rate,
            resampled_rate,
            binarizer: DapsSRBinarizer
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[SRDataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.sort_by_len = sort_by_len
        self.max_frames = max_frames
        self.max_samples = max_samples
        self.n_fft = n_fft
        self.test_spkers = test_spkers
        self.binarizer = binarizer

        self.audio_sample_rate = audio_sample_rate
        self.resampled_rate = resampled_rate

    def prepare_data(self):
        target_dir = Path(self.data_dir).joinpath(str(self.resampled_rate))
        if not os.path.exists(target_dir):
            self.binarizer.process()

        self.data_train = SRDataset(shuffle=True,
                                    sort_by_len=self.sort_by_len,
                                    max_frames=self.max_frames,
                                    prefix='train',
                                    binary_data_dir=self.data_dir,
                                    max_samples=self.max_samples,
                                    n_fft=self.n_fft,
                                    audio_sample_rate=self.audio_sample_rate,
                                    resampled_rate=self.resampled_rate)

    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=self.data_train.collater
        )

    def val_dataloader(self):
        return None

    def test_dataloader(self):
        return None