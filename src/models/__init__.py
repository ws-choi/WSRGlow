import glob
from pathlib import Path
import hydra
import torch
from omegaconf import OmegaConf

from src.models.TesteeWrapper import WSRGlowTestee
from src.models.task import Task4


def load_wrapper_model(hydra_dir, device='cpu'):
    hydra_dir = Path(str(hydra_dir))

    with open(hydra_dir.joinpath('.hydra/config.yaml')) as f:
        train_config = OmegaConf.load(f)
        model_config = train_config['model']
        skeleton_model = hydra.utils.instantiate(model_config)
        print('skeleton_model is loaded.\n\tsr: {} to {}'.format(train_config['datamodule']['target_rate'],
                                                                train_config['datamodule']['audio_sample_rate']))

        checkpoints = glob.glob(str(hydra_dir.joinpath('checkpoints/*.ckpt')))
        print('checkpoints list:')
        for checkpoint in checkpoints:
            print('\t{}'.format(checkpoint))

        wav2spec_args = train_config['datamodule']['binarizer']['wav2spec_args']

        sr_rates = {}
        sr_rates['audio_sample_rate'] = train_config['datamodule']['audio_sample_rate']
        sr_rates['target_rate'] = train_config['datamodule']['target_rate']
        sr_rates['resampled_rate'] = train_config['datamodule']['resampled_rate']
        sigma = train_config['model']['criterion']['sigma']
        return WSRGlowTestee(skeleton_model, sr_rates, wav2spec_args, sigma, device), checkpoints
