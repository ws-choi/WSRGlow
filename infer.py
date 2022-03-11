import glob
import os
from pathlib import Path

import dotenv
import hydra
import soundfile as sf
from omegaconf import DictConfig, OmegaConf

from src.infer import *
from src.models.TesteeWrapper import WSRGlowTestee

dotenv.load_dotenv(override=True)


@hydra.main(config_path="configs/", config_name="infer.yaml")
def main(config: DictConfig):

    eval_dir = Path(config['eval_dir'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    from src.models import load_wrapper_model as lm
    wsrglow_testee, checkpoints = lm(eval_dir, device)
    wsrglow_testee.load_checkpoints(checkpoints[0])

    output_dir = Path(config['output_dir'])
    if not output_dir.exists():
        os.mkdir(output_dir)
    output_dir = output_dir.joinpath(
        '{}_to_{}_{}'.format(wsrglow_testee.sr_rates['target_rate'],
                             wsrglow_testee.sr_rates['audio_sample_rate'],
                             Path(config['ckpt']).name)
    )

    if not output_dir.exists():
        os.mkdir(output_dir)

    fns = []
    data_dir = config['vctk_root']
    file_format = config['file_format']
    for spker in sorted(config['test_spkers']):
        fns += sorted(glob.glob(f'{data_dir}/wav48/{spker}/*.{file_format}'))

    for lr_fn in fns:
        lr, sr = load_wav(lr_fn, wsrglow_testee.sr_rates, wsrglow_testee.wav2spec_args)
        # lr, _ = dataset_utils.wav2spec(lr_fn, wav2spec_args)
        sr = wsrglow_testee.sr_rates['target_rate']
        print(f'sampling rate (lr) = {sr}')
        print(f'lr.shape = {lr.shape}', flush=True)
        with torch.no_grad():
            pred = wsrglow_testee.infer(lr, None)
        print(lr.shape, pred.shape)
        pred_fn = f'pred_{lr_fn}'
        print(f'sampling rate = { wsrglow_testee.sr_rates["audio_sample_rate"]}')

        origin_path = Path(pred_fn)
        output_path = output_dir.joinpath(origin_path.parent.name)
        if not output_path.exists():
            os.mkdir(output_path)

        sf.write(output_path.joinpath(origin_path.name), pred, wsrglow_testee.sr_rates['audio_sample_rate'])


def load_model(config):
    eval_dir = Path(config['eval_dir'])
    assert eval_dir.exists()
    ckpt = Path(config['ckpt'])
    assert ckpt.exists()

    with open(eval_dir.joinpath('.hydra/config.yaml')) as f:
        train_config = OmegaConf.load(f)
        model_config = train_config['model']
        model = hydra.utils.instantiate(model_config)
        print('model loaded. sr: {} to {}'.format(train_config['datamodule']['target_rate'],
                                                  train_config['datamodule']['audio_sample_rate']))

        checkpoint = torch.load(str(ckpt), map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        print('checkpoint loaded: {}'.format(str(ckpt)))

        wav2spec_args = train_config['datamodule']['binarizer']['wav2spec_args']

        sr_rates = {}
        sr_rates['audio_sample_rate'] = train_config['datamodule']['audio_sample_rate']
        sr_rates['target_rate'] = train_config['datamodule']['target_rate']
        sr_rates['resampled_rate'] = train_config['datamodule']['resampled_rate']
        return model, sr_rates, wav2spec_args, train_config['model']['criterion']['sigma']


if __name__ == "__main__":
    main()
