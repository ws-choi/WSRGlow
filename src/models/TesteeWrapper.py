import warnings
import torch
import numpy as np


class WSRGlowTestee:
    def __init__(self, skeleton_model, sr_rates, wav2spec_args, sigma, device='cpu') -> None:
        super().__init__()
        self.model = skeleton_model.to(device)
        print('model is on {}'.format(device))

        self.sr_rates = sr_rates
        warnings.warn('\n **IMPORTANT** WSRGlowTestee: \n'
                      'Input audio must have been resampled to 24k'
                      'using librosa library to match the dimension.')
        self.wav2spec_args = wav2spec_args
        self.sigma = sigma
        self.device = device
        self.model.eval()

    def load_checkpoints (self, checkpoint):
        ckpt_path = str(checkpoint)
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        print('checkpoint loaded: {}'.format(ckpt_path))

    def infer(self, x, target):
        """A testee that do nothing
        Args:
            x (np.array): [sample,], with original_sr sample rate
            ***IMPORTANT*** x must be re-resampled to 24k

            target (np.array): [sample,], with target_sr sample rate
        Returns:
            np.array: [sample,]
        """

        wav = torch.Tensor(x).reshape(1, -1)
        wav = wav.to(self.device)
        output = np.array(self.model.infer(wav, sigma=self.sigma)[0].cpu().detach())
        x = output.reshape(-1)

        return x
