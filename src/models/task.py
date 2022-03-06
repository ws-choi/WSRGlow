import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.optim import Adam


class Task4(pl.LightningModule):

    def __init__(self, net, optim_config, criterion):
        super(Task4, self).__init__()

        # self.save_hyperparameters() # TODO: need to be flatten

        self.net = net
        self.optim_config = optim_config
        self.criterion = criterion

    def configure_optimizers(self):
        return Adam(self.parameters(), **self.optim_config)

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        sample, _ = args
        lr = sample['resampled_wavs']
        hr = sample['wavs']
        output = self.net(lr, hr)
        loss = self.criterion(output)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss