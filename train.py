from models.CrackModule import CrackModule
from data.pldatamodule import CrackDataModule
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl


def main():
    cfg = OmegaConf.load("config/train.yaml")
    cfg_model = OmegaConf.load("config\model.yaml")
    
    selected_model = cfg.model.selected   # "cafnet"
    model_info = cfg_model[selected_model]

    
    datamodule = CrackDataModule(**cfg.data)
    model = CrackModule(model_info.name, model_info.hparams, **cfg.optim)
    
    logger = TensorBoardLogger(**cfg.logger)
    checkpoint_cb = ModelCheckpoint(**cfg.checkpoint)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    
    trainer = pl.Trainer(
        **cfg.trainer,
        logger=logger,
        callbacks=[checkpoint_cb, lr_monitor]
    )
