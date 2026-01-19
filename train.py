from models.CrackModule import CrackModule
from data.pldatamodule import CrackDataModule
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from pathlib import Path

def main():
    BASE_DIR = Path(__file__).resolve().parent
    cfg = OmegaConf.load(BASE_DIR / "config" / "train.yaml")
    cfg_model = OmegaConf.load(BASE_DIR / "config" / "model.yaml")
    
    selected_model = cfg.model.selected   # "cafnet"
    model_info = cfg_model[selected_model]

    datamodule = CrackDataModule(**cfg.data)
    model = CrackModule(model_info.name, model_info.hparams, **cfg.optim)
    
    logger = TensorBoardLogger(**cfg.logger)
    checkpoint_cb = ModelCheckpoint(**cfg.checkpoint)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    early_stop_cb = EarlyStopping(**cfg.early_stopping)

    
    trainer = pl.Trainer(
        **cfg.trainer,
        enable_progress_bar=True,
        logger=logger,
        callbacks=[checkpoint_cb, lr_monitor, early_stop_cb, TQDMProgressBar(refresh_rate=10)]
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.validate(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)

if __name__ == "__main__":
    main()