import torch
import pytorch_lightning as pl

from pathlib import Path
from omegaconf import OmegaConf
from omegaconf import DictConfig

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    TQDMProgressBar
)

from models.CrackModule import CrackModule
from data.pldatamodule import CrackDataModule


def main():

    # =========================
    # LOAD CONFIG
    # =========================
    BASE_DIR = Path(__file__).resolve().parent

    cfg = OmegaConf.load(BASE_DIR / "config" / "train.yaml")
    cfg_model = OmegaConf.load(BASE_DIR / "config" / "model.yaml")

    selected_model = cfg.model.selected
    model_info = cfg_model[selected_model]

    # =========================
    # DATA
    # =========================
    datamodule = CrackDataModule(**cfg.data)

    # =========================
    # MODEL
    # =========================
    model = CrackModule(
        model_info.name,
        model_info.hparams,
        **cfg.optim
    )

    # =========================
    # LOGGER
    # =========================
    logger = TensorBoardLogger(**cfg.logger)

    # =========================
    # CALLBACKS
    # =========================
    checkpoint_cb = ModelCheckpoint(**cfg.checkpoint)

    early_stop_cb = EarlyStopping(**cfg.early_stopping)

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    progress_bar = TQDMProgressBar(refresh_rate=10)

    # =========================
    # TRAINER
    # =========================
    trainer = pl.Trainer(
        **cfg.trainer,
        logger=logger,
        callbacks=[
            checkpoint_cb,
            early_stop_cb,
            lr_monitor,
            progress_bar,
        ],
        enable_progress_bar=True,
    )

    # =========================
    # TRAIN
    # =========================
    trainer.fit(model, datamodule=datamodule)

    # =========================
    # PRINT BEST CHECKPOINT
    # =========================
    if trainer.global_rank == 0:
        print("\nBest checkpoint:")
        print(checkpoint_cb.best_model_path)

    # =========================
    # VALIDATE BEST MODEL
    # =========================
    trainer.validate(
        datamodule=datamodule,
        ckpt_path=checkpoint_cb.best_model_path,
        weights_only=False
    )

    # =========================
    # TEST BEST MODEL
    # =========================
    trainer.test(
        datamodule=datamodule,
        ckpt_path=checkpoint_cb.best_model_path,
        weights_only=False
    )

if __name__ == "__main__":
    main()