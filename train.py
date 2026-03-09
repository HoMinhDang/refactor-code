import torch
import pytorch_lightning as pl

from pathlib import Path
from omegaconf import OmegaConf

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
    # TRAINER (TRAIN)
    # =========================
    trainer = pl.Trainer(
        **cfg.trainer,
        logger=logger,
        callbacks=[
            checkpoint_cb,
            early_stop_cb,
            lr_monitor,
            progress_bar
        ],
        enable_progress_bar=True
    )

    # =========================
    # TRAIN
    # =========================
    trainer.fit(model, datamodule=datamodule)

    # =========================
    # BEST CHECKPOINT
    # =========================
    best_ckpt = checkpoint_cb.best_model_path

    print("\nBest checkpoint:")
    print(best_ckpt)

    # =========================
    # EXPORT BEST WEIGHTS (.pth)
    # =========================
    best_model = CrackModule.load_from_checkpoint(best_ckpt)

    pth_path = best_ckpt.replace(".ckpt", ".pth")

    torch.save(best_model.model.state_dict(), pth_path)

    print("Saved weights:", pth_path)

    # =========================
    # EVALUATION TRAINER
    # =========================
    eval_trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision=cfg.trainer.precision,
        logger=False
    )

    # =========================
    # VALIDATE BEST MODEL
    # =========================
    eval_trainer.validate(
        best_model,
        datamodule=datamodule
    )

    # =========================
    # TEST BEST MODEL
    # =========================
    eval_trainer.test(
        best_model,
        datamodule=datamodule
    )


if __name__ == "__main__":
    main()