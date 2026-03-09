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

    # chọn model
    selected_model = cfg.model.selected
    model_info = cfg_model[selected_model]

    # datamodule
    datamodule = CrackDataModule(**cfg.data)

    # model
    model = CrackModule(
        model_info.name,
        model_info.hparams,
        **cfg.optim
    )

    # logger
    logger = TensorBoardLogger(**cfg.logger)

    # callbacks
    checkpoint_cb = ModelCheckpoint(**cfg.checkpoint)

    early_stop_cb = EarlyStopping(**cfg.early_stopping)

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    progress_bar = TQDMProgressBar(refresh_rate=10)

    # trainer for training
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

    # ====================
    # TRAIN
    # ====================

    trainer.fit(model, datamodule=datamodule)

    print("\nBest checkpoint:")
    print(checkpoint_cb.best_model_path)

    # ====================
    # EVALUATION (single GPU for correct metrics)
    # ====================

    eval_trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision=cfg.trainer.precision,
        logger=False
    )

    # validate best model
    eval_trainer.validate(
        model,
        datamodule=datamodule,
        ckpt_path=checkpoint_cb.best_model_path
    )

    # test best model
    eval_trainer.test(
        model,
        datamodule=datamodule,
        ckpt_path=checkpoint_cb.best_model_path
    )

if __name__ == "__main__":
    main()