import argparse
from pathlib import Path

import pytorch_lightning as pl
from omegaconf import OmegaConf

from models.crack_module import CrackModule
from data.crack_data_module import CrackDataModule


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a crack segmentation model on a test set")
    p.add_argument("--ckpt", required=True, help="Path to checkpoint .ckpt")
    p.add_argument("--data", required=True, help="Root dir of dataset (contains test/IMG, test/GT)")
    p.add_argument("--model", default=None, help="Model name from model.yaml (default: use selected)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    return p.parse_args()


def main():
    args = parse_args()
    base = Path(__file__).resolve().parent

    cfg = OmegaConf.load(base / "config" / "train.yaml")
    cfg_model = OmegaConf.load(base / "config" / "model.yaml")

    model_name = args.model or cfg.model.selected
    model_info = cfg_model[model_name]

    cfg.data.root_dir = args.data
    cfg.data.batch_size = args.batch_size
    cfg.data.img_size = [args.img_size, args.img_size]

    datamodule = CrackDataModule(**cfg.data, enable_contrastive=False)

    model = CrackModule.load_from_checkpoint(
        args.ckpt,
        model_name=model_info.name,
        model_hparams=model_info.hparams,
        weights_only=False,
    )

    trainer = pl.Trainer(
        accelerator=args.device,
        devices=1,
        logger=False,
        enable_progress_bar=True,
    )

    results = trainer.test(model, datamodule=datamodule)
    print("\n", results)


if __name__ == "__main__":
    main()
