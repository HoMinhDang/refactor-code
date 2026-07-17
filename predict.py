import argparse
from pathlib import Path

import torchvision.utils as vutils
import pytorch_lightning as pl
from omegaconf import OmegaConf

from models.crack_module import CrackModule
from data.crack_data_module import CrackDataModule


class SavePredictionsCallback(pl.Callback):
    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = Path(output_dir)

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        _, _, filenames = batch
        for pred, fname in zip(outputs, filenames):
            name = Path(fname).stem
            vutils.save_image(pred.float(), self.output_dir / f"{name}_pred.png")


def parse_args():
    p = argparse.ArgumentParser(description="Run prediction and save mask images")
    p.add_argument("--ckpt", required=True, help="Path to checkpoint .ckpt")
    p.add_argument("--data", required=True, help="Root dir of dataset (contains test/IMG, test/GT)")
    p.add_argument("--output", default="predict_results", help="Output directory for predicted masks")
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

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        callbacks=[SavePredictionsCallback(output_dir)],
        enable_progress_bar=True,
    )

    trainer.predict(model, datamodule=datamodule, return_predictions=False)
    print(f"\nPredictions saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
