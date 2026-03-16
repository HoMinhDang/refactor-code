import os
import shutil
import torch
import pytorch_lightning as pl
from pathlib import Path
from omegaconf import OmegaConf
import torchvision.utils as vutils

from models.CrackModule import CrackModule
from data.pldatamodule import CrackDataModule

class SavePredictionsCallback(pl.Callback):
    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = Path(output_dir)

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # outputs là kết quả từ predict_step (đã xử lý ở CrackModule)
        preds = outputs
        _, _, filenames = batch 
        
        for i, pred in enumerate(preds):
            original_name = Path(filenames[i]).stem
            save_path = self.output_dir / f"pred_{original_name}.png"
            vutils.save_image(pred.float(), save_path)

def main():
    BASE_DIR = Path(__file__).resolve().parent
    cfg = OmegaConf.load(BASE_DIR / "config" / "train.yaml")
    cfg_model = OmegaConf.load(BASE_DIR / "config" / "model.yaml")

    CKPT_PATH = "PATH.ckpt" # => tự chỉnh
    OUTPUT_DIR = BASE_DIR / "predict_results"
    
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        if OUTPUT_DIR.exists(): shutil.rmtree(OUTPUT_DIR)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    datamodule = CrackDataModule(**cfg.data)
    model_info = cfg_model[cfg.model.selected]
    
    model = CrackModule.load_from_checkpoint(
        CKPT_PATH, 
        model_name=model_info.name,
        hparams=model_info.hparams,
        weights_only=False,
        **cfg.optim
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        logger=False,
        callbacks=[SavePredictionsCallback(OUTPUT_DIR)],
        inference_mode=True
    )

    print("\n--- Running Evaluation ---")
    trainer.test(model, datamodule=datamodule, ckpt_path=CKPT_PATH, weights_only=False)

    print("\n--- Running Prediction ---")
    trainer.predict(model, datamodule=datamodule, ckpt_path=CKPT_PATH, return_predictions=False)

    if trainer.is_global_zero:
        print(f"\n--- Zipping results ---")
        if any(OUTPUT_DIR.iterdir()):
            shutil.make_archive(str(BASE_DIR / "test_predictions"), 'zip', OUTPUT_DIR)
            shutil.rmtree(OUTPUT_DIR)
            print("Finish: test_predictions.zip created.")
        else:
            print("Error: No images found.")

if __name__ == "__main__":
    main()