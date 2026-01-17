import pytorch_lightning as pl
from data.dataset import CrackDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import os

class CrackDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, img_size=(256, 256), batch_size=16, num_workers=4):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_workers = num_workers

        mu = [0.51789941, 0.51360926, 0.547762]
        std = [0.1812099, 0.17746663, 0.20386334]

        self.transform = A.Compose([
            A.Resize(img_size[0], img_size[1]),
            A.Normalize(mean=mu, std=std),
            ToTensorV2(),
        ])

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = CrackDataset(
                os.path.join(self.root_dir, "train/IMG"),
                os.path.join(self.root_dir, "train/GT"),
                self.transform,
            )
            self.val_dataset = CrackDataset(
                os.path.join(self.root_dir, "val/IMG"),
                os.path.join(self.root_dir, "val/GT"),
                self.transform,
            )

        if stage == "test" or stage is None:
            self.test_dataset = CrackDataset(
                os.path.join(self.root_dir, "test/IMG"),
                os.path.join(self.root_dir, "test/GT"),
                self.transform,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
