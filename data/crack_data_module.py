import pytorch_lightning as pl
from data.dataset import CrackDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import os

class CrackDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, img_size=(256, 256), batch_size=16, num_workers=4,
                 enable_contrastive=False):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_workers = num_workers
        self.enable_contrastive = enable_contrastive

        mu = [0.51789941, 0.51360926, 0.547762]
        std = [0.1812099, 0.17746663, 0.20386334]

        self.base_transform = A.Compose([
            A.Resize(img_size[0], img_size[1]),
        ])

        self.perturbation_transform = A.Compose([
            A.OneOf([
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
            ], p=0.8),
            A.OneOf([
                A.GaussNoise(p=1.0),
                A.ISONoise(p=1.0),
                A.ImageCompression(quality_lower=60, quality_upper=100, p=1.0),
            ], p=0.5),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=5, p=1.0),
            ], p=0.3),
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(1, 32),
                hole_width_range=(1, 32),
                fill=0,
                p=0.5
            ),
        ])

        self.tensor_transform = A.Compose([
            A.Normalize(mean=mu, std=std),
            ToTensorV2(),
        ])

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = CrackDataset(
                os.path.join(self.root_dir, "train/IMG"),
                os.path.join(self.root_dir, "train/GT"),
                self.base_transform,
                self.perturbation_transform if self.enable_contrastive else None,
                self.tensor_transform,
                contrastive=self.enable_contrastive
            )
            self.val_dataset = CrackDataset(
                os.path.join(self.root_dir, "val/IMG"),
                os.path.join(self.root_dir, "val/GT"),
                self.base_transform,
                None,
                self.tensor_transform,
                contrastive=False
            )

        if stage in ["test", "predict"] or stage is None:
            self.test_dataset = CrackDataset(
                os.path.join(self.root_dir, "test/IMG"),
                os.path.join(self.root_dir, "test/GT"),
                self.base_transform,
                None,
                self.tensor_transform,
                contrastive=False
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

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )