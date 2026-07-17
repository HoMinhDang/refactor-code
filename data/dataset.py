from PIL import Image
from torch.utils.data import Dataset
import os
import numpy as np
import torch

class CrackDataset(Dataset):
    def __init__(self, img_dir, mask_dir, base_transform=None, pert_transform=None,
                 tensor_transform=None, contrastive=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.base_transform = base_transform
        self.pert_transform = pert_transform
        self.tensor_transform = tensor_transform
        self.contrastive = contrastive
        self.img_names = os.listdir(img_dir)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]

        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.base_transform:
            augmented = self.base_transform(image=img, mask=mask)
            img_clean = augmented['image']
            mask = augmented['mask']
        else:
            img_clean = img

        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).float()
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)

        if self.contrastive and self.pert_transform:
            img_pert = self.pert_transform(image=img_clean)['image']
            if self.tensor_transform:
                img_clean = self.tensor_transform(image=img_clean)['image']
                img_pert = self.tensor_transform(image=img_pert)['image']
            return img_clean, img_pert, mask, img_name

        if self.tensor_transform:
            img_clean = self.tensor_transform(image=img_clean)['image']
        return img_clean, mask, img_name