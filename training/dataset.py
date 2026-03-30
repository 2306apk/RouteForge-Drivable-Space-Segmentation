import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A

class DrivableDataset(Dataset):
    def __init__(self, img_dir, mask_dir, augment=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.files = sorted(os.listdir(mask_dir))
        self.augment = augment

        if self.augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=10, p=0.5, border_mode=cv2.BORDER_REFLECT), # Reduced rotation, changed border_mode
                A.RandomBrightnessContrast(p=0.2),
                A.ShiftScaleRotate(shift_limit=0.04, scale_limit=0.08, rotate_limit=5, p=0.3, border_mode=cv2.BORDER_REFLECT) # Reduced limits, changed border_mode
            ])
        else:
            self.transform = None

        print("Total masks:", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]

        img_name = name.replace(".png", ".jpg")

        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, name)

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)

        if img is None or mask is None:
            # Handle cases where image or mask might be missing by skipping or returning None
            # For simplicity, let's just skip for now and rely on initial dataset checks
            # In a real scenario, you might want to log this or filter files upfront
            return self.__getitem__((idx + 1) % len(self.files))

        # Resize images to a consistent size before augmentation
        # 🔥 FAST SIZE (KEY FOR FPS)
        img = cv2.resize(img, (96, 96))
        mask = cv2.resize(mask, (96, 96))

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        img = img / 255.0
        mask = mask / 255.0

        img = np.transpose(img, (2, 0, 1))

        return (
            torch.tensor(img, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32)
        )
