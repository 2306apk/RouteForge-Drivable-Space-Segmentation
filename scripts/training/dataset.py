import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class NuScenesDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        if not os.path.isdir(mask_dir):
            raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

        valid_exts = (".jpg", ".jpeg", ".png")
        self.images = sorted(
            [f for f in os.listdir(image_dir) if f.lower().endswith(valid_exts)]
        )

        if len(self.images) == 0:
            raise ValueError(f"No images found in: {image_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)

        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Image not found or unreadable: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Mask not found or unreadable: {mask_path}")

        # Binary mask: drivable area = 1, everything else = 0
        mask = (mask > 127).astype(np.float32)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            # Fallback resize if no transform is provided
            image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_NEAREST)

            image = image.astype(np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1))

        # Ensure tensor format
        if isinstance(image, np.ndarray):
            image = torch.tensor(image, dtype=torch.float32)

        if isinstance(mask, np.ndarray):
            if mask.ndim == 2:
                mask = np.expand_dims(mask, axis=0)
            mask = torch.tensor(mask, dtype=torch.float32)
        elif torch.is_tensor(mask):
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            mask = mask.float()

        return image, mask