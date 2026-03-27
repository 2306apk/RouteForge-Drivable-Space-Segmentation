import os
import cv2
import torch
from torch.utils.data import Dataset

class DrivableDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir

        # Only keep images that have masks
        self.images = [
            f for f in os.listdir(img_dir)
            if os.path.exists(os.path.join(mask_dir, f))
        ]

        print(f"Total valid samples: {len(self.images)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]

        img_path = os.path.join(self.img_dir, name)
        mask_path = os.path.join(self.mask_dir, name)

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)

        if img is None or mask is None:
            raise ValueError(f"Error loading {name}")

        img = cv2.resize(img, (256, 256))
        mask = cv2.resize(mask, (256, 256))

        img = img / 255.0
        mask = mask / 255.0

        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return img, mask