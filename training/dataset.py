import os
import cv2
import numpy as np
from torch.utils.data import Dataset

class RoadDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(mask_dir))  # important

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]

        # 🔥 IMPORTANT: get corresponding image
        img_path = os.path.join(self.image_dir, name)
        mask_path = os.path.join(self.mask_dir, name)

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)

        img = cv2.resize(img, (256, 256))
        mask = cv2.resize(mask, (256, 256))

        img = img / 255.0
        mask = mask / 255.0

        img = img.transpose(2, 0, 1)

        return img.astype(np.float32), mask.astype(np.float32)