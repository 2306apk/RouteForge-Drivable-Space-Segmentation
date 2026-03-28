import csv
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class DrivableDataset(Dataset):
    def __init__(self, csv_path, image_size=(288, 512), is_train=False):
        self.rows = []
        self.h, self.w = image_size
        self.is_train = is_train

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                self.rows.append(r)

        if len(self.rows) == 0:
            raise ValueError(f"No rows found in {csv_path}")

    def __len__(self):
        return len(self.rows)

    def _augment(self, image, mask):
        if random.random() < 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)

        if random.random() < 0.4:
            alpha = random.uniform(0.85, 1.15)
            beta = random.uniform(-20, 20)
            image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        if random.random() < 0.2:
            image = cv2.GaussianBlur(image, (3, 3), 0)

        return image, mask

    def __getitem__(self, idx):
        row = self.rows[idx]
        image_path = row["image_path"]
        mask_path = row["mask_path"]

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        if mask is None:
            raise RuntimeError(f"Could not read mask: {mask_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = (mask > 127).astype(np.uint8)

        image = cv2.resize(image, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.w, self.h), interpolation=cv2.INTER_NEAREST)

        if self.is_train:
            image, mask = self._augment(image, mask)

        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std

        image = np.transpose(image, (2, 0, 1))
        mask = np.expand_dims(mask.astype(np.float32), axis=0)

        return torch.from_numpy(image), torch.from_numpy(mask)