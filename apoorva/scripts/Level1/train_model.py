import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================
# CONFIG
# =========================
IMG_DIR = "data/train/images"
MASK_DIR = "data/train/masks"

IMG_SIZE = 96
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# DATASET
# =========================
class RoadDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.files = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]

        img = cv2.imread(os.path.join(self.img_dir, name))
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        img = np.transpose(img, (2, 0, 1))

        mask = cv2.imread(os.path.join(self.mask_dir, name), 0)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
        mask = mask / 255.0
        mask = np.expand_dims(mask, axis=0)

        return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)


# =========================
# UNET MODEL
# =========================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.d1 = DoubleConv(3, 64)
        self.d2 = DoubleConv(64, 128)
        self.d3 = DoubleConv(128, 256)

        self.pool = nn.MaxPool2d(2)

        self.u1 = DoubleConv(256 + 128, 128)
        self.u2 = DoubleConv(128 + 64, 64)

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        c1 = self.d1(x)
        p1 = self.pool(c1)

        c2 = self.d2(p1)
        p2 = self.pool(c2)

        c3 = self.d3(p2)

        u1 = self.up(c3)
        u1 = torch.cat([u1, c2], dim=1)
        u1 = self.u1(u1)

        u2 = self.up(u1)
        u2 = torch.cat([u2, c1], dim=1)
        u2 = self.u2(u2)

        return torch.sigmoid(self.out(u2))


# =========================
# LOSS
# =========================
def dice_loss(pred, target, smooth=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


# =========================
# TRAIN
# =========================
dataset = RoadDataset(IMG_DIR, MASK_DIR)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = UNet().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

bce = nn.BCELoss()

print("🚀 Training started...")

for epoch in range(EPOCHS):
    total_loss = 0

    for imgs, masks in loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

        preds = model(imgs)

        loss = bce(preds, masks) + dice_loss(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

# =========================
# SAVE MODEL
# =========================
torch.save(model.state_dict(), "models/unet.pth")
print("✅ Model saved!")