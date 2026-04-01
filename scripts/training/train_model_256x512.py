import os
import sys
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))

if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from dataset import NuScenesDataset
from model import UNet
from utils import get_transforms, DiceLoss, iou_score


# -----------------------------
# PATHS (UNCHANGED)
# -----------------------------
IMAGE_DIR = os.path.join(ROOT_DIR, "data", "samples", "CAM_FRONT")
MASK_DIR = os.path.join(ROOT_DIR, "masks", "train_map_final")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")

# 👉 NEW MODEL NAMES (SAFE)
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "unet_best_256x512.pth")
LAST_MODEL_PATH = os.path.join(MODELS_DIR, "unet_last_256x512.pth")


# -----------------------------
# CONFIG
# -----------------------------
IMG_H = 256
IMG_W = 512

DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)

BATCH_SIZE = 2   # ⚠️ reduce (higher resolution)
EPOCHS = 20
LR = 1e-4
SEED = 42


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# TRAIN
# -----------------------------
def train():
    seed_everything(SEED)

    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"Using device: {DEVICE}")

    dataset = NuScenesDataset(
        IMAGE_DIR,
        MASK_DIR,
        transform=get_transforms(image_size=(IMG_H, IMG_W))  # 🔥 key change
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = UNet().to(DEVICE)

    bce = nn.BCEWithLogitsLoss()
    dice = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_iou = -1

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")

        for images, masks in loop:
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            preds = model(images)
            loss = bce(preds, masks) + dice(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)

        # VALIDATION
        model.eval()
        val_iou = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                preds = model(images)
                val_iou += iou_score(preds, masks)

        val_iou /= len(val_loader)

        print(f"\nEpoch {epoch+1}")
        print(f"Loss: {avg_loss:.4f}")
        print(f"Val IoU: {val_iou:.4f}")

        # SAVE LAST
        torch.save(model.state_dict(), LAST_MODEL_PATH)

        # SAVE BEST
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print("✅ Best model saved (256x512)")

    print("✅ DONE TRAINING 256x512")


if __name__ == "__main__":
    train()
