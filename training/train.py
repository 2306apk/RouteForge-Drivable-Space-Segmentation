import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import time
from torch.utils.data import DataLoader
from data_preparation.dataset import DrivableDataset
from training.model import UNet
from training.loss import combined_loss


# ================= SETTINGS =================
IMG_DIR = "Dataset/samples/CAM_FRONT"
MASK_DIR = "data_preparation/outputs/masks"  # Must match generate_masks_map.py output

BATCH_SIZE = 2          # small for CPU
EPOCHS = 25
LR = 1e-3

# ================= DEVICE =================
device = torch.device("cpu")
print("Using device:", device)

# ================= DATA =================
dataset = DrivableDataset(IMG_DIR, MASK_DIR)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0   # IMPORTANT for Windows/CPU
)

# ================= MODEL =================
model = UNet().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ================= TRAIN =================
for epoch in range(EPOCHS):

    start_time = time.time()

    total_loss = 0
    total_iou = 0
    count = 0

    model.train()

    for imgs, masks in loader:

        imgs = imgs.to(device)
        masks = masks.to(device)

        preds = model(imgs)

        # ===== LOSS =====
        loss = combined_loss(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # ===== IoU =====
        preds_bin = (preds > 0.5).float()

        intersection = (preds_bin * masks).sum()
        union = preds_bin.sum() + masks.sum() - intersection

        iou = (intersection / (union + 1e-6)).item()

        total_iou += iou
        count += 1

    epoch_time = time.time() - start_time

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Loss: {total_loss:.4f} | "
          f"IoU: {total_iou / count:.4f} | "
          f"Time: {epoch_time:.2f}s")

# ================= SAVE =================
torch.save(model.state_dict(), "model.pth")
print("Model saved as model.pth")