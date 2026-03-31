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
# PATHS
# -----------------------------
IMAGE_DIR = os.path.join(ROOT_DIR, "data", "samples", "CAM_FRONT")
MASK_DIR = os.path.join(ROOT_DIR, "masks", "train_map_final")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
DEBUG_DIR = os.path.join(OUTPUTS_DIR, "debug")

BEST_MODEL_PATH = os.path.join(MODELS_DIR, "unet_best.pth")
LAST_MODEL_PATH = os.path.join(MODELS_DIR, "unet_last.pth")


# -----------------------------
# CONFIG
# -----------------------------
IMAGE_SIZE = 128
DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)

BATCH_SIZE = 4
EPOCHS = 20
LR = 1e-4
SEED = 42


# -----------------------------
# REPRODUCIBILITY
# -----------------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# SAVE SAMPLE PREDICTION
# -----------------------------
def save_prediction(model, dataset, epoch):
    model.eval()

    img, mask = dataset[0]

    if isinstance(img, torch.Tensor):
        img_tensor = img.unsqueeze(0).to(DEVICE)
    else:
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(img_tensor)
        pred = torch.sigmoid(pred)[0, 0].detach().cpu().numpy()

    pred_bin = (pred > 0.5).astype(np.uint8) * 255
    pred_bin = cv2.resize(pred_bin, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)

    os.makedirs(DEBUG_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(DEBUG_DIR, f"epoch_{epoch}_pred.png"), pred_bin)

    # Save a quick overlay for the debug sample
    img_np = img.detach().cpu().numpy() if torch.is_tensor(img) else img
    if img_np.shape[0] == 3:
        img_np = np.transpose(img_np, (1, 2, 0))
    img_np = (img_np * 255.0).astype(np.uint8)

    overlay = img_np.copy()
    green = np.zeros_like(overlay)
    green[:, :, 1] = 255

    mask_bool = pred_bin.astype(bool)
    overlay[mask_bool] = (0.45 * overlay[mask_bool] + 0.55 * green[mask_bool]).astype(np.uint8)

    cv2.imwrite(
        os.path.join(DEBUG_DIR, f"epoch_{epoch}_overlay.png"),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    )


# -----------------------------
# TRAIN FUNCTION
# -----------------------------
def train():
    seed_everything(SEED)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)

    print(f"Using device: {DEVICE}")
    print(f"Images found: {len([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])}")
    print(f"Masks folder: {MASK_DIR}")

    # Dataset
    dataset = NuScenesDataset(
        IMAGE_DIR,
        MASK_DIR,
        transform=get_transforms(image_size=IMAGE_SIZE)
    )

    # Train / Val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # Model
    model = UNet().to(DEVICE)

    # Loss
    bce = nn.BCEWithLogitsLoss()
    dice = DiceLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_iou = -1.0

    # -----------------------------
    # TRAIN LOOP
    # -----------------------------
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{EPOCHS}]")

        for images, masks in loop:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            preds = model(images)

            loss = bce(preds, masks) + dice(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(1, len(train_loader))

        # -----------------------------
        # VALIDATION
        # -----------------------------
        model.eval()
        val_iou = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                preds = model(images)
                val_iou += iou_score(preds, masks)

        val_iou /= max(1, len(val_loader))

        print(f"\nEpoch {epoch + 1}")
        print(f"Train Loss: {avg_loss:.4f}")
        print(f"Val IoU: {val_iou:.4f}")

        # Save debug prediction
        save_prediction(model, dataset, epoch + 1)

        # Save last checkpoint every epoch
        torch.save(model.state_dict(), LAST_MODEL_PATH)

        # Save best checkpoint
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"✅ Best model updated: {BEST_MODEL_PATH}")

        # Optional epoch checkpoint
        torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"unet_epoch_{epoch + 1}.pth"))

    print("✅ Training Complete")
    print(f"Best Val IoU: {best_val_iou:.4f}")
    print(f"Best Model: {BEST_MODEL_PATH}")
    print(f"Last Model: {LAST_MODEL_PATH}")


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    train()