import torch
from torch.utils.data import DataLoader, random_split
import time
import os

# PATHS
IMG_DIR = "/content/Dataset/samples/CAM_FRONT"
MASK_DIR = "/content/outputs/masks"

# SETTINGS
BATCH_SIZE = 4 # Changed from 8 to 4
EPOCHS = 25
LR = 5e-4 # Changed from 1e-3 to 5e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Create dataset and split into training and validation sets
dataset = DrivableDataset(IMG_DIR, MASK_DIR, augment=True) # Enable augmentation for training

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

model = UNet().to(device)

# --- MODIFICATION START ---
# Load previously saved model weights if they exist
model_path = "/content/model.pth"
if os.path.exists(model_path):
    print(f"Loading pre-trained model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
else:
    print("No pre-trained model found, starting training from scratch.")
# --- MODIFICATION END ---

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

scaler = torch.amp.GradScaler("cuda")

# TRAIN
for epoch in range(EPOCHS):
    start = time.time()

    model.train() # Set model to training mode
    total_train_loss = 0
    total_train_iou = 0
    train_count = 0

    for imgs, masks in train_loader:
        imgs = imgs.to(device)
        masks = masks.unsqueeze(1).to(device)

        with torch.amp.autocast("cuda"):
            preds = model(imgs)
            loss = combined_loss(preds, masks)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_train_loss += loss.item()

        preds_bin = (preds > 0.5).float()

        intersection = (preds_bin * masks).sum()
        union = preds_bin.sum() + masks.sum() - intersection

        iou = (intersection / (union + 1e-6)).item()

        total_train_iou += iou
        train_count += 1

    # VALIDATION
    model.eval() # Set model to evaluation mode
    total_val_loss = 0
    total_val_iou = 0
    val_count = 0
    with torch.no_grad(): # Disable gradient calculations
        for imgs, masks in val_loader:
            imgs = imgs.to(device)
            masks = masks.unsqueeze(1).to(device)

            preds = model(imgs)
            loss = combined_loss(preds, masks)

            total_val_loss += loss.item()

            preds_bin = (preds > 0.5).float()
            intersection = (preds_bin * masks).sum()
            union = preds_bin.sum() + masks.sum() - intersection
            iou = (intersection / (union + 1e-6)).item()
            total_val_iou += iou
            val_count += 1

    epoch_time = time.time() - start

    # Using len(dataset) for total images processed for overall FPS estimation
    images_processed = len(dataset)
    fps = images_processed / epoch_time

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {total_train_loss/train_count:.4f} | Train IoU: {total_train_iou/train_count:.4f} | "
          f"Val Loss: {total_val_loss/val_count:.4f} | Val IoU: {total_val_iou/val_count:.4f} | "
          f"FPS: {fps:.2f} imgs/sec | "
          f"Time: {epoch_time:.2f}s")

torch.save(model.state_dict(), "/content/model.pth")
print("Model saved!")