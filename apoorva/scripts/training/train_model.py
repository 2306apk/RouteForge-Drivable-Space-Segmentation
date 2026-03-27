import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from pathlib import Path

# -----------------------------
# PATHS
# -----------------------------
DATA_ROOT = Path("data/train_clean")
IMG_DIR = DATA_ROOT / "images"
MASK_DIR = DATA_ROOT / "masks"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 8
EPOCHS = 30
LR = 1e-3
IMG_SIZE = 128


# -----------------------------
# DATASET
# -----------------------------
class RoadDataset(Dataset):
    def __init__(self):
        self.files = sorted(os.listdir(IMG_DIR))

        self.img_tf = T.Compose([
            T.ToPILImage(),
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]

        img_path = IMG_DIR / name
        mask_path = MASK_DIR / name

        img = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), 0)

        if img is None or mask is None:
            raise ValueError(f"Missing file: {name}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # -------- IMAGE --------
        img = self.img_tf(img)

        # -------- MASK (IMPORTANT FIX) --------
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        mask = torch.tensor(mask, dtype=torch.float32) / 255.0
        mask = mask.unsqueeze(0)  # (1, H, W)

        return img, mask


# -----------------------------
# MODEL
# -----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.d1 = DoubleConv(3, 32)
        self.p1 = nn.MaxPool2d(2)

        self.d2 = DoubleConv(32, 64)
        self.p2 = nn.MaxPool2d(2)

        self.bn = DoubleConv(64, 128)

        self.u1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.c1 = DoubleConv(128, 64)

        self.u2 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.c2 = DoubleConv(64, 32)

        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        d1 = self.d1(x)
        p1 = self.p1(d1)

        d2 = self.d2(p1)
        p2 = self.p2(d2)

        bn = self.bn(p2)

        u1 = self.u1(bn)
        u1 = torch.cat([u1, d2], dim=1)
        u1 = self.c1(u1)

        u2 = self.u2(u1)
        u2 = torch.cat([u2, d1], dim=1)
        u2 = self.c2(u2)

        return self.out(u2)


# -----------------------------
# SAVE PREDICTIONS (CRITICAL)
# -----------------------------
def save_prediction(model, dataset, epoch):
    model.eval()

    img, _ = dataset[0]
    img = img.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(img)
        pred = torch.sigmoid(pred)[0, 0].cpu().numpy()

    pred = (pred > 0.5).astype("uint8") * 255

    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(f"outputs/epoch_{epoch}.png", pred)


# -----------------------------
# TRAIN
# -----------------------------
dataset = RoadDataset()
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = UNet().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.BCEWithLogitsLoss()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}: Loss {avg_loss:.4f}")

    # 🔥 SAVE OUTPUT EVERY EPOCH
    save_prediction(model, dataset, epoch + 1)


# -----------------------------
# SAVE MODEL
# -----------------------------
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/road_seg.pth")

print("✅ Model saved")