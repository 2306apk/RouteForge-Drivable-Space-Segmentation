import time
import torch
import cv2
import numpy as np

from model import UNet


# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "models/unet_epoch_20.pth"
IMAGE_PATH = "data/samples/CAM_FRONT"


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
IMG_H, IMG_W = 192, 384


# -----------------------------
# LOAD MODEL
# -----------------------------
model = UNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


# -----------------------------
# PREPROCESS
# -----------------------------
def preprocess(img):
    img = cv2.resize(img, (IMG_W, IMG_H))
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    return torch.tensor(img, dtype=torch.float32).unsqueeze(0)


# -----------------------------
# LOAD SAMPLE IMAGE
# -----------------------------
import os
img_name = os.listdir(IMAGE_PATH)[0]
img = cv2.imread(os.path.join(IMAGE_PATH, img_name))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

x = preprocess(img).to(DEVICE)


# -----------------------------
# WARMUP
# -----------------------------
for _ in range(10):
    _ = model(x)


# -----------------------------
# FPS TEST
# -----------------------------
runs = 50
start = time.time()

for _ in range(runs):
    _ = model(x)

end = time.time()

fps = runs / (end - start)

print(f"\n🚀 FPS: {fps:.2f}")