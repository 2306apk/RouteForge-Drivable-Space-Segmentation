import time
import torch
import cv2
import numpy as np
import os
from glob import glob

from scripts.training.model import UNet

# -----------------------------
# CONFIG (EDIT ONLY THIS IF NEEDED)
# -----------------------------
MODEL_PATH = "models/unet_best.pth"   # if removed, comment load section
IMAGE_DIR = "data/train"              # <-- YOUR ACTUAL FOLDER
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256

# -----------------------------
# LOAD MODEL
# -----------------------------
model = UNet()

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print("✅ Model loaded")
else:
    print("⚠️ Model not found, running dummy forward")

model.to(DEVICE)
model.eval()

# OPTIONAL SPEED BOOST
if DEVICE == "cuda":
    model.half()

# -----------------------------
# PREPROCESS
# -----------------------------
def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

    if DEVICE == "cuda":
        tensor = tensor.half()

    return tensor

# -----------------------------
# LOAD IMAGES
# -----------------------------
image_paths = glob(os.path.join(IMAGE_DIR, "*.jpg"))[:100]

print("📸 Total images found:", len(image_paths))

if len(image_paths) == 0:
    print("❌ No images found. Check IMAGE_DIR path.")
    exit()

# -----------------------------
# METRICS
# -----------------------------
latencies = []

# -----------------------------
# INFERENCE LOOP
# -----------------------------
for img_path in image_paths:
    img = cv2.imread(img_path)
    if img is None:
        continue

    inp = preprocess(img).to(DEVICE)

    start = time.time()

    with torch.no_grad():
        out = model(inp)
        out = torch.sigmoid(out)

    end = time.time()

    latency = (end - start) * 1000  # ms
    latencies.append(latency)

# -----------------------------
# RESULTS
# -----------------------------
if len(latencies) == 0:
    print("❌ No valid frames processed")
    exit()

avg_latency = np.mean(latencies)
fps = 1000 / avg_latency

print("\n===== PERFORMANCE =====")
print(f"Avg Latency: {avg_latency:.2f} ms")
print(f"FPS: {fps:.2f}")
print("======================\n")