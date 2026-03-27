import os
import cv2
import numpy as np
from generate_mask_hybrid import generate_mask

# -----------------------------
# PATHS
# -----------------------------
SRC = "data/samples/CAM_FRONT"
DST_IMG = "data/train/images"
DST_MASK = "data/train/masks"

os.makedirs(DST_IMG, exist_ok=True)
os.makedirs(DST_MASK, exist_ok=True)

# -----------------------------
# SETTINGS
# -----------------------------
MAX_IMAGES = 800   # increase for better training

# -----------------------------
# BUILD DATASET
# -----------------------------
files = sorted(os.listdir(SRC))

count = 0
skipped = 0

for f in files:
    if count >= MAX_IMAGES:
        break

    src_path = os.path.join(SRC, f)

    # -----------------------------
    # READ IMAGE
    # -----------------------------
    img = cv2.imread(src_path)

    if img is None:
        skipped += 1
        continue

    # -----------------------------
    # GENERATE MASK
    # -----------------------------
    mask = generate_mask(img)

    # -----------------------------
    # FILTER BAD MASKS (VERY IMPORTANT)
    # -----------------------------
    mask_ratio = np.mean(mask) / 255.0

    if mask_ratio < 0.03:
        skipped += 1
        continue

    if mask_ratio > 0.65:
        skipped += 1
        continue

    # -----------------------------
    # SAVE
    # -----------------------------
    img_name = f"img{count}.png"
    mask_name = f"mask{count}.png"

    cv2.imwrite(os.path.join(DST_IMG, img_name), img)
    cv2.imwrite(os.path.join(DST_MASK, mask_name), mask)

    print(f"Saved {img_name}")

    count += 1

# -----------------------------
# SUMMARY
# -----------------------------
print("\n==========================")
print(f"✅ Dataset built")
print(f"Images saved: {count}")
print(f"Skipped: {skipped}")
print("==========================")