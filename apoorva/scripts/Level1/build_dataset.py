import os
import cv2

# -----------------------------
# CONFIG
# -----------------------------
SRC_DIR = "data/samples/CAM_FRONT"
OUT_IMG_DIR = "data/train/images"
OUT_MASK_DIR = "data/train/masks"   # empty for now
IMG_SIZE = 128

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_MASK_DIR, exist_ok=True)

# -----------------------------
# PROCESS IMAGES
# -----------------------------
image_files = sorted(os.listdir(SRC_DIR))

print(f"Total images found: {len(image_files)}")

count = 0

for img_name in image_files:

    img_path = os.path.join(SRC_DIR, img_name)

    img = cv2.imread(img_path)
    if img is None:
        continue

    # Resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Save
    out_name = f"{count}.png"
    cv2.imwrite(os.path.join(OUT_IMG_DIR, out_name), img)

    # TEMP MASK (all zeros for now)
    mask = 255 * (img[:, :, 0] > -1).astype("uint8")  # dummy
    cv2.imwrite(os.path.join(OUT_MASK_DIR, out_name), mask)

    count += 1

print(f"Dataset built with {count} images")