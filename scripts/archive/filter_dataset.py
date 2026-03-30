import os
import cv2
import numpy as np
from pathlib import Path

# 🔥 FIXED SOURCE
DATA_ROOT = Path("data/train_auto")

IMG_DIR = DATA_ROOT / "images"
MASK_DIR = DATA_ROOT / "masks"

OUT_ROOT = Path("data/train_clean")
OUT_IMG = OUT_ROOT / "images"
OUT_MASK = OUT_ROOT / "masks"

OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_MASK.mkdir(parents=True, exist_ok=True)


def is_valid_mask(mask):
    h, w = mask.shape

    mask = (mask > 0).astype(np.uint8)

    # -------------------------
    # 1. area check
    # -------------------------
    area_ratio = mask.sum() / (h * w)

    if area_ratio < 0.03:
        return False
    if area_ratio > 0.6:
        return False

    # -------------------------
    # 2. bottom presence
    # -------------------------
    bottom = mask[int(h * 0.7):, :]
    if bottom.sum() < 0.01 * (h * w):
        return False

    # -------------------------
    # 3. connected components
    # -------------------------
    num_labels, labels = cv2.connectedComponents(mask)

    if num_labels > 8:
        return False

    # -------------------------
    # 4. largest component
    # -------------------------
    sizes = [(labels == i).sum() for i in range(1, num_labels)]

    if len(sizes) == 0:
        return False

    largest = max(sizes)

    if largest < 0.02 * (h * w):
        return False

    return True


def main():
    kept = 0
    total = 0

    for name in sorted(os.listdir(IMG_DIR)):
        img_path = IMG_DIR / name
        mask_path = MASK_DIR / name

        if not mask_path.exists():
            continue

        mask = cv2.imread(str(mask_path), 0)

        total += 1

        if not is_valid_mask(mask):
            continue

        cv2.imwrite(str(OUT_IMG / name), cv2.imread(str(img_path)))
        cv2.imwrite(str(OUT_MASK / name), mask)

        kept += 1

    print(f"✅ Kept {kept}/{total}")


if __name__ == "__main__":
    main()