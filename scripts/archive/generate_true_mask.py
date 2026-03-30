import os
import cv2
import numpy as np
from pathlib import Path

IMG_DIR = Path("data/train_good/images")
OUT_MASK_DIR = Path("data/train_good/masks")

OUT_MASK_DIR.mkdir(parents=True, exist_ok=True)


def generate_mask(img):
    h, w, _ = img.shape

    # convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # threshold (road is usually darker)
    _, mask = cv2.threshold(blur, 140, 255, cv2.THRESH_BINARY_INV)

    # keep only bottom region (road prior)
    roi = np.zeros_like(mask)
    roi[int(h * 0.5):, :] = 1
    mask = mask * roi

    # clean noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def main():
    for name in os.listdir(IMG_DIR):
        img_path = IMG_DIR / name
        img = cv2.imread(str(img_path))

        if img is None:
            continue

        mask = generate_mask(img)

        cv2.imwrite(str(OUT_MASK_DIR / name), mask)

    print("✅ Masks regenerated")


if __name__ == "__main__":
    main()