import os
import cv2
import numpy as np
from pathlib import Path

SRC_IMG = Path("data/train_good/images")
DST_ROOT = Path("data/train_auto")

DST_IMG = DST_ROOT / "images"
DST_MASK = DST_ROOT / "masks"

DST_IMG.mkdir(parents=True, exist_ok=True)
DST_MASK.mkdir(parents=True, exist_ok=True)


def generate_mask(img):
    h, w, _ = img.shape

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # tighter road color
    lower = np.array([0, 0, 60])
    upper = np.array([180, 50, 180])
    mask = cv2.inRange(hsv, lower, upper)

    # -------------------------
    # CENTER BIAS (IMPORTANT)
    # -------------------------
    center_mask = np.zeros_like(mask)
    center_mask[:, int(w * 0.2):int(w * 0.8)] = 255
    mask = cv2.bitwise_and(mask, center_mask)

    # -------------------------
    # BOTTOM REGION
    # -------------------------
    roi = np.zeros_like(mask)
    roi[int(h * 0.5):, :] = 255
    mask = cv2.bitwise_and(mask, roi)

    # -------------------------
    # MORPHOLOGY
    # -------------------------
    kernel = np.ones((11, 11), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # -------------------------
    # KEEP LARGEST COMPONENT ONLY
    # -------------------------
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.where(labels == largest, 255, 0).astype(np.uint8)

    return mask


def main():
    count = 0

    for name in os.listdir(SRC_IMG):
        img_path = SRC_IMG / name
        img = cv2.imread(str(img_path))

        if img is None:
            continue

        mask = generate_mask(img)

        cv2.imwrite(str(DST_IMG / name), img)
        cv2.imwrite(str(DST_MASK / name), mask)

        count += 1

    print(f"✅ Generated {count} FINAL masks")


if __name__ == "__main__":
    main()