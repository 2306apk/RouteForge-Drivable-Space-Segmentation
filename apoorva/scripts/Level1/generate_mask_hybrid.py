import cv2
import numpy as np
import os
from tqdm import tqdm

IMAGE_DIR = "data/train/images"
MASK_DIR = "data/train/masks"

os.makedirs(MASK_DIR, exist_ok=True)


def generate_mask(img):
    h, w, _ = img.shape

    # -------- HSV MASK --------
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 50])
    upper = np.array([180, 80, 200])
    hsv_mask = cv2.inRange(hsv, lower, upper)

    # -------- GRAYSCALE MASK --------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_mask = cv2.inRange(gray, 80, 180)

    # -------- EDGE DETECTION (LANES) --------
    edges = cv2.Canny(gray, 50, 150)

    # -------- COMBINE --------
    mask = cv2.bitwise_and(hsv_mask, gray_mask)

    # ADD edges (this is key improvement 🔥)
    mask = cv2.bitwise_or(mask, edges)

    # -------- ROI (BOTTOM 60%) --------
    roi = np.zeros_like(mask)
    roi[int(h * 0.4):, :] = 255
    mask = cv2.bitwise_and(mask, roi)

    # -------- MORPH CLEANUP --------
    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def main():
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith((".png", ".jpg"))]

    print(f"Total images: {len(image_files)}")

    for file in tqdm(image_files):
        img_path = os.path.join(IMAGE_DIR, file)
        mask_path = os.path.join(MASK_DIR, file)

        img = cv2.imread(img_path)
        if img is None:
            continue

        mask = generate_mask(img)
        cv2.imwrite(mask_path, mask)

    print("✅ Hybrid masks generated")


if __name__ == "__main__":
    main()