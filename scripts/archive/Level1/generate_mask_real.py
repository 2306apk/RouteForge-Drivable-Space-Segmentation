import cv2
import numpy as np
import os
from tqdm import tqdm

# Paths
IMAGE_DIR = "data/train/images"
MASK_DIR = "data/train/masks"

os.makedirs(MASK_DIR, exist_ok=True)


def generate_mask(img):
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Road colors: low saturation (gray-ish)
    lower = np.array([0, 0, 50])
    upper = np.array([180, 60, 200])
    hsv_mask = cv2.inRange(hsv, lower, upper)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Better road intensity range
    gray_mask = cv2.inRange(gray, 80, 180)

    # Combine HSV + grayscale
    mask = cv2.bitwise_and(hsv_mask, gray_mask)

    # -------- EDGE REMOVAL --------
    edges = cv2.Canny(gray, 50, 150)
    kernel_edge = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel_edge)

    mask = cv2.bitwise_and(mask, cv2.bitwise_not(edges))

    # -------- ROI (BOTTOM HALF ONLY) --------
    h, w = mask.shape
    roi = np.zeros_like(mask)
    roi[int(h * 0.5):, :] = 255  # keep bottom 50%
    mask = cv2.bitwise_and(mask, roi)

    # -------- MORPHOLOGICAL CLEANUP --------
    kernel = np.ones((5, 5), np.uint8)

    # Fill gaps
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Remove noise
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

    print("✅ Generated all masks")


if __name__ == "__main__":
    main()