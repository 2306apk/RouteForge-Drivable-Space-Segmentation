import os
import cv2
import numpy as np


# =========================
# ORIGINAL (STABLE) MASK
# =========================
def generate_mask(img):

    h, w, _ = img.shape

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 🔥 KEY FIX: adaptive threshold instead of fixed
    mask = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )

    # -----------------------------
    # ROI
    # -----------------------------
    roi = np.zeros_like(mask)

    pts = np.array([
        [int(0.1 * w), int(0.95 * h)],
        [int(0.9 * w), int(0.95 * h)],
        [int(0.6 * w), int(0.55 * h)],
        [int(0.4 * w), int(0.55 * h)]
    ])

    cv2.fillPoly(roi, [pts], 255)
    mask = cv2.bitwise_and(mask, roi)

    # -----------------------------
    # CLEANING
    # -----------------------------
    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.medianBlur(mask, 5)

    # -----------------------------
    # KEEP LARGEST REGION
    # -----------------------------
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        clean = np.zeros_like(mask)
        clean[labels == largest] = 255
        mask = clean

    return mask


# =========================
# LOAD ONLY CAM_FRONT
# =========================
def get_cam_front_images(input_dir):

    cam_front_dir = os.path.join(input_dir, "CAM_FRONT")

    if not os.path.exists(cam_front_dir):
        print("❌ CAM_FRONT folder not found")
        return []

    files = sorted(os.listdir(cam_front_dir))

    image_paths = []

    for file in files:
        if file.endswith(".jpg") or file.endswith(".png"):
            image_paths.append(os.path.join(cam_front_dir, file))

    return image_paths


# =========================
# DATASET GENERATION
# =========================
def generate_dataset(input_dir, save_img_dir, save_mask_dir):

    print("\n===== USING CAM_FRONT ONLY =====\n")

    image_paths = get_cam_front_images(input_dir)

    print(f"📂 Found {len(image_paths)} CAM_FRONT images")

    if len(image_paths) == 0:
        print("❌ No images found!")
        return

    os.makedirs(save_img_dir, exist_ok=True)
    os.makedirs(save_mask_dir, exist_ok=True)

    for i, path in enumerate(image_paths):

        print(f"➡️ Processing {i}")

        img = cv2.imread(path)

        if img is None:
            continue

        mask = generate_mask(img)

        name = f"img{i}.png"

        cv2.imwrite(os.path.join(save_img_dir, name), img)
        cv2.imwrite(os.path.join(save_mask_dir, name), mask)

    print("\n===== DONE =====")


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    generate_dataset(
        input_dir="data/samples",
        save_img_dir="data/train/images",
        save_mask_dir="data/train/masks"
    )