import os
import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes

DATA_ROOT = r"C:\Users\Arnav\Projects\HACKATHONS\MAHE Hackathon 2026\Dataset"
OUTPUT_DIR = "outputs/masks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

nusc = NuScenes(version='v1.0-mini', dataroot=DATA_ROOT, verbose=True)

print("Total samples:", len(nusc.sample))

for i, sample in enumerate(nusc.sample):

    print(f"Processing {i}")

    cam_token = sample['data']['CAM_FRONT']
    img_path = nusc.get_sample_data_path(cam_token)

    image = cv2.imread(img_path)

    if image is None:
        print("Skipping invalid image")
        continue

    h, w = image.shape[:2]

    # ROI
    roi = image[int(h*0.4):h, :]

    # LAB conversion
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # Threshold
    _, mask1 = cv2.threshold(L, 140, 255, cv2.THRESH_BINARY_INV)

    # Edge removal
    edges = cv2.Canny(roi, 50, 150)
    mask2 = cv2.bitwise_not(edges)

    mask = cv2.bitwise_and(mask1, mask2)

    # Morphology
    kernel = np.ones((7,7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Full mask
    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[int(h*0.4):h, :] = mask

    # Save
    filename = os.path.basename(img_path)
    save_path = os.path.join(OUTPUT_DIR, filename)

    cv2.imwrite(save_path, full_mask)

    print("Saved:", save_path)

print("DONE")