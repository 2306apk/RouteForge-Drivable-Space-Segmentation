import os
import cv2
import torch
import numpy as np
from tqdm import tqdm

from .model import UNet


# -----------------------------
# CONFIG
# -----------------------------
IMAGE_SIZE = 128              # change via CLI
MASK_THRESHOLD = 0.48         # slight improvement over 0.5

MODEL_PATH_128 = "models/unet_best.pth"
MODEL_PATH_256 = "models/unet_best_256x512.pth"

IMAGE_DIR = "data/images"
OUTPUT_MASK_DIR = "outputs/masks"
OUTPUT_OVERLAY_DIR = "outputs/overlays"


# -----------------------------
# DEVICE
# -----------------------------
device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)


# -----------------------------
# HELPERS
# -----------------------------
def preprocess(img):
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return torch.tensor(img).unsqueeze(0)


def postprocess(pred):
    pred = pred.squeeze().cpu().numpy()
    mask = (pred > MASK_THRESHOLD).astype(np.uint8)
    return mask


def refine_with_color(mask, img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 0, 50])
    upper = np.array([180, 60, 255])

    road_mask = cv2.inRange(hsv, lower, upper)
    road_mask = road_mask // 255

    return mask * road_mask


def perspective_constraint(mask):
    h, w = mask.shape
    constrained = np.zeros_like(mask)

    for y in range(h):
        width = int((y / h) * w * 0.9)
        center = w // 2
        left = max(center - width // 2, 0)
        right = min(center + width // 2, w)

        constrained[y, left:right] = mask[y, left:right]

    return constrained


def keep_largest_component(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        return mask

    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest).astype(np.uint8)


def remove_small_objects(mask, min_size=500):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            cleaned[labels == i] = 1

    return cleaned


def safe_obstacle_removal(mask):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def overlay_mask(img, mask):
    overlay = img.copy()
    color = np.zeros_like(img)
    color[:, :] = [0, 255, 0]

    mask_3ch = np.stack([mask]*3, axis=-1)
    overlay = np.where(mask_3ch, cv2.addWeighted(img, 0.5, color, 0.5, 0), img)

    return overlay


# -----------------------------
# LOAD MODEL
# -----------------------------
def load_model():
    model = UNet()
    model.to(device)

    if IMAGE_SIZE == 128:
        path = MODEL_PATH_128
    else:
        path = MODEL_PATH_256

    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()

    print(f"Loaded model: {path}")
    print(f"Using device: {device}")

    return model


# -----------------------------
# INFERENCE PIPELINE
# -----------------------------
def process_image(model, img_path):
    img = cv2.imread(img_path)
    orig = img.copy()

    h, w = img.shape[:2]

    inp = preprocess(img).to(device)

    with torch.no_grad():
        pred = model(inp)
        pred = torch.sigmoid(pred)

    mask = postprocess(pred)

    # resize back
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # ---- YOUR BEST PIPELINE ----
    mask = refine_with_color(mask, orig)
    mask = perspective_constraint(mask)
    mask = keep_largest_component(mask)
    mask = remove_small_objects(mask)
    mask = safe_obstacle_removal(mask)
    # ---------------------------

    overlay = overlay_mask(orig, mask)

    return mask, overlay


# -----------------------------
# MAIN
# -----------------------------
def run():
    os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)
    os.makedirs(OUTPUT_OVERLAY_DIR, exist_ok=True)

    model = load_model()

    image_paths = [
        os.path.join(IMAGE_DIR, f)
        for f in os.listdir(IMAGE_DIR)
        if f.endswith((".jpg", ".png"))
    ]

    print(f"Images: {len(image_paths)}")

    for i, path in enumerate(tqdm(image_paths)):
        mask, overlay = process_image(model, path)

        name = os.path.basename(path)

        cv2.imwrite(os.path.join(OUTPUT_MASK_DIR, name), mask * 255)
        cv2.imwrite(os.path.join(OUTPUT_OVERLAY_DIR, name), overlay)

        if i % 50 == 0:
            print(f"Processed {i}/{len(image_paths)}")

    print("DONE")
    print(f"Masks saved to: {OUTPUT_MASK_DIR}")
    print(f"Overlays saved to: {OUTPUT_OVERLAY_DIR}")


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=128, choices=[128, 256])

    args = parser.parse_args()

    IMAGE_SIZE = args.size

    run()
