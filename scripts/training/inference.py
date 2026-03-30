import os
import sys
import cv2
import torch
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))

if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from model import UNet


# -----------------------------
# PATHS
# -----------------------------
IMAGE_DIR = os.path.join(ROOT_DIR, "data", "samples", "CAM_FRONT")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
MASK_OUT_DIR = os.path.join(OUTPUT_DIR, "masks")
OVERLAY_OUT_DIR = os.path.join(OUTPUT_DIR, "overlays")

BEST_MODEL_PATH = os.path.join(MODELS_DIR, "unet_best.pth")
LAST_MODEL_PATH = os.path.join(MODELS_DIR, "unet_last.pth")
FALLBACK_EPOCH_PATH = os.path.join(MODELS_DIR, "unet_epoch_20.pth")

IMAGE_SIZE = 128
MASK_THRESHOLD = 0.5
ALPHA = 0.50


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
def preprocess(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    img_resized = img_resized.astype(np.float32) / 255.0
    img_resized = np.transpose(img_resized, (2, 0, 1))
    return torch.tensor(img_resized, dtype=torch.float32).unsqueeze(0)


def postprocess_mask(prob_map):
    mask = (prob_map >= 0.5).astype(np.uint8)

    h, w = mask.shape

    # -----------------------------
    # 1. REMOVE TOP REGION (sky/buildings)
    # -----------------------------
    mask[:int(0.35 * h), :] = 0

    # -----------------------------
    # 2. STRONG MORPH CLEANING
    # -----------------------------
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # -----------------------------
    # 3. REMOVE SMALL BLOBS (people, cones)
    # -----------------------------
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    min_area = 2000  # tune this if needed

    clean = np.zeros_like(mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        # Keep only large regions
        if area > min_area:
            clean[labels == i] = 1

    mask = clean

    return mask


def keep_bottom_connected_component(mask):
    h, w = mask.shape

    num_labels, labels = cv2.connectedComponents(mask)

    if num_labels <= 1:
        return mask

    # bottom-center seed
    seed_x = w // 2

    for y in range(h - 1, int(h * 0.5), -1):
        if mask[y, seed_x] == 1:
            seed_label = labels[y, seed_x]
            return (labels == seed_label).astype(np.uint8)

    return mask


def make_overlay(img, mask):
    overlay = img.copy().astype(np.float32)

    green = np.zeros_like(overlay)
    green[:, :, 1] = 255

    mask_bool = mask.astype(bool)

    overlay[mask_bool] = (
        0.6 * overlay[mask_bool] + 0.4 * green[mask_bool]
    )

    return overlay.astype(np.uint8)


def load_model():
    if os.path.exists(BEST_MODEL_PATH):
        model_path = BEST_MODEL_PATH
    elif os.path.exists(LAST_MODEL_PATH):
        model_path = LAST_MODEL_PATH
    elif os.path.exists(FALLBACK_EPOCH_PATH):
        model_path = FALLBACK_EPOCH_PATH
    else:
        raise FileNotFoundError(
            f"No model checkpoint found.\n"
            f"Checked:\n"
            f"  {BEST_MODEL_PATH}\n"
            f"  {LAST_MODEL_PATH}\n"
            f"  {FALLBACK_EPOCH_PATH}"
        )

    model = UNet().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    print(f"Loaded model: {model_path}")
    return model


# -----------------------------
# MAIN RUN
# -----------------------------
def run():
    os.makedirs(MASK_OUT_DIR, exist_ok=True)
    os.makedirs(OVERLAY_OUT_DIR, exist_ok=True)

    if not os.path.isdir(IMAGE_DIR):
        raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")

    model = load_model()

    images = sorted([
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    print(f"Using device: {device}")
    print(f"Images: {len(images)}")

    for name in images:
        img_path = os.path.join(IMAGE_DIR, name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipping unreadable image: {img_path}")
            continue

        orig = img.copy()
        h, w = orig.shape[:2]

        x = preprocess(img).to(device)

        with torch.no_grad():
            pred = model(x)

        prob = torch.sigmoid(pred)[0, 0].detach().cpu().numpy()

        # Resize probability map back to original resolution
        prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)

        # Binary mask
        mask_bin = postprocess_mask(prob)

        # Keep main drivable connected region from bottom center
        mask_bin = keep_bottom_connected_component(mask_bin)

        # Optional final clean-up after connected component selection
        kernel = np.ones((3, 3), np.uint8)
        mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel)

        overlay = make_overlay(orig, mask_bin)

        base = os.path.splitext(name)[0]
        mask_path = os.path.join(MASK_OUT_DIR, base + ".png")
        overlay_path = os.path.join(OVERLAY_OUT_DIR, base + ".jpg")

        cv2.imwrite(mask_path, (mask_bin * 255).astype(np.uint8))
        cv2.imwrite(overlay_path, overlay)

    print("✅ DONE")
    print(f"Masks saved to: {MASK_OUT_DIR}")
    print(f"Overlays saved to: {OVERLAY_OUT_DIR}")


if __name__ == "__main__":
    run()