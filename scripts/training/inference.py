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

MASK_DIR = os.path.join(OUTPUT_DIR, "masks")
OVERLAY_DIR = os.path.join(OUTPUT_DIR, "overlays")

BEST_MODEL_PATH = os.path.join(MODELS_DIR, "unet_best.pth")

IMAGE_SIZE = 128
MASK_THRESHOLD = 0.52


# -----------------------------
# DEVICE
# -----------------------------
device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)


# -----------------------------
# PREPROCESS
# -----------------------------
def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return torch.tensor(img).unsqueeze(0)


# -----------------------------
# POSTPROCESS (ORIGINAL CORE)
# -----------------------------
def postprocess_mask(prob):
    mask = (prob > MASK_THRESHOLD).astype(np.uint8)

    h, w = mask.shape

    # remove sky
    mask[:int(0.30 * h), :] = 0

    # small smoothing
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


# -----------------------------
# 🔥 ONLY FIX: NIGHT BOOST
# -----------------------------
def boost_night(mask, prob):
    if np.mean(prob) < 0.08:
        boosted = (prob > 0.3).astype(np.uint8)
        mask = np.maximum(mask, boosted)
    return mask


# -----------------------------
# KEEP MAIN ROAD (ORIGINAL)
# -----------------------------
def keep_largest(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    if num_labels <= 1:
        return mask

    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest).astype(np.uint8)


def keep_bottom(mask):
    h, w = mask.shape

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    bottom = int(h * 0.9)
    ids = np.unique(labels[bottom:, :])
    ids = [i for i in ids if i != 0]

    if not ids:
        return mask

    new_mask = np.zeros_like(mask)
    for i in ids:
        new_mask[labels == i] = 1

    return new_mask


# -----------------------------
# COLOR FILTER (LIGHT ONLY)
# -----------------------------
def refine_color(mask, img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    mask[sat > 80] = 0
    mask[val > 250] = 0

    return mask


# -----------------------------
# CLEANING (ORIGINAL)
# -----------------------------
def remove_small(mask, min_area=800):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    new_mask = np.zeros_like(mask)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > min_area:
            new_mask[labels == i] = 1

    return new_mask


def smooth(mask):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


# -----------------------------
# OVERLAY
# -----------------------------
def overlay(img, mask):
    out = img.copy().astype(np.float32)

    green = np.zeros_like(out)
    green[:, :, 1] = 255

    m = mask.astype(bool)

    out[m] = 0.6 * out[m] + 0.4 * green[m]

    return out.astype(np.uint8)

def remove_tiny_vertical(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    h, w = mask.shape

    new_mask = np.zeros_like(mask)

    for i in range(1, num_labels):
        x, y, bw, bh, area = stats[i]

        aspect = bh / (bw + 1e-5)

        # 🔥 only remove VERY obvious humans
        if aspect > 2.5 and area < 3000:
            continue

        new_mask[labels == i] = 1

    return new_mask


# -----------------------------
# LOAD MODEL
# -----------------------------
def load_model():
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError("Model not found")

    model = UNet().to(device)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.eval()

    print("Loaded:", BEST_MODEL_PATH)
    return model


# -----------------------------
# MAIN
# -----------------------------
def run():
    os.makedirs(MASK_DIR, exist_ok=True)
    os.makedirs(OVERLAY_DIR, exist_ok=True)

    model = load_model()

    images = sorted([
        f for f in os.listdir(IMAGE_DIR)
        if f.endswith((".jpg", ".png"))
    ])

    print("Images:", len(images))

    for name in images:
        path = os.path.join(IMAGE_DIR, name)
        img = cv2.imread(path)

        if img is None:
            continue

        h, w = img.shape[:2]

        x = preprocess(img).to(device)

        with torch.no_grad():
            pred = model(x)

        prob = torch.sigmoid(pred)[0, 0].cpu().numpy()
        prob = cv2.resize(prob, (w, h))

        # -----------------------------
        # FINAL PIPELINE (STABLE)
        # -----------------------------
        mask = postprocess_mask(prob)

        mask = boost_night(mask, prob)

        mask = refine_color(mask, img)

        mask = keep_largest(mask)
        mask = keep_bottom(mask)

        mask = remove_tiny_vertical(mask)   # ⭐ new

        # ⭐ MOVE NIGHT SMOOTHING HERE
        if np.mean(prob) < 0.1:
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        mask = smooth(mask)
        mask = remove_small(mask)

        # -----------------------------
        # OUTPUT
        # -----------------------------
        out = overlay(img, mask)

        base = os.path.splitext(name)[0]

        cv2.imwrite(os.path.join(MASK_DIR, base + ".png"), mask * 255)
        cv2.imwrite(os.path.join(OVERLAY_DIR, base + ".jpg"), out)


    print("DONE ✅")


if __name__ == "__main__":
    run()