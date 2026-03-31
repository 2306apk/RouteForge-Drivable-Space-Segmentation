import cv2
import torch
import numpy as np
from train_model import UNet

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "models/road_seg.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 128

# -----------------------------
# LOAD MODEL
# -----------------------------
model = UNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("✅ Model loaded")

# -----------------------------
# VIDEO SOURCE
# -----------------------------
cap = cv2.VideoCapture("test_video.mp4")   # or 0 for webcam

if not cap.isOpened():
    print("❌ Error opening video")
    exit()

# -----------------------------
# INFERENCE LOOP
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    orig = frame.copy()
    h, w, _ = frame.shape

    # -----------------------------
    # PREPROCESS
    # -----------------------------
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # -----------------------------
    # PREDICTION
    # -----------------------------
    with torch.no_grad():
        pred = model(img)
        pred = torch.sigmoid(pred).squeeze().cpu().numpy()

    # Resize back
    pred = cv2.resize(pred, (w, h))

    # Threshold
    mask = (pred > 0.5).astype(np.uint8) * 255

    # -----------------------------
    # REMOVE CAR HOOD (STRONG FIX)
    # -----------------------------
    mask[int(0.8 * h):, :] = 0   # remove bottom 20%

    roi = np.zeros_like(mask)

    pts = np.array([
        [int(0.1 * w), int(0.8 * h)],
        [int(0.9 * w), int(0.8 * h)],
        [int(0.6 * w), int(0.55 * h)],
        [int(0.4 * w), int(0.55 * h)]
    ])

    cv2.fillPoly(roi, [pts], 255)
    mask = cv2.bitwise_and(mask, roi)

    # -----------------------------
    # SMOOTH MASK
    # -----------------------------
    kernel = np.ones((11, 11), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (9, 9), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # -----------------------------
    # COLOR OVERLAY
    # -----------------------------
    overlay = orig.copy()
    overlay[mask > 0] = [0, 255, 0]

    output = cv2.addWeighted(orig, 0.7, overlay, 0.3, 0)

    # -----------------------------
    # DISPLAY
    # -----------------------------
    cv2.imshow("Drivable Area", output)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()