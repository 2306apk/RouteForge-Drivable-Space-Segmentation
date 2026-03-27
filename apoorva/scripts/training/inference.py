import cv2
import torch
import torch.nn as nn
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "models/road_seg.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 128

# -----------------------------
# MODEL (same as training)
# -----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.down1 = DoubleConv(3, 16)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(32, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv1 = DoubleConv(64, 32)

        self.up2 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.conv2 = DoubleConv(32, 16)

        self.final = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)

        d2 = self.down2(p1)
        p2 = self.pool2(d2)

        bn = self.bottleneck(p2)

        u1 = self.up1(bn)
        u1 = torch.cat([u1, d2], dim=1)
        u1 = self.conv1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, d1], dim=1)
        u2 = self.conv2(u2)

        return self.final(u2)


# -----------------------------
# LOAD MODEL
# -----------------------------
model = UNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("✅ Model loaded")


# -----------------------------
# CLEAN PREDICTION
# -----------------------------
def clean_prediction(pred):
    pred = (pred > 0.5).astype(np.uint8)

    # Morph cleanup
    kernel = np.ones((3, 3), np.uint8)
    pred = cv2.morphologyEx(pred, cv2.MORPH_OPEN, kernel)
    pred = cv2.morphologyEx(pred, cv2.MORPH_CLOSE, kernel)

    # Keep largest component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        pred = (labels == largest).astype(np.uint8)

    h, w = pred.shape

    # 🔥 TRAPEZOID ROAD ROI (KEY FIX)
    mask = np.zeros_like(pred)

    pts = np.array([
        [int(0.1*w), int(0.75*h)],   # bottom-left
        [int(0.9*w), int(0.75*h)],   # bottom-right
        [int(0.6*w), int(0.5*h)],    # top-right
        [int(0.4*w), int(0.5*h)]     # top-left
    ])

    cv2.fillPoly(mask, [pts], 1)

    pred = pred * mask

    return pred
# -----------------------------
# PREPROCESS
# -----------------------------
def preprocess(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    return img.to(DEVICE)


# -----------------------------
# OVERLAY MASK
# -----------------------------
def overlay_mask(frame, mask):
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

    colored = np.zeros_like(frame)
    colored[:, :, 1] = mask * 255

    output = cv2.addWeighted(frame, 1.0, colored, 0.4, 0)

    # label
    cv2.putText(output, "Drivable Area", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    return output


# -----------------------------
# WEBCAM
# -----------------------------
cap = cv2.VideoCapture("test_video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_tensor = preprocess(frame)

    with torch.no_grad():
        pred = model(input_tensor)
        pred = torch.sigmoid(pred).squeeze().cpu().numpy()

    pred = clean_prediction(pred)

    output = overlay_mask(frame, pred)

    cv2.imshow("Drivable Space Segmentation", output)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()