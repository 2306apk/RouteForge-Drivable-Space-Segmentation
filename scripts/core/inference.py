import argparse
import os
from collections import deque
from typing import List

import cv2
import numpy as np
import torch
import torch.nn as nn


DEFAULT_MODEL_PATH = "models/road_seg.pth"
DEFAULT_INPUT = "test_video.mp4"
DEFAULT_OUTPUT_DIR = "outputs/inference"
DEFAULT_IMG_SIZE = 128
DEFAULT_THRESHOLD = 0.5
DEFAULT_SMOOTH_WINDOW = 3


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
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


def load_model(model_path: str, device: str):
    model = UNet().to(device)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def preprocess_frame(frame: np.ndarray, img_size: int):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (img_size, img_size))
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    tensor = torch.from_numpy(image).unsqueeze(0)
    return tensor


def postprocess_mask(mask: np.ndarray, threshold: float = 0.5):
    mask = (mask > threshold).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = (labels == largest).astype(np.uint8)
    return mask


def overlay_mask(frame: np.ndarray, mask: np.ndarray):
    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    colored = np.zeros_like(frame)
    colored[:, :, 1] = mask_resized * 255
    overlay = cv2.addWeighted(frame, 0.7, colored, 0.3, 0)
    cv2.putText(overlay, "Drivable Area", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    return overlay


def smooth_mask(mask: np.ndarray, buffer: deque, window: int):
    buffer.append(mask.astype(np.uint8))
    if len(buffer) > window:
        buffer.popleft()
    if len(buffer) == 0:
        return mask
    stack = np.stack(list(buffer), axis=0)
    avg = np.mean(stack, axis=0)
    return (avg >= 0.5).astype(np.uint8)


def infer_image_folder(
    model,
    device: str,
    input_dir: str,
    output_dir: str,
    img_size: int,
    threshold: float,
    smooth_window: int,
):
    os.makedirs(output_dir, exist_ok=True)
    mask_dir = os.path.join(output_dir, "masks")
    overlay_dir = os.path.join(output_dir, "overlay")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    filenames = sorted(
        [f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    )
    buffer = deque()
    for filename in filenames:
        frame = cv2.imread(os.path.join(input_dir, filename))
        if frame is None:
            continue
        tensor = preprocess_frame(frame, img_size).to(device)
        with torch.no_grad():
            pred = model(tensor)
            pred = torch.sigmoid(pred).squeeze().cpu().numpy()
        mask = postprocess_mask(pred, threshold)
        if smooth_window > 1:
            mask = smooth_mask(mask, buffer, smooth_window)
        overlay = overlay_mask(frame, mask)
        cv2.imwrite(os.path.join(mask_dir, filename), mask * 255)
        cv2.imwrite(os.path.join(overlay_dir, filename), overlay)
    print(f"Inference completed for folder {input_dir}. Masks -> {mask_dir}, overlays -> {overlay_dir}")


def infer_video(
    model,
    device: str,
    input_path: str,
    output_dir: str,
    img_size: int,
    threshold: float,
    smooth_window: int,
):
    os.makedirs(output_dir, exist_ok=True)
    mask_dir = os.path.join(output_dir, "masks")
    overlay_dir = os.path.join(output_dir, "overlay")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    frame_idx = 0
    buffer = deque()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        tensor = preprocess_frame(frame, img_size).to(device)
        with torch.no_grad():
            pred = model(tensor)
            pred = torch.sigmoid(pred).squeeze().cpu().numpy()
        mask = postprocess_mask(pred, threshold)
        if smooth_window > 1:
            mask = smooth_mask(mask, buffer, smooth_window)
        overlay = overlay_mask(frame, mask)
        base = f"frame_{frame_idx:05d}.png"
        cv2.imwrite(os.path.join(mask_dir, base), mask * 255)
        cv2.imwrite(os.path.join(overlay_dir, base), overlay)
        frame_idx += 1
    cap.release()
    print(f"Inference completed for video {input_path}. Masks -> {mask_dir}, overlays -> {overlay_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run segmentation inference on CAM_FRONT images or video.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to the trained model.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Image folder or video file path.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory to write inference outputs.")
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE, help="Inference input size.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Binary threshold for mask.")
    parser.add_argument("--smooth-window", type=int, default=DEFAULT_SMOOTH_WINDOW, help="Temporal smoothing window size.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference on.")
    args = parser.parse_args()

    model = load_model(args.model_path, args.device)
    if os.path.isdir(args.input):
        infer_image_folder(
            model=model,
            device=args.device,
            input_dir=args.input,
            output_dir=args.output_dir,
            img_size=args.img_size,
            threshold=args.threshold,
            smooth_window=args.smooth_window,
        )
    elif os.path.isfile(args.input):
        infer_video(
            model=model,
            device=args.device,
            input_path=args.input,
            output_dir=args.output_dir,
            img_size=args.img_size,
            threshold=args.threshold,
            smooth_window=args.smooth_window,
        )
    else:
        raise FileNotFoundError(f"Input path not found: {args.input}")


if __name__ == "__main__":
    main()
