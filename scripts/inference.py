import argparse
import os
import sys
from collections import deque
from typing import Deque

import cv2
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.unet_tiny import TinyUNet


DEFAULT_MODEL_PATH = "checkpoints/best.pt"
DEFAULT_INPUT = "samples/CAM_FRONT"
DEFAULT_OUTPUT_DIR = "outputs/inference"
DEFAULT_IMG_H = 256
DEFAULT_IMG_W = 512
DEFAULT_THRESHOLD = 0.5
DEFAULT_SMOOTH_WINDOW = 3
DEFAULT_OVERLAY_VIDEO_NAME = "overlay.mp4"
DEFAULT_OVERLAY_FPS = 12.0


def load_model(model_path: str, device: torch.device) -> TinyUNet:
    model = TinyUNet(in_ch=3, out_ch=1, base=32).to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess_frame(frame: np.ndarray, img_h: int, img_w: int) -> torch.Tensor:
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255.0

    # Match training-time normalization from DrivableDataset.
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std

    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image).unsqueeze(0)


def postprocess_mask(mask_prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    mask = (mask_prob > threshold).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = (labels == largest).astype(np.uint8)

    return mask


def smooth_mask(mask: np.ndarray, buffer: Deque[np.ndarray], window: int) -> np.ndarray:
    buffer.append(mask.astype(np.uint8))
    if len(buffer) > window:
        buffer.popleft()

    if not buffer:
        return mask

    avg = np.mean(np.stack(list(buffer), axis=0), axis=0)
    return (avg >= 0.5).astype(np.uint8)


def overlay_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    color = np.zeros_like(frame)
    color[:, :, 1] = mask_resized * 255
    overlay = cv2.addWeighted(frame, 0.7, color, 0.3, 0)
    cv2.putText(overlay, "Drivable Area", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    return overlay


def run_model(model: TinyUNet, tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    return probs


def make_video_writer(path: str, width: int, height: int, fps: float) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create overlay video: {path}")
    return writer


def infer_image_folder(
    model: TinyUNet,
    device: torch.device,
    input_dir: str,
    output_dir: str,
    img_h: int,
    img_w: int,
    threshold: float,
    smooth_window: int,
    save_overlay_video: bool,
    overlay_video_name: str,
    overlay_fps: float,
) -> None:
    mask_dir = os.path.join(output_dir, "masks")
    overlay_dir = os.path.join(output_dir, "overlay")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    names = sorted([n for n in os.listdir(input_dir) if n.lower().endswith((".png", ".jpg", ".jpeg"))])
    if not names:
        raise RuntimeError(f"No images found in folder: {input_dir}")

    writer = None
    overlay_video_path = os.path.join(output_dir, overlay_video_name)
    buffer: Deque[np.ndarray] = deque()
    for index, name in enumerate(names):
        frame = cv2.imread(os.path.join(input_dir, name))
        if frame is None:
            continue

        probs = run_model(model, preprocess_frame(frame, img_h, img_w), device)
        mask = postprocess_mask(probs, threshold)
        if smooth_window > 1:
            mask = smooth_mask(mask, buffer, smooth_window)

        overlay = overlay_mask(frame, mask)
        if save_overlay_video and writer is None:
            writer = make_video_writer(
                overlay_video_path,
                width=frame.shape[1],
                height=frame.shape[0],
                fps=overlay_fps,
            )

        cv2.imwrite(os.path.join(mask_dir, name), mask * 255)
        cv2.imwrite(os.path.join(overlay_dir, name), overlay)
        if writer is not None:
            writer.write(overlay)

        if (index + 1) % 200 == 0:
            print(f"Processed {index + 1}/{len(names)} frames")

    if writer is not None:
        writer.release()

    print(f"Inference completed for folder {input_dir}")
    print(f"Masks: {mask_dir}")
    print(f"Overlay: {overlay_dir}")
    if save_overlay_video:
        print(f"Overlay video: {overlay_video_path}")


def infer_video(
    model: TinyUNet,
    device: torch.device,
    input_path: str,
    output_dir: str,
    img_h: int,
    img_w: int,
    threshold: float,
    smooth_window: int,
    save_overlay_video: bool,
    overlay_video_name: str,
    overlay_fps: float,
) -> None:
    mask_dir = os.path.join(output_dir, "masks")
    overlay_dir = os.path.join(output_dir, "overlay")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    source_fps = source_fps if source_fps and source_fps > 0 else overlay_fps
    writer = None
    overlay_video_path = os.path.join(output_dir, overlay_video_name)

    buffer: Deque[np.ndarray] = deque()
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        probs = run_model(model, preprocess_frame(frame, img_h, img_w), device)
        mask = postprocess_mask(probs, threshold)
        if smooth_window > 1:
            mask = smooth_mask(mask, buffer, smooth_window)

        overlay = overlay_mask(frame, mask)
        if save_overlay_video and writer is None:
            writer = make_video_writer(
                overlay_video_path,
                width=frame.shape[1],
                height=frame.shape[0],
                fps=source_fps,
            )

        name = f"frame_{idx:05d}.png"
        cv2.imwrite(os.path.join(mask_dir, name), mask * 255)
        cv2.imwrite(os.path.join(overlay_dir, name), overlay)
        if writer is not None:
            writer.write(overlay)
        idx += 1

    if writer is not None:
        writer.release()
    cap.release()
    print(f"Inference completed for video {input_path}")
    print(f"Frames: {idx}")
    print(f"Masks: {mask_dir}")
    print(f"Overlay: {overlay_dir}")
    if save_overlay_video:
        print(f"Overlay video: {overlay_video_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run drivable-space segmentation inference on folder/video.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to checkpoint (.pt/.pth).")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Image folder or video path.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for masks and overlays.")
    parser.add_argument("--img-h", type=int, default=DEFAULT_IMG_H, help="Inference resize height.")
    parser.add_argument("--img-w", type=int, default=DEFAULT_IMG_W, help="Inference resize width.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Mask threshold.")
    parser.add_argument("--smooth-window", type=int, default=DEFAULT_SMOOTH_WINDOW, help="Temporal smoothing window.")
    parser.add_argument(
        "--save-overlay-video",
        action="store_true",
        help="Save a single MP4 overlay video in output-dir.",
    )
    parser.add_argument(
        "--overlay-video-name",
        default=DEFAULT_OVERLAY_VIDEO_NAME,
        help="Filename for the overlay video saved under output-dir.",
    )
    parser.add_argument(
        "--overlay-fps",
        type=float,
        default=DEFAULT_OVERLAY_FPS,
        help="Overlay video FPS for image folders or fallback when source video FPS is unavailable.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device: cuda or cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.smooth_window < 1:
        raise ValueError("--smooth-window must be >= 1")
    if args.overlay_fps <= 0:
        raise ValueError("--overlay-fps must be > 0")

    device = torch.device(args.device)
    model = load_model(args.model_path, device)

    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.isdir(args.input):
        infer_image_folder(
            model=model,
            device=device,
            input_dir=args.input,
            output_dir=args.output_dir,
            img_h=args.img_h,
            img_w=args.img_w,
            threshold=args.threshold,
            smooth_window=args.smooth_window,
            save_overlay_video=args.save_overlay_video,
            overlay_video_name=args.overlay_video_name,
            overlay_fps=args.overlay_fps,
        )
    elif os.path.isfile(args.input):
        infer_video(
            model=model,
            device=device,
            input_path=args.input,
            output_dir=args.output_dir,
            img_h=args.img_h,
            img_w=args.img_w,
            threshold=args.threshold,
            smooth_window=args.smooth_window,
            save_overlay_video=args.save_overlay_video,
            overlay_video_name=args.overlay_video_name,
            overlay_fps=args.overlay_fps,
        )
    else:
        raise FileNotFoundError(f"Input path not found: {args.input}")


if __name__ == "__main__":
    main()
