import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.unet_tiny import TinyUNet


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FriendUNet(nn.Module):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


@dataclass
class ModelConfig:
    name: str
    arch: str
    ckpt: str
    img_h: int
    img_w: int
    threshold: float
    preprocess: str
    smooth_window: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare segmentation models on a single image.")
    p.add_argument("--image-path", required=True, help="Path to test image.")
    p.add_argument("--output-dir", default="outputs/compare", help="Directory for comparison outputs.")

    p.add_argument("--a-name", default="model_a")
    p.add_argument("--a-arch", choices=["tiny", "friend_unet"], default="tiny")
    p.add_argument("--a-ckpt", required=True)
    p.add_argument("--a-img-h", type=int, default=256)
    p.add_argument("--a-img-w", type=int, default=512)
    p.add_argument("--a-threshold", type=float, default=0.5)
    p.add_argument("--a-preprocess", choices=["auto", "tiny", "friend"], default="auto")
    p.add_argument("--a-smooth-window", type=int, default=1)

    p.add_argument("--b-enable", action="store_true", help="Enable second model for side-by-side compare.")
    p.add_argument("--b-name", default="model_b")
    p.add_argument("--b-arch", choices=["tiny", "friend_unet"], default="friend_unet")
    p.add_argument("--b-ckpt", default=None)
    p.add_argument("--b-img-h", type=int, default=128)
    p.add_argument("--b-img-w", type=int, default=128)
    p.add_argument("--b-threshold", type=float, default=0.5)
    p.add_argument("--b-preprocess", choices=["auto", "tiny", "friend"], default="auto")
    p.add_argument("--b-smooth-window", type=int, default=1)

    p.add_argument("--fps-iters", type=int, default=100)
    p.add_argument("--fps-warmup", type=int, default=20)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def normalize_preprocess(arch: str, preprocess: str) -> str:
    if preprocess != "auto":
        return preprocess
    if arch == "tiny":
        return "tiny"
    return "friend"


def build_model(arch: str) -> nn.Module:
    if arch == "tiny":
        return TinyUNet(in_ch=3, out_ch=1, base=32)
    return FriendUNet()


def load_weights(model: nn.Module, ckpt_path: str, device: torch.device) -> None:
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict)


def preprocess_image(frame_bgr: np.ndarray, mode: str, img_h: int, img_w: int) -> torch.Tensor:
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0

    if mode == "tiny":
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std

    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).unsqueeze(0)


def postprocess_mask(mask_prob: np.ndarray, threshold: float) -> np.ndarray:
    mask = (mask_prob > threshold).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def overlay_mask(frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask_full = cv2.resize(mask, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay = frame_bgr.copy()
    overlay[mask_full == 1] = [0, 255, 0]
    return overlay


def infer_once(model: nn.Module, tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    return probs


def estimate_fps(model: nn.Module, tensor: torch.Tensor, device: torch.device, warmup: int, iters: int) -> float:
    model.eval()
    tensor = tensor.to(device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(tensor)

        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(iters):
            _ = model(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0

    return float(iters / dt)


def build_config(prefix: str, args: argparse.Namespace) -> ModelConfig:
    return ModelConfig(
        name=getattr(args, f"{prefix}_name"),
        arch=getattr(args, f"{prefix}_arch"),
        ckpt=getattr(args, f"{prefix}_ckpt"),
        img_h=getattr(args, f"{prefix}_img_h"),
        img_w=getattr(args, f"{prefix}_img_w"),
        threshold=getattr(args, f"{prefix}_threshold"),
        preprocess=normalize_preprocess(getattr(args, f"{prefix}_arch"), getattr(args, f"{prefix}_preprocess")),
        smooth_window=getattr(args, f"{prefix}_smooth_window"),
    )


def save_binary_mask(path: str, mask: np.ndarray, out_h: int, out_w: int) -> None:
    mask_full = cv2.resize(mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(path, mask_full * 255)


def run_one(cfg: ModelConfig, frame_bgr: np.ndarray, device: torch.device, fps_warmup: int, fps_iters: int):
    model = build_model(cfg.arch).to(device)
    load_weights(model, cfg.ckpt, device)
    model.eval()

    tensor = preprocess_image(frame_bgr, cfg.preprocess, cfg.img_h, cfg.img_w)
    probs = infer_once(model, tensor, device)
    mask = postprocess_mask(probs, cfg.threshold)
    overlay = overlay_mask(frame_bgr, mask)
    fps = estimate_fps(model, tensor, device, warmup=fps_warmup, iters=fps_iters)

    ratio = float(mask.mean())
    return mask, overlay, fps, ratio


def add_caption(img: np.ndarray, text: str) -> np.ndarray:
    canvas = img.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(canvas, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return canvas


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    if not os.path.isfile(args.image_path):
        raise FileNotFoundError(f"Image not found: {args.image_path}")

    frame = cv2.imread(args.image_path)
    if frame is None:
        raise RuntimeError(f"Could not read image: {args.image_path}")

    os.makedirs(args.output_dir, exist_ok=True)

    cfg_a = build_config("a", args)
    mask_a, overlay_a, fps_a, ratio_a = run_one(
        cfg_a,
        frame,
        device,
        fps_warmup=args.fps_warmup,
        fps_iters=args.fps_iters,
    )

    cv2.imwrite(os.path.join(args.output_dir, f"{cfg_a.name}_overlay.png"), overlay_a)
    save_binary_mask(
        os.path.join(args.output_dir, f"{cfg_a.name}_mask.png"),
        mask_a,
        out_h=frame.shape[0],
        out_w=frame.shape[1],
    )

    panels = [
        add_caption(frame, "Original"),
        add_caption(overlay_a, f"{cfg_a.name} | fps={fps_a:.2f} | ratio={ratio_a:.3f}"),
    ]

    if args.b_enable:
        if not args.b_ckpt:
            raise ValueError("--b-enable requires --b-ckpt")

        cfg_b = build_config("b", args)
        mask_b, overlay_b, fps_b, ratio_b = run_one(
            cfg_b,
            frame,
            device,
            fps_warmup=args.fps_warmup,
            fps_iters=args.fps_iters,
        )

        cv2.imwrite(os.path.join(args.output_dir, f"{cfg_b.name}_overlay.png"), overlay_b)
        save_binary_mask(
            os.path.join(args.output_dir, f"{cfg_b.name}_mask.png"),
            mask_b,
            out_h=frame.shape[0],
            out_w=frame.shape[1],
        )

        diff = cv2.absdiff(mask_a.astype(np.uint8), mask_b.astype(np.uint8))
        diff_full = cv2.resize(diff, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST) * 255
        diff_color = cv2.cvtColor(diff_full, cv2.COLOR_GRAY2BGR)

        cv2.imwrite(os.path.join(args.output_dir, "diff_mask.png"), diff_full)
        panels.append(add_caption(overlay_b, f"{cfg_b.name} | fps={fps_b:.2f} | ratio={ratio_b:.3f}"))
        panels.append(add_caption(diff_color, "Difference (white=disagree)"))

        print(f"{cfg_a.name}: fps={fps_a:.2f}, drivable_ratio={ratio_a:.4f}")
        print(f"{cfg_b.name}: fps={fps_b:.2f}, drivable_ratio={ratio_b:.4f}")
        print(f"Absolute ratio gap: {abs(ratio_a - ratio_b):.4f}")
    else:
        print(f"{cfg_a.name}: fps={fps_a:.2f}, drivable_ratio={ratio_a:.4f}")

    board = np.hstack(panels)
    board_path = os.path.join(args.output_dir, "comparison_board.png")
    cv2.imwrite(board_path, board)

    print(f"Saved outputs to: {args.output_dir}")
    print(f"Comparison board: {board_path}")


if __name__ == "__main__":
    main()
