import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

"""
    use this command for running: 
    python data_preparation\visual_mask_test.py --image "sample-image-path" --mask "binary-mask-path"
"""


def verify_mask(image, mask, alpha=0.5):
    """
    Overlay a binary drivable mask on an RGB image.

    image: H x W x 3 (RGB), uint8
    mask:  H x W, values {0, 1} or {0, 255}
    alpha: blend factor for overlay intensity
    """
    image = np.asarray(image, dtype=np.uint8)
    mask = np.asarray(mask)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected image shape (H, W, 3), got {image.shape}")

    if mask.ndim != 2:
        raise ValueError(f"Expected mask shape (H, W), got {mask.shape}")

    if image.shape[:2] != mask.shape:
        raise ValueError(
            f"Image/mask size mismatch: image={image.shape[:2]}, mask={mask.shape}"
        )

    # Normalize mask to {0,1}
    mask_bin = (mask > 127).astype(np.uint8)

    overlay = image.copy()

    # Blend drivable pixels toward green.
    green = np.array([0, 200, 80], dtype=np.float32)
    drivable_pixels = mask_bin == 1
    overlay[drivable_pixels] = (
        overlay[drivable_pixels].astype(np.float32) * (1.0 - alpha) + green * alpha
    ).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(mask_bin, cmap="gray")
    axes[1].set_title("Mask (white = drivable)")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay (green = drivable)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


def load_image_rgb(image_path):
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def load_mask_gray(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")
    return mask


def parse_args():
    parser = argparse.ArgumentParser(description="Visual check for generated drivable masks.")
    parser.add_argument(
        "--image",
        required=True,
        help="Path to original camera image (e.g., samples/CAM_FRONT/xxx.jpg)",
    )
    parser.add_argument(
        "--mask",
        required=True,
        help="Path to generated mask image (e.g., Output/masks/xxx.jpg)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Overlay blend factor in [0, 1]. Default: 0.6",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image path does not exist: {args.image}")
    if not os.path.exists(args.mask):
        raise FileNotFoundError(f"Mask path does not exist: {args.mask}")
    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError("--alpha must be between 0 and 1")

    image = load_image_rgb(args.image)
    mask = load_mask_gray(args.mask)

    verify_mask(image, mask, alpha=args.alpha)


if __name__ == "__main__":
    main()