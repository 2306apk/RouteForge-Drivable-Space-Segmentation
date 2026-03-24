import argparse
import os
import sys
from glob import glob

import cv2
import numpy as np


def sanity_check_mask(mask, image_name="", min_pct=5.0, max_pct=70.0, threshold=127):
    """
    Validate one drivable mask and print diagnostics.

    Supports raw mask values like:
    - {0, 1}
    - {0, 255}
    - grayscale masks (converted via threshold)
    """
    mask = np.asarray(mask)

    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D (H, W). Got shape: {mask.shape}")

    raw_unique = np.unique(mask)
    mask_bin = (mask > threshold).astype(np.uint8)
    bin_unique = np.unique(mask_bin)

    h, w = mask_bin.shape
    total_pixels = h * w
    drivable_pixels = int(mask_bin.sum())
    pct = (drivable_pixels / total_pixels) * 100.0

    problems = []
    if drivable_pixels == 0:
        problems.append("PROBLEM: completely empty mask - extraction may have failed")
    if pct < min_pct:
        problems.append(
            f"PROBLEM: very low drivable area ({pct:.1f}%) - projection/threshold may be wrong"
        )
    if pct > max_pct:
        problems.append(
            f"PROBLEM: very high drivable area ({pct:.1f}%) - leakage/over-segmentation likely"
        )

    print(f"[{image_name}]")
    print(f"  Shape: {mask_bin.shape}")
    print(f"  Raw unique values: {raw_unique}")
    print(f"  Binary unique values: {bin_unique}  (expected [0 1], [0], or [1])")
    print(f"  Drivable %: {pct:.2f}%")
    print(f"  All-zero mask: {drivable_pixels == 0}")
    if problems:
        for p in problems:
            print(f"  {p}")
    print()

    return {
        "name": image_name,
        "shape": mask_bin.shape,
        "raw_unique": raw_unique.tolist(),
        "binary_unique": bin_unique.tolist(),
        "drivable_pct": pct,
        "all_zero": drivable_pixels == 0,
        "problems": problems,
    }


def load_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")
    return mask


def check_single_mask(mask_path, min_pct, max_pct, threshold):
    mask = load_mask(mask_path)
    return sanity_check_mask(
        mask=mask,
        image_name=os.path.basename(mask_path),
        min_pct=min_pct,
        max_pct=max_pct,
        threshold=threshold,
    )


def collect_mask_paths(mask_dir, exts, recursive):
    paths = []
    for ext in exts:
        ext = ext.lstrip(".")
        pattern = f"**/*.{ext}" if recursive else f"*.{ext}"
        pattern_upper = f"**/*.{ext.upper()}" if recursive else f"*.{ext.upper()}"
        paths.extend(glob(os.path.join(mask_dir, pattern), recursive=recursive))
        paths.extend(glob(os.path.join(mask_dir, pattern_upper), recursive=recursive))
    return sorted(set(paths))


def check_mask_folder(mask_dir, exts, recursive, min_pct, max_pct, threshold):
    paths = collect_mask_paths(mask_dir, exts, recursive)

    if not paths:
        raise FileNotFoundError(
            f"No mask files found in: {mask_dir} for extensions={exts} (recursive={recursive})"
        )

    print(f"Found {len(paths)} mask files\n")
    results = []

    for p in paths:
        try:
            mask = load_mask(p)
            result = sanity_check_mask(
                mask=mask,
                image_name=os.path.basename(p),
                min_pct=min_pct,
                max_pct=max_pct,
                threshold=threshold,
            )
            results.append(result)
        except Exception as e:
            print(f"[{os.path.basename(p)}]")
            print(f"  PROBLEM: failed to process file - {e}\n")
            results.append({"name": os.path.basename(p), "problems": [str(e)]})

    problem_count = sum(1 for r in results if r.get("problems"))
    avg_pct = np.mean([r["drivable_pct"] for r in results if "drivable_pct" in r])

    print("Summary")
    print(f"  Total files: {len(results)}")
    print(f"  Files with problems: {problem_count}")
    print(f"  Mean drivable %: {avg_pct:.2f}%")

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sanity-check drivable mask files (single file or folder)."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mask", type=str, help="Path to one mask image")
    group.add_argument("--mask-dir", type=str, help="Path to folder of mask images")

    parser.add_argument(
        "--exts",
        nargs="+",
        default=["png", "jpg", "jpeg"],
        help="Extensions for --mask-dir mode (default: png jpg jpeg)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subfolders when using --mask-dir",
    )
    parser.add_argument(
        "--min-pct",
        type=float,
        default=5.0,
        help="Flag masks below this drivable percentage (default: 5)",
    )
    parser.add_argument(
        "--max-pct",
        type=float,
        default=70.0,
        help="Flag masks above this drivable percentage (default: 70)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=127,
        help="Binarization threshold for grayscale masks (default: 127)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.min_pct < 0 or args.max_pct > 100 or args.min_pct > args.max_pct:
        raise ValueError("Invalid min/max percentage range.")
    if not (0 <= args.threshold <= 255):
        raise ValueError("--threshold must be in [0, 255]")

    if args.mask:
        result = check_single_mask(args.mask, args.min_pct, args.max_pct, args.threshold)
        has_problem = len(result.get("problems", [])) > 0
        sys.exit(1 if has_problem else 0)
    else:
        results = check_mask_folder(
            mask_dir=args.mask_dir,
            exts=args.exts,
            recursive=args.recursive,
            min_pct=args.min_pct,
            max_pct=args.max_pct,
            threshold=args.threshold,
        )
        has_problem = any(len(r.get("problems", [])) > 0 for r in results)
        sys.exit(1 if has_problem else 0)


if __name__ == "__main__":
    main()