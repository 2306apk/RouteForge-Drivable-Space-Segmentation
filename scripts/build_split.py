import argparse
import csv
import os
import random
from collections import defaultdict

from nuscenes.nuscenes import NuScenes


def parse_args():
    parser = argparse.ArgumentParser(description="Build scene-wise train/val/test CSV splits.")
    parser.add_argument("--data-root", required=True, help="NuScenes root (contains v1.0-mini, samples, ...)")
    parser.add_argument("--masks-dir", required=True, help="Directory with generated mask PNGs")
    parser.add_argument("--out-dir", default="data", help="Output folder for CSV split files")
    parser.add_argument("--version", default="v1.0-mini", help="NuScenes version")
    parser.add_argument("--camera", default="CAM_FRONT", help="Camera channel")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "mask_path", "scene_token", "sample_token"])
        w.writerows(rows)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=True)

    scene_to_rows = defaultdict(list)
    missing_masks = 0

    for sample in nusc.sample:
        scene_token = sample["scene_token"]
        cam_token = sample["data"][args.camera]
        img_path = nusc.get_sample_data_path(cam_token)

        stem = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(args.masks_dir, f"{stem}.png")

        if not os.path.isfile(mask_path):
            missing_masks += 1
            continue

        row = [img_path, mask_path, scene_token, sample["token"]]
        scene_to_rows[scene_token].append(row)

    scenes = list(scene_to_rows.keys())
    random.Random(args.seed).shuffle(scenes)

    n_total = len(scenes)
    n_test = max(1, int(round(n_total * args.test_ratio)))
    n_val = max(1, int(round(n_total * args.val_ratio)))
    n_train = max(1, n_total - n_val - n_test)

    train_scenes = set(scenes[:n_train])
    val_scenes = set(scenes[n_train:n_train + n_val])
    test_scenes = set(scenes[n_train + n_val:])

    train_rows = []
    val_rows = []
    test_rows = []

    for scene, rows in scene_to_rows.items():
        if scene in train_scenes:
            train_rows.extend(rows)
        elif scene in val_scenes:
            val_rows.extend(rows)
        else:
            test_rows.extend(rows)

    write_csv(os.path.join(args.out_dir, "train.csv"), train_rows)
    write_csv(os.path.join(args.out_dir, "val.csv"), val_rows)
    write_csv(os.path.join(args.out_dir, "test.csv"), test_rows)

    print("Split build done.")
    print(f"Scenes total: {n_total}")
    print(f"Train scenes: {len(train_scenes)}, rows: {len(train_rows)}")
    print(f"Val scenes: {len(val_scenes)}, rows: {len(val_rows)}")
    print(f"Test scenes: {len(test_scenes)}, rows: {len(test_rows)}")
    print(f"Skipped (missing mask): {missing_masks}")


if __name__ == "__main__":
    main()