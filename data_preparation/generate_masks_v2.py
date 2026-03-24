import argparse
import os
from typing import Dict, Tuple

import cv2
import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

'''
    pip install pyquaternion (if not already there)
    download map expansion pack from nuscenes website
    put the expansion folder inside maps
    command to run this code:
    python data_preparation\generate_masks_v2.py --data-root "dataset dirctory" --output-dir "o/p directory"
'''


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate drivable masks from NuScenes map projection."
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="NuScenes root containing v1.0-mini/, samples/, sweeps/.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save output masks.",
    )
    parser.add_argument(
        "--version",
        default="v1.0-mini",
        help="NuScenes version (default: v1.0-mini).",
    )
    parser.add_argument(
        "--camera",
        default="CAM_FRONT",
        help="Camera channel (default: CAM_FRONT).",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=4,
        help="Pixel stride for projection (lower = denser + slower).",
    )
    parser.add_argument(
        "--roi-start",
        type=float,
        default=0.45,
        help="Process from this vertical ratio downward (0..1).",
    )
    parser.add_argument(
        "--patch-size",
        type=float,
        default=100.0,
        help="Map patch size in meters around ego position.",
    )
    parser.add_argument(
        "--canvas-size",
        type=int,
        default=500,
        help="Map raster size in pixels for get_map_mask.",
    )
    parser.add_argument(
        "--close-kernel",
        type=int,
        default=9,
        help="Morphological close kernel size.",
    )
    parser.add_argument(
        "--open-kernel",
        type=int,
        default=5,
        help="Morphological open kernel size.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional limit for quick testing (0 = all).",
    )
    return parser.parse_args()


def build_maps(data_root: str) -> Dict[str, NuScenesMap]:
    # nuScenes mini map locations.
    map_names = [
        "singapore-onenorth",
        "singapore-hollandvillage",
        "singapore-queenstown",
        "boston-seaport",
    ]
    return {name: NuScenesMap(dataroot=data_root, map_name=name) for name in map_names}


def get_drivable_layer_name(nusc_map: NuScenesMap) -> str:
    # Different devkit/dataset variants expose different names.
    valid = set(nusc_map.non_geometric_layers)
    if "drivable_area" in valid:
        return "drivable_area"
    if "drivable_surface" in valid:
        return "drivable_surface"
    raise ValueError(
        f"No drivable layer found. Available non-geometric layers: {sorted(valid)}"
    )


def get_location_for_sample(nusc: NuScenes, sample: dict) -> str:
    scene = nusc.get("scene", sample["scene_token"])
    log = nusc.get("log", scene["log_token"])
    return log["location"]


def world_to_patch_pixels(
    x_world: np.ndarray,
    y_world: np.ndarray,
    ego_x: float,
    ego_y: float,
    patch_size: float,
    canvas_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    # Patch in world coordinates:
    # x: [ego_x - s/2, ego_x + s/2]
    # y: [ego_y - s/2, ego_y + s/2]
    x_min = ego_x - patch_size / 2.0
    y_max = ego_y + patch_size / 2.0

    px = ((x_world - x_min) / patch_size * canvas_size).astype(np.int32)
    # Flip y because image row grows downward.
    py = ((y_max - y_world) / patch_size * canvas_size).astype(np.int32)
    return px, py


def make_kernel(size: int) -> np.ndarray:
    size = max(1, int(size))
    if size % 2 == 0:
        size += 1
    return np.ones((size, size), dtype=np.uint8)


def project_mask_for_sample(
    nusc: NuScenes,
    nusc_maps: Dict[str, NuScenesMap],
    sample: dict,
    drivable_layer_by_location: Dict[str, str],
    camera: str,
    step: int,
    roi_start: float,
    patch_size: float,
    canvas_size: int,
    close_kernel: int,
    open_kernel: int,
    cs_cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    grid_cache: Dict[Tuple[int, int, int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, str]:
    cam_token = sample["data"][camera]
    cam_data = nusc.get("sample_data", cam_token)

    img_path = nusc.get_sample_data_path(cam_token)
    image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Could not read image: {img_path}")

    h, w = image_bgr.shape[:2]
    roi_y0 = int(h * roi_start)
    if roi_y0 < 0 or roi_y0 >= h:
        raise ValueError(f"Invalid roi_start={roi_start} for image height={h}")

    cs_token = cam_data["calibrated_sensor_token"]
    if cs_token not in cs_cache:
        cs = nusc.get("calibrated_sensor", cs_token)
        k_inv = np.linalg.inv(np.array(cs["camera_intrinsic"], dtype=np.float64))
        cam_rot = Quaternion(cs["rotation"]).rotation_matrix
        cam_trans = np.array(cs["translation"], dtype=np.float64)
        cs_cache[cs_token] = (k_inv, cam_rot, cam_trans)
    else:
        k_inv, cam_rot, cam_trans = cs_cache[cs_token]

    pose = nusc.get("ego_pose", cam_data["ego_pose_token"])
    ego_rot = Quaternion(pose["rotation"]).rotation_matrix
    ego_trans = np.array(pose["translation"], dtype=np.float64)

    grid_key = (h, w, roi_y0, step)
    if grid_key not in grid_cache:
        ys = np.arange(roi_y0, h, step, dtype=np.int32)
        xs = np.arange(0, w, step, dtype=np.int32)
        grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
        pixels = np.stack(
            [
                grid_x.reshape(-1).astype(np.float64),
                grid_y.reshape(-1).astype(np.float64),
                np.ones(grid_x.size, dtype=np.float64),
            ],
            axis=0,
        )  # 3 x N
        grid_cache[grid_key] = (grid_y, grid_x, pixels)
    else:
        grid_y, grid_x, pixels = grid_cache[grid_key]

    location = get_location_for_sample(nusc, sample)
    if location not in nusc_maps:
        raise KeyError(f"Map location not loaded: {location}")
    nusc_map = nusc_maps[location]
    layer_name = drivable_layer_by_location[location]

    ego_x, ego_y = pose["translation"][0], pose["translation"][1]
    map_mask = nusc_map.get_map_mask(
        patch_box=(ego_x, ego_y, patch_size, patch_size),
        patch_angle=0.0,
        layer_names=[layer_name],
        canvas_size=(canvas_size, canvas_size),
    )
    drivable_map = map_mask[0].astype(np.uint8)

    # camera -> ego -> world
    r_world_cam = ego_rot @ cam_rot
    cam_center_world = ego_trans + ego_rot @ cam_trans

    rays_cam = k_inv @ pixels
    rays_world = r_world_cam @ rays_cam

    denom = rays_world[2, :]
    valid = np.abs(denom) > 1e-6

    lam = np.zeros_like(denom, dtype=np.float64)
    lam[valid] = -cam_center_world[2] / denom[valid]
    in_front = lam > 0.0
    use = valid & in_front

    ground_pts = cam_center_world[:, None] + lam[None, :] * rays_world
    gx = ground_pts[0, :]
    gy = ground_pts[1, :]

    map_x, map_y = world_to_patch_pixels(
        x_world=gx,
        y_world=gy,
        ego_x=ego_x,
        ego_y=ego_y,
        patch_size=patch_size,
        canvas_size=canvas_size,
    )

    in_map = (
        (map_x >= 0)
        & (map_x < canvas_size)
        & (map_y >= 0)
        & (map_y < canvas_size)
    )
    use = use & in_map

    drivable_vals = np.zeros(grid_x.size, dtype=np.uint8)
    drivable_vals[use] = drivable_map[map_y[use], map_x[use]]

    low_res = drivable_vals.reshape(grid_y.shape).astype(np.uint8)
    roi_h = h - roi_y0
    roi_mask = cv2.resize(low_res, (w, roi_h), interpolation=cv2.INTER_NEAREST)

    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[roi_y0:, :] = roi_mask

    full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_CLOSE, make_kernel(close_kernel))
    full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, make_kernel(open_kernel))

    return full_mask, img_path


def main():
    args = parse_args()

    if args.step <= 0:
        raise ValueError("--step must be >= 1")
    if not (0.0 <= args.roi_start < 1.0):
        raise ValueError("--roi-start must be in [0, 1)")
    if args.patch_size <= 0:
        raise ValueError("--patch-size must be > 0")
    if args.canvas_size <= 0:
        raise ValueError("--canvas-size must be > 0")

    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.isdir(args.data_root):
        raise FileNotFoundError(f"data-root does not exist: {args.data_root}")

    version_dir = os.path.join(args.data_root, args.version)
    if not os.path.isdir(version_dir):
        raise FileNotFoundError(
            f"Expected version folder missing: {version_dir}. Check --data-root and --version."
        )

    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=True)
    nusc_maps = build_maps(args.data_root)

    drivable_layer_by_location = {
        loc: get_drivable_layer_name(nm) for loc, nm in nusc_maps.items()
    }
    for loc, layer in drivable_layer_by_location.items():
        print(f"Map '{loc}' -> using layer '{layer}'")

    samples = nusc.sample
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    total = len(samples)
    ok = 0
    failed = 0

    cs_cache = {}
    grid_cache = {}

    print(f"Total samples: {total}")
    print(f"Saving masks to: {args.output_dir}")

    for i, sample in enumerate(samples, start=1):
        print(f"Processing {i}/{total}")
        try:
            mask, img_path = project_mask_for_sample(
                nusc=nusc,
                nusc_maps=nusc_maps,
                sample=sample,
                drivable_layer_by_location=drivable_layer_by_location,
                camera=args.camera,
                step=args.step,
                roi_start=args.roi_start,
                patch_size=args.patch_size,
                canvas_size=args.canvas_size,
                close_kernel=args.close_kernel,
                open_kernel=args.open_kernel,
                cs_cache=cs_cache,
                grid_cache=grid_cache,
            )
            base = os.path.splitext(os.path.basename(img_path))[0]
            out_path = os.path.join(args.output_dir, f"{base}.png")
            cv2.imwrite(out_path, (mask * 255).astype(np.uint8))
            ok += 1
        except Exception as e:
            failed += 1
            print(f"  Failed: {e}")

    print("DONE")
    print(f"Success: {ok}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()