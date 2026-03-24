import argparse
import math
import os
from typing import Dict, Tuple

import cv2
import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

'''
    run with: 
    python data_preparation\generate_masks_v2.py --data-root "root_directory" --output-dir "output_dir" --step 1 --min-distance 2 --max-distance 30 --roi-start 0.42 --min-forward 2 --max-forward 36 --max-lateral-near 3.8 --max-lateral-far 8.5 --open-kernel 9 --bottom-band-ratio 0.35 --max-center-offset-ratio 0.25
'''



def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate drivable masks from NuScenes map projection."
    )
    parser.add_argument("--data-root", required=True, help="NuScenes root path.")
    parser.add_argument("--output-dir", required=True, help="Output directory for masks.")
    parser.add_argument("--version", default="v1.0-mini", help="NuScenes version.")
    parser.add_argument("--camera", default="CAM_FRONT", help="Camera channel.")

    parser.add_argument("--roi-start", type=float, default=0.42, help="ROI start ratio [0,1).")
    parser.add_argument("--step", type=int, default=1, help="Projection stride in pixels.")
    parser.add_argument("--patch-size", type=float, default=100.0, help="Patch size in meters.")
    parser.add_argument("--canvas-size", type=int, default=900, help="Map raster size in pixels.")

    parser.add_argument("--min-distance", type=float, default=2.0, help="Min ground distance (m).")
    parser.add_argument("--max-distance", type=float, default=30.0, help="Max ground distance (m).")

    parser.add_argument("--min-forward", type=float, default=2.0, help="Min forward distance in ego frame (m).")
    parser.add_argument("--max-forward", type=float, default=36.0, help="Max forward distance in ego frame (m).")
    parser.add_argument("--max-lateral-near", type=float, default=3.8, help="Max lateral half-width near ego (m).")
    parser.add_argument("--max-lateral-far", type=float, default=8.5, help="Max lateral half-width at max-forward (m).")

    parser.add_argument("--close-kernel", type=int, default=9, help="Close kernel size.")
    parser.add_argument("--open-kernel", type=int, default=9, help="Open kernel size.")

    parser.add_argument("--bottom-band-ratio", type=float, default=0.35, help="Bottom center band width ratio.")
    parser.add_argument("--max-center-offset-ratio", type=float, default=0.25, help="Max centroid x-offset ratio from center.")
    parser.add_argument("--max-samples", type=int, default=0, help="0 => all samples.")

    return parser.parse_args()


def make_kernel(size: int) -> np.ndarray:
    k = max(1, int(size))
    if k % 2 == 0:
        k += 1
    return np.ones((k, k), dtype=np.uint8)


def quat_yaw_deg(q: Quaternion) -> float:
    return math.degrees(
        math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
    )


def build_maps(data_root: str) -> Dict[str, NuScenesMap]:
    map_names = [
        "singapore-onenorth",
        "singapore-hollandvillage",
        "singapore-queenstown",
        "boston-seaport",
    ]
    return {m: NuScenesMap(dataroot=data_root, map_name=m) for m in map_names}


def choose_drivable_layer(nusc_map: NuScenesMap) -> str:
    valid = set(nusc_map.non_geometric_layers)
    if "drivable_area" in valid:
        return "drivable_area"
    if "drivable_surface" in valid:
        return "drivable_surface"
    raise ValueError(f"No drivable layer found. Available: {sorted(valid)}")


def get_location_for_sample(nusc: NuScenes, sample: dict) -> str:
    scene = nusc.get("scene", sample["scene_token"])
    log = nusc.get("log", scene["log_token"])
    return log["location"]


def world_to_patch_pixels_rotated(
    x_world: np.ndarray,
    y_world: np.ndarray,
    ego_x: float,
    ego_y: float,
    patch_size: float,
    canvas_size: int,
    patch_angle_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    dx = x_world - ego_x
    dy = y_world - ego_y

    th = math.radians(patch_angle_deg)
    c = math.cos(th)
    s = math.sin(th)

    # Rotate world deltas into patch frame.
    u = c * dx + s * dy
    v = -s * dx + c * dy

    px = ((u / patch_size) + 0.5) * canvas_size
    py = (0.5 - (v / patch_size)) * canvas_size
    return px.astype(np.int32), py.astype(np.int32)


def keep_bottom_center_connected(
    mask_u8: np.ndarray,
    min_area: int = 800,
    bottom_band_ratio: float = 0.35,
    max_center_offset_ratio: float = 0.25,
) -> np.ndarray:
    h, w = mask_u8.shape
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    out = np.zeros_like(mask_u8)

    x0 = int((1.0 - bottom_band_ratio) * 0.5 * w)
    x1 = int((1.0 + bottom_band_ratio) * 0.5 * w)
    y0 = int(h * 0.92)

    bottom_band = labels[y0:, x0:x1]
    touching = set(np.unique(bottom_band).tolist())

    cx_ref = 0.5 * w
    max_off = max_center_offset_ratio * w

    for lab in range(1, num_labels):
        area = stats[lab, cv2.CC_STAT_AREA]
        cx = centroids[lab][0]
        if area >= min_area and lab in touching and abs(cx - cx_ref) <= max_off:
            out[labels == lab] = 1

    return out


def project_mask_for_sample(
    nusc: NuScenes,
    nusc_maps: Dict[str, NuScenesMap],
    drivable_layer_by_location: Dict[str, str],
    sample: dict,
    camera: str,
    roi_start: float,
    step: int,
    patch_size: float,
    canvas_size: int,
    min_distance: float,
    max_distance: float,
    min_forward: float,
    max_forward: float,
    max_lateral_near: float,
    max_lateral_far: float,
    close_kernel: int,
    open_kernel: int,
    bottom_band_ratio: float,
    max_center_offset_ratio: float,
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
    y0 = int(h * roi_start)
    y0 = max(0, min(y0, h - 1))

    cs_token = cam_data["calibrated_sensor_token"]
    if cs_token not in cs_cache:
        cs = nusc.get("calibrated_sensor", cs_token)
        k_inv = np.linalg.inv(np.asarray(cs["camera_intrinsic"], dtype=np.float64))
        r_ego_cam = Quaternion(cs["rotation"]).rotation_matrix
        t_ego_cam = np.asarray(cs["translation"], dtype=np.float64)
        cs_cache[cs_token] = (k_inv, r_ego_cam, t_ego_cam)
    else:
        k_inv, r_ego_cam, t_ego_cam = cs_cache[cs_token]

    pose = nusc.get("ego_pose", cam_data["ego_pose_token"])
    q_world_ego = Quaternion(pose["rotation"])
    r_world_ego = q_world_ego.rotation_matrix
    t_world_ego = np.asarray(pose["translation"], dtype=np.float64)

    gkey = (h, w, y0, step)
    if gkey not in grid_cache:
        ys = np.arange(y0, h, step, dtype=np.int32)
        xs = np.arange(0, w, step, dtype=np.int32)
        gy, gx = np.meshgrid(ys, xs, indexing="ij")
        pix = np.stack(
            [
                gx.reshape(-1).astype(np.float64),
                gy.reshape(-1).astype(np.float64),
                np.ones(gx.size, dtype=np.float64),
            ],
            axis=0,
        )
        grid_cache[gkey] = (gy, gx, pix)
    else:
        gy, gx, pix = grid_cache[gkey]

    location = get_location_for_sample(nusc, sample)
    if location not in nusc_maps:
        raise KeyError(f"Map location not loaded: {location}")
    nusc_map = nusc_maps[location]
    layer_name = drivable_layer_by_location[location]

    ego_x = float(t_world_ego[0])
    ego_y = float(t_world_ego[1])
    patch_angle = quat_yaw_deg(q_world_ego)

    map_mask = nusc_map.get_map_mask(
        patch_box=(ego_x, ego_y, patch_size, patch_size),
        patch_angle=patch_angle,
        layer_names=[layer_name],
        canvas_size=(canvas_size, canvas_size),
    )
    drivable_map = map_mask[0].astype(np.uint8)

    # camera -> world
    r_world_cam = r_world_ego @ r_ego_cam
    c_world = t_world_ego + r_world_ego @ t_ego_cam

    rays_cam = k_inv @ pix
    rays_world = r_world_cam @ rays_cam

    # Ground plane at current ego z.
    z_plane = float(t_world_ego[2])
    denom = rays_world[2, :]
    valid = np.abs(denom) > 1e-6

    lam = np.zeros_like(denom, dtype=np.float64)
    lam[valid] = (z_plane - c_world[2]) / denom[valid]
    in_front = lam > 0.0
    use = valid & in_front

    pts_world = c_world[:, None] + lam[None, :] * rays_world
    xw = pts_world[0, :]
    yw = pts_world[1, :]

    # Ego-frame corridor gating.
    pts_ego = r_world_ego.T @ (pts_world - t_world_ego[:, None])
    forward = pts_ego[0, :]
    lateral = pts_ego[1, :]

    use &= (forward >= min_forward) & (forward <= max_forward)

    den_fw = max(max_forward - min_forward, 1e-6)
    alpha = np.clip((forward - min_forward) / den_fw, 0.0, 1.0)
    lat_limit = max_lateral_near + alpha * (max_lateral_far - max_lateral_near)
    use &= np.abs(lateral) <= lat_limit

    dist = np.sqrt((xw - ego_x) ** 2 + (yw - ego_y) ** 2)
    use &= (dist >= min_distance) & (dist <= max_distance)

    mx, my = world_to_patch_pixels_rotated(
        x_world=xw,
        y_world=yw,
        ego_x=ego_x,
        ego_y=ego_y,
        patch_size=patch_size,
        canvas_size=canvas_size,
        patch_angle_deg=patch_angle,
    )

    in_map = (mx >= 0) & (mx < canvas_size) & (my >= 0) & (my < canvas_size)
    use &= in_map

    vals = np.zeros(gx.size, dtype=np.uint8)
    vals[use] = drivable_map[my[use], mx[use]]

    low = vals.reshape(gy.shape).astype(np.uint8)
    roi_h = h - y0

    low = cv2.medianBlur((low * 255).astype(np.uint8), 3)
    roi_mask = cv2.resize(low, (w, roi_h), interpolation=cv2.INTER_LINEAR)
    roi_mask = (roi_mask > 110).astype(np.uint8)

    full = np.zeros((h, w), dtype=np.uint8)
    full[y0:, :] = roi_mask

    full = cv2.morphologyEx(full, cv2.MORPH_CLOSE, make_kernel(close_kernel))
    full = cv2.morphologyEx(full, cv2.MORPH_OPEN, make_kernel(open_kernel))

    full = keep_bottom_center_connected(
        full,
        min_area=max(800, (h * w) // 300),
        bottom_band_ratio=bottom_band_ratio,
        max_center_offset_ratio=max_center_offset_ratio,
    )

    return full, img_path


def main():
    args = parse_args()

    if not (0.0 <= args.roi_start < 1.0):
        raise ValueError("--roi-start must be in [0,1).")
    if args.step < 1:
        raise ValueError("--step must be >= 1.")
    if args.patch_size <= 0 or args.canvas_size <= 0:
        raise ValueError("--patch-size and --canvas-size must be > 0.")
    if args.min_distance < 0 or args.max_distance <= args.min_distance:
        raise ValueError("Invalid distance range.")
    if args.min_forward < 0 or args.max_forward <= args.min_forward:
        raise ValueError("Invalid forward range.")
    if args.max_lateral_near <= 0 or args.max_lateral_far <= 0:
        raise ValueError("Lateral limits must be > 0.")
    if not (0 < args.bottom_band_ratio <= 1):
        raise ValueError("--bottom-band-ratio must be in (0,1].")
    if not (0 < args.max_center_offset_ratio <= 0.5):
        raise ValueError("--max-center-offset-ratio must be in (0,0.5].")

    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.isdir(args.data_root):
        raise FileNotFoundError(f"data-root not found: {args.data_root}")

    version_dir = os.path.join(args.data_root, args.version)
    if not os.path.isdir(version_dir):
        raise FileNotFoundError(
            f"Expected version folder missing: {version_dir}. Check --data-root and --version."
        )

    nusc = NuScenes(version=args.version, dataroot=args.data_root, verbose=True)
    nusc_maps = build_maps(args.data_root)
    drivable_layer_by_location = {loc: choose_drivable_layer(m) for loc, m in nusc_maps.items()}

    for loc, layer in drivable_layer_by_location.items():
        print(f"Map '{loc}': using layer '{layer}'")

    samples = nusc.sample if args.max_samples <= 0 else nusc.sample[: args.max_samples]

    total = len(samples)
    ok = 0
    failed = 0

    cs_cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    grid_cache: Dict[Tuple[int, int, int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    print(f"Total samples: {total}")
    print(f"Output dir: {args.output_dir}")

    for i, sample in enumerate(samples, start=1):
        print(f"Processing {i}/{total}")
        try:
            mask, img_path = project_mask_for_sample(
                nusc=nusc,
                nusc_maps=nusc_maps,
                drivable_layer_by_location=drivable_layer_by_location,
                sample=sample,
                camera=args.camera,
                roi_start=args.roi_start,
                step=args.step,
                patch_size=args.patch_size,
                canvas_size=args.canvas_size,
                min_distance=args.min_distance,
                max_distance=args.max_distance,
                min_forward=args.min_forward,
                max_forward=args.max_forward,
                max_lateral_near=args.max_lateral_near,
                max_lateral_far=args.max_lateral_far,
                close_kernel=args.close_kernel,
                open_kernel=args.open_kernel,
                bottom_band_ratio=args.bottom_band_ratio,
                max_center_offset_ratio=args.max_center_offset_ratio,
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