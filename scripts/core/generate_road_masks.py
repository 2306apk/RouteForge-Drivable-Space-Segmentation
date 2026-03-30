import argparse
import math
import os
from typing import Dict, Tuple

import cv2
import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--camera", default="CAM_FRONT")

    parser.add_argument("--roi-start", type=float, default=0.42)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--patch-size", type=float, default=100.0)
    parser.add_argument("--canvas-size", type=int, default=900)

    parser.add_argument("--min-distance", type=float, default=2.0)
    parser.add_argument("--max-distance", type=float, default=30.0)

    parser.add_argument("--min-forward", type=float, default=2.0)
    parser.add_argument("--max-forward", type=float, default=36.0)
    parser.add_argument("--max-lateral-near", type=float, default=3.8)
    parser.add_argument("--max-lateral-far", type=float, default=8.5)

    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def quat_yaw_deg(q):
    return math.degrees(
        math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
    )


def build_maps(data_root):
    maps = [
        "singapore-onenorth",
        "singapore-hollandvillage",
        "singapore-queenstown",
        "boston-seaport",
    ]
    return {m: NuScenesMap(dataroot=data_root, map_name=m) for m in maps}


def choose_drivable_layer(nusc_map):
    if "drivable_area" in nusc_map.non_geometric_layers:
        return "drivable_area"
    return "road_segment"


def world_to_patch(xw, yw, ego_x, ego_y, patch_size, canvas, angle):
    dx = xw - ego_x
    dy = yw - ego_y

    th = math.radians(angle)
    c, s = math.cos(th), math.sin(th)

    u = c * dx + s * dy
    v = -s * dx + c * dy

    px = ((u / patch_size) + 0.5) * canvas
    py = (0.5 - (v / patch_size)) * canvas
    return px.astype(int), py.astype(int)


def project_mask(nusc, maps, layer_map, sample, args, cs_cache, grid_cache):
    cam_token = sample["data"][args.camera]
    sd = nusc.get("sample_data", cam_token)

    img_path = nusc.get_sample_data_path(cam_token)
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    y0 = int(h * args.roi_start)

    # camera intrinsics
    cs_token = sd["calibrated_sensor_token"]
    if cs_token not in cs_cache:
        cs = nusc.get("calibrated_sensor", cs_token)
        Kinv = np.linalg.inv(np.array(cs["camera_intrinsic"]))
        R = Quaternion(cs["rotation"]).rotation_matrix
        t = np.array(cs["translation"])
        cs_cache[cs_token] = (Kinv, R, t)

    Kinv, R_cam, t_cam = cs_cache[cs_token]

    pose = nusc.get("ego_pose", sd["ego_pose_token"])
    R_ego = Quaternion(pose["rotation"]).rotation_matrix
    t_ego = np.array(pose["translation"])

    # grid
    key = (h, w, y0)
    if key not in grid_cache:
        ys = np.arange(y0, h)
        xs = np.arange(w)
        gy, gx = np.meshgrid(ys, xs, indexing="ij")

        pix = np.stack([
            gx.reshape(-1),
            gy.reshape(-1),
            np.ones(gx.size)
        ], axis=0)

        grid_cache[key] = (gy, gx, pix)

    gy, gx, pix = grid_cache[key]

    # map
    scene = nusc.get("scene", sample["scene_token"])
    log = nusc.get("log", scene["log_token"])
    loc = log["location"]

    nmap = maps[loc]
    layer = layer_map[loc]

    ego_x, ego_y = t_ego[:2]
    angle = quat_yaw_deg(Quaternion(pose["rotation"]))

    mask_map = nmap.get_map_mask(
        patch_box=(ego_x, ego_y, args.patch_size, args.patch_size),
        patch_angle=angle,
        layer_names=[layer],
        canvas_size=(args.canvas_size, args.canvas_size),
    )[0]

    # projection
    R_world_cam = R_ego @ R_cam
    cam_pos = t_ego + R_ego @ t_cam

    rays = R_world_cam @ (Kinv @ pix)

    z_plane = t_ego[2]
    lam = (z_plane - cam_pos[2]) / rays[2]
    valid = lam > 0

    pts = cam_pos[:, None] + lam * rays
    xw, yw = pts[0], pts[1]

    mx, my = world_to_patch(xw, yw, ego_x, ego_y,
                           args.patch_size, args.canvas_size, angle)

    in_map = (mx >= 0) & (mx < args.canvas_size) & (my >= 0) & (my < args.canvas_size)
    valid &= in_map

    vals = np.zeros(gx.size)
    vals[valid] = mask_map[my[valid], mx[valid]]

    low = vals.reshape(gy.shape).astype(np.uint8)

    # ===== FIX 1: NO DISTORTION =====
    roi = cv2.resize(low, (w, h - y0), interpolation=cv2.INTER_NEAREST)

    full = np.zeros((h, w), dtype=np.uint8)
    full[y0:, :] = roi

    # ===== FIX 2: CLEANUP =====
    kernel = np.ones((5, 5), np.uint8)
    full = cv2.morphologyEx(full, cv2.MORPH_CLOSE, kernel)
    full = cv2.medianBlur(full, 5)

    # ===== FIX 3: KEEP BOTTOM CONNECTED =====
    num, labels = cv2.connectedComponents(full)
    final = np.zeros_like(full)

    bottom = np.unique(labels[h-10:h, :])
    for l in bottom:
        if l == 0:
            continue
        final[labels == l] = 255

    return final, img_path


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    nusc = NuScenes(version=args.version, dataroot=args.data_root)
    maps = build_maps(args.data_root)
    layer_map = {k: choose_drivable_layer(v) for k, v in maps.items()}

    samples = nusc.sample if args.max_samples == 0 else nusc.sample[:args.max_samples]

    cs_cache = {}
    grid_cache = {}

    for i, sample in enumerate(samples):
        try:
            mask, path = project_mask(nusc, maps, layer_map, sample,
                                     args, cs_cache, grid_cache)

            name = os.path.basename(path).replace(".jpg", ".png")
            cv2.imwrite(os.path.join(args.output_dir, name), mask)

            if i % 50 == 0:
                print(i, "done")

        except Exception as e:
            print("fail:", e)

    print("DONE")


if __name__ == "__main__":
    main()