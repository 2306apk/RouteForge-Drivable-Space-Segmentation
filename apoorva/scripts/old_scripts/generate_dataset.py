import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2**40)

import json
import cv2
import numpy as np

DATA_ROOT = "data"
META_ROOT = os.path.join(DATA_ROOT, "v1.0-mini")
MAPS_DIR = os.path.join(DATA_ROOT, "maps")

SAVE_IMG_DIR = os.path.join(DATA_ROOT, "train_v3/images")
SAVE_MASK_DIR = os.path.join(DATA_ROOT, "train_v3/masks")

os.makedirs(SAVE_IMG_DIR, exist_ok=True)
os.makedirs(SAVE_MASK_DIR, exist_ok=True)

# =========================
# LOAD JSON
# =========================
def load_json(name):
    with open(os.path.join(META_ROOT, name)) as f:
        return json.load(f)

# =========================
# LOAD META
# =========================
def load_meta():
    return {
        "sample_data": load_json("sample_data.json"),
        "calibrated_sensor": load_json("calibrated_sensor.json"),
        "ego_pose": load_json("ego_pose.json"),
        "sensor": load_json("sensor.json")
    }

# =========================
# LOOKUPS
# =========================
def build_lookup(table, key="token"):
    return {item[key]: item for item in table}

# =========================
# CAMERA PARAMS
# =========================
def get_camera_params(entry, meta):
    calib_lookup = build_lookup(meta["calibrated_sensor"])
    ego_lookup = build_lookup(meta["ego_pose"])

    calib = calib_lookup[entry["calibrated_sensor_token"]]
    ego = ego_lookup[entry["ego_pose_token"]]

    K = np.array(calib["camera_intrinsic"])
    cam_t = np.array(calib["translation"])
    cam_r = np.array(calib["rotation"])

    ego_t = np.array(ego["translation"])
    ego_r = np.array(ego["rotation"])

    return K, cam_t, cam_r, ego_t, ego_r

# =========================
# QUATERNION → ROTATION
# =========================
def quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
        [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
        [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]
    ])

# =========================
# PROJECT POINT
# =========================
def project_point(pt, K, cam_t, cam_r, ego_t, ego_r):
    R_cam = quat_to_rot(cam_r)
    R_ego = quat_to_rot(ego_r)

    # world → ego
    pt_ego = R_ego.T @ (pt - ego_t)

    # ego → camera
    pt_cam = R_cam.T @ (pt_ego - cam_t)

    if pt_cam[2] <= 0:
        return None

    pt_img = K @ pt_cam
    pt_img /= pt_img[2]

    return int(pt_img[0]), int(pt_img[1])

# =========================
# GENERATE MASK
# =========================
def generate_mask(entry, meta, map_img):
    img_path = os.path.join(DATA_ROOT, entry["filename"])
    img = cv2.imread(img_path)

    if img is None:
        raise ValueError("Image not loaded")

    K, cam_t, cam_r, ego_t, ego_r = get_camera_params(entry, meta)

    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    resolution = 0.1  # meters per pixel

    h, w = map_img.shape

    ys, xs = np.where(map_img > 200)

    for i in range(0, len(xs), 10):  # downsample
        x = xs[i]
        y = ys[i]

        # 🔥 FIX: center map correctly
        wx = (x - w / 2) * resolution
        wy = (y - h / 2) * resolution
        wz = 0

        pt_world = np.array([wx, wy, wz])

        proj = project_point(pt_world, K, cam_t, cam_r, ego_t, ego_r)

        if proj is None:
            continue

        px, py = proj

        if 0 <= px < img.shape[1] and 0 <= py < img.shape[0]:
            mask[py, px] = 255

    # Fill mask
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return img, mask

# =========================
# MAIN
# =========================
def main():
    print("Loading metadata...")
    meta = load_meta()

    calib_lookup = build_lookup(meta["calibrated_sensor"])
    sensor_lookup = build_lookup(meta["sensor"])

    # Load map
    map_path = os.path.join(MAPS_DIR, os.listdir(MAPS_DIR)[0])
    map_img = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)

    if map_img is None:
        raise ValueError("Map not loaded")

    print("Map shape:", map_img.shape)

    count = 0

    for entry in meta["sample_data"]:
        calib = calib_lookup[entry["calibrated_sensor_token"]]
        sensor = sensor_lookup[calib["sensor_token"]]

        if sensor["channel"] != "CAM_FRONT":
            continue

        if not entry["is_key_frame"]:
            continue

        print(f"Processing frame {count}...")

        try:
            img, mask = generate_mask(entry, meta, map_img)
        except:
            continue

        name = f"img{count}.png"

        cv2.imwrite(os.path.join(SAVE_IMG_DIR, name), img)
        cv2.imwrite(os.path.join(SAVE_MASK_DIR, name), mask)

        count += 1

        if count >= 20:  # debug first
            break

    print("Done!")

if __name__ == "__main__":
    main()