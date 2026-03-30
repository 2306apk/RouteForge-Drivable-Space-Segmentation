import json
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

DATA_ROOT = "data"
META_ROOT = "data/v1.0-mini"


# =========================
# LOAD JSON
# =========================
def load_json(name):
    with open(os.path.join(META_ROOT, name), "r") as f:
        return json.load(f)


def load_meta():
    return {
        "sample_data": load_json("sample_data.json"),
        "calibrated_sensor": load_json("calibrated_sensor.json"),
        "ego_pose": load_json("ego_pose.json"),
        "sensor": load_json("sensor.json")
    }


# =========================
# LOOKUP
# =========================
def build_lookup(table):
    return {item["token"]: item for item in table}


# =========================
# QUATERNION → ROTATION
# =========================
def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
    ])


# =========================
# GET CAM_FRONT FRAME
# =========================
def get_cam_front_frame(meta):
    sensor_lookup = build_lookup(meta["sensor"])
    calib_lookup = build_lookup(meta["calibrated_sensor"])

    for entry in meta["sample_data"]:
        calib = calib_lookup[entry["calibrated_sensor_token"]]
        sensor = sensor_lookup[calib["sensor_token"]]

        if sensor["modality"] != "camera":
            continue

        if sensor["channel"] != "CAM_FRONT":
            continue

        if not entry["is_key_frame"]:
            continue

        img_path = os.path.join(DATA_ROOT, entry["filename"])

        if not os.path.exists(img_path):
            continue

        return {
            "image_path": img_path,
            "calibrated_sensor_token": entry["calibrated_sensor_token"],
            "ego_pose_token": entry["ego_pose_token"]
        }

    return None


# =========================
# MAIN
# =========================
def main():
    print("Loading metadata...")
    meta = load_meta()

    calib_lookup = build_lookup(meta["calibrated_sensor"])
    ego_lookup = build_lookup(meta["ego_pose"])

    frame = get_cam_front_frame(meta)

    if frame is None:
        print("❌ No frame found")
        return

    print("Using image:", frame["image_path"])

    img = cv2.imread(frame["image_path"])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    calib = calib_lookup[frame["calibrated_sensor_token"]]
    ego = ego_lookup[frame["ego_pose_token"]]

    # =========================
    # STEP 1: FORWARD POINT (FIXED)
    # =========================
    ego_rot = quaternion_to_rotation_matrix(ego["rotation"])
    forward = ego_rot @ np.array([1, 0, 0])

    point_world = np.array(ego["translation"]) + forward * 10.0

    # =========================
    # STEP 2: WORLD → EGO
    # =========================
    ego_trans = np.array(ego["translation"])
    point_ego = np.dot(ego_rot.T, (point_world - ego_trans))

    # =========================
    # STEP 3: EGO → CAMERA
    # =========================
    cam_rot = quaternion_to_rotation_matrix(calib["rotation"])
    cam_trans = np.array(calib["translation"])

    point_cam = np.dot(cam_rot.T, (point_ego - cam_trans))

    print("Point in camera coords:", point_cam)

    if point_cam[2] <= 0:
        print("❌ Point behind camera")
        return

    # =========================
    # STEP 4: PROJECT
    # =========================
    K = np.array(calib["camera_intrinsic"])

    pixel = np.dot(K, point_cam)
    pixel = pixel[:2] / pixel[2]

    x, y = int(pixel[0]), int(pixel[1])
    print("Projected pixel:", x, y)

    # =========================
    # DRAW
    # =========================
    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
        cv2.circle(img, (x, y), 10, (255, 0, 0), -1)
        print("✅ Point drawn")
    else:
        print("⚠️ Point outside image")

    # =========================
    # SHOW
    # =========================
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.title("Projection Test (FINAL FIX)")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()