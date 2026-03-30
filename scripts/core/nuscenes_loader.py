import os
import json

DATA_ROOT = "data"
META_ROOT = os.path.join(DATA_ROOT, "v1.0-mini")


def load_json(name):
    with open(os.path.join(META_ROOT, name)) as f:
        return json.load(f)


def build_index(data, key="token"):
    return {item[key]: item for item in data}


def load_all_metadata():
    print("Loading metadata...")

    return {
        "sample_data": load_json("sample_data.json"),
        "sample": build_index(load_json("sample.json")),
        "scene": build_index(load_json("scene.json")),
        "log": build_index(load_json("log.json")),
        "map": load_json("map.json"),  # keep as list
        "ego_pose": build_index(load_json("ego_pose.json")),
        "calibrated_sensor": build_index(load_json("calibrated_sensor.json")),
        "sensor": build_index(load_json("sensor.json")),
    }


# find map using log_token
def get_map_from_log(log_token, maps):
    for m in maps:
        if log_token in m["log_tokens"]:
            return m["filename"]
    return None


def get_cam_front_frames(meta):
    frames = []

    for entry in meta["sample_data"]:
        calib = meta["calibrated_sensor"][entry["calibrated_sensor_token"]]
        sensor = meta["sensor"][calib["sensor_token"]]

        if sensor["channel"] != "CAM_FRONT":
            continue

        if not entry["is_key_frame"]:
            continue

        sample = meta["sample"][entry["sample_token"]]
        scene = meta["scene"][sample["scene_token"]]
        log = meta["log"][scene["log_token"]]

        map_filename = get_map_from_log(log["token"], meta["map"])

        if map_filename is None:
            continue

        frames.append({
            "image_path": os.path.join(DATA_ROOT, entry["filename"]),
            "ego_pose_token": entry["ego_pose_token"],
            "calibrated_sensor_token": entry["calibrated_sensor_token"],
            "map_name": map_filename
        })

    print(f"✅ CAM_FRONT frames: {len(frames)}")
    return frames