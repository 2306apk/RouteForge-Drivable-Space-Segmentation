import numpy as np
from nuscenes_loader import load_all_metadata, get_cam_front_frames

meta = load_all_metadata()
frames = get_cam_front_frames(meta)

frame = frames[0]

# Get calibration
calib = meta["calibrated_sensor"][frame["calibrated_sensor_token"]]
ego = meta["ego_pose"][frame["ego_pose_token"]]

print("\n=== CAMERA INTRINSIC ===")
print(np.array(calib["camera_intrinsic"]))

print("\n=== CAMERA TRANSLATION ===")
print(np.array(calib["translation"]))

print("\n=== CAMERA ROTATION (QUATERNION) ===")
print(np.array(calib["rotation"]))

print("\n=== EGO TRANSLATION ===")
print(np.array(ego["translation"]))

print("\n=== EGO ROTATION ===")
print(np.array(ego["rotation"]))