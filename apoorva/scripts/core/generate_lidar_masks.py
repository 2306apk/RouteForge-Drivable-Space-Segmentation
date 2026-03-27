import os
import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

DATAROOT = "./data/nuscenes"
VERSION = "v1.0-mini"
OUTPUT_DIR = "./masks/train_auto_v2"

os.makedirs(OUTPUT_DIR, exist_ok=True)

nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=True)


def get_matrix(translation, rotation, inverse=False):
    T = np.eye(4)
    T[:3, :3] = Quaternion(rotation).rotation_matrix
    T[:3, 3] = translation

    if inverse:
        T = np.linalg.inv(T)
    return T


def load_lidar_points(path):
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
    return pts[:, :3]  # xyz


for sample in nusc.sample:

    cam_token = sample['data']['CAM_FRONT']
    lidar_token = sample['data']['LIDAR_TOP']

    cam_data = nusc.get('sample_data', cam_token)
    lidar_data = nusc.get('sample_data', lidar_token)

    img_path = os.path.join(DATAROOT, cam_data['filename'])
    lidar_path = os.path.join(DATAROOT, lidar_data['filename'])

    image = cv2.imread(img_path)
    if image is None:
        continue

    h, w, _ = image.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    points = load_lidar_points(lidar_path)

    # GET TRANSFORMS
    cam_calib = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    lidar_calib = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])

    ego_pose_cam = nusc.get('ego_pose', cam_data['ego_pose_token'])
    ego_pose_lidar = nusc.get('ego_pose', lidar_data['ego_pose_token'])

    # MATRICES
    T_lidar_to_ego = get_matrix(lidar_calib['translation'], lidar_calib['rotation'])
    T_ego_to_global = get_matrix(ego_pose_lidar['translation'], ego_pose_lidar['rotation'])

    T_global_to_ego_cam = get_matrix(ego_pose_cam['translation'], ego_pose_cam['rotation'], inverse=True)
    T_ego_to_cam = get_matrix(cam_calib['translation'], cam_calib['rotation'], inverse=True)

    # TRANSFORM POINTS
    pts = np.hstack((points, np.ones((points.shape[0], 1)))).T

    pts = T_lidar_to_ego @ pts
    pts = T_ego_to_global @ pts
    pts = T_global_to_ego_cam @ pts
    pts = T_ego_to_cam @ pts

    pts = pts[:3, :]

    # FILTER FRONT
    valid = pts[2, :] > 1.0
    pts = pts[:, valid]

    # PROJECT
    K = np.array(cam_calib['camera_intrinsic'])
    pts = pts / pts[2, :]
    pts = K @ pts

    xs = pts[0, :].astype(int)
    ys = pts[1, :].astype(int)

    # FILTER IMAGE BOUNDS
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    xs = xs[valid]
    ys = ys[valid]

    # CREATE MASK
    mask[ys, xs] = 255

    # DILATE (IMPORTANT)
    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.dilate(mask, kernel)

    # CLOSE GAPS
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    filename = os.path.basename(img_path)
    cv2.imwrite(os.path.join(OUTPUT_DIR, filename), mask)

print("DONE LIDAR MASKS")