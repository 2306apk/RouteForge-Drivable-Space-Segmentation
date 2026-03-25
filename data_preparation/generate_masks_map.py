import os
import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

# ================= CONFIG =================
DATA_ROOT = r"C:\Users\Arnav\Projects\HACKATHONS\MAHE Hackathon 2026\Dataset"
OUTPUT_DIR = "outputs/masks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= INIT =================
nusc = NuScenes(version='v1.0-mini', dataroot=DATA_ROOT, verbose=True)

maps = {
    "boston-seaport": NuScenesMap(dataroot=DATA_ROOT, map_name="boston-seaport"),
    "singapore-hollandvillage": NuScenesMap(dataroot=DATA_ROOT, map_name="singapore-hollandvillage"),
    "singapore-onenorth": NuScenesMap(dataroot=DATA_ROOT, map_name="singapore-onenorth"),
    "singapore-queenstown": NuScenesMap(dataroot=DATA_ROOT, map_name="singapore-queenstown"),
}

print("Total samples:", len(nusc.sample))

# ================= SETTINGS =================
DYNAMIC_CLASSES = [
    'vehicle.car',
    'vehicle.truck',
    'vehicle.bus',
    'vehicle.motorcycle',
    'vehicle.bicycle'
]

# ================= FUNCTIONS =================

def get_drivable_polygons(nusc_map):
    polygons = []
    for record in nusc_map.drivable_area:
        for token in record['polygon_tokens']:  # FIXED
            poly = nusc_map.extract_polygon(token)
            polygons.append(poly)
    return polygons


def world_to_camera(points, ego_pose, cam_calib):
    # World → Ego
    points = points - np.array(ego_pose['translation'])
    rot = Quaternion(ego_pose['rotation']).inverse.rotation_matrix
    points = np.dot(rot, points.T).T

    # Ego → Camera
    points = points - np.array(cam_calib['translation'])
    rot = Quaternion(cam_calib['rotation']).inverse.rotation_matrix
    points = np.dot(rot, points.T).T

    return points


def project_to_image(points, cam_intrinsic):
    mask = points[:, 2] > 1.0
    points = points[mask]

    if len(points) == 0:
        return None

    pts_2d = np.dot(points[:, :3], cam_intrinsic.T)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]

    return pts_2d[:, :2]


def remove_dynamic_objects(nusc, sample, mask, ego_pose, cam_calib, cam_intrinsic, w, h):

    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)

        category = ann['category_name']

        if not any(cls in category for cls in DYNAMIC_CLASSES):
            continue

        box = Box(
            ann['translation'],
            ann['size'],
            Quaternion(ann['rotation'])
        )

        # World → Ego
        box.translate(-np.array(ego_pose['translation']))
        box.rotate(Quaternion(ego_pose['rotation']).inverse)

        # Ego → Camera
        box.translate(-np.array(cam_calib['translation']))
        box.rotate(Quaternion(cam_calib['rotation']).inverse)

        if box.center[2] <= 0:
            continue

        corners = box.corners()

        mask_front = corners[2, :] > 0
        corners = corners[:, mask_front]

        if corners.shape[1] < 3:
            continue

        pts_2d = np.dot(corners.T, cam_intrinsic.T)
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]

        pts_2d = pts_2d[:, :2].astype(int)

        # Clip to image bounds
        pts_2d[:, 0] = np.clip(pts_2d[:, 0], 0, w - 1)
        pts_2d[:, 1] = np.clip(pts_2d[:, 1], 0, h - 1)

        cv2.fillConvexPoly(mask, pts_2d, 0)

    return mask


# ================= MAIN LOOP =================

for i, sample in enumerate(nusc.sample):

    print(f"Processing {i}")

    try:
        cam_token = sample['data']['CAM_FRONT']
        cam_data = nusc.get('sample_data', cam_token)

        img_path = os.path.join(DATA_ROOT, cam_data['filename'])
        image = cv2.imread(img_path)

        if image is None:
            continue

        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        ego_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])
        cam_calib = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        cam_intrinsic = np.array(cam_calib['camera_intrinsic'])

        scene = nusc.get('scene', sample['scene_token'])
        log = nusc.get('log', scene['log_token'])
        map_name = log['location']

        nusc_map = maps[map_name]
        polygons = get_drivable_polygons(nusc_map)

        # ===== MAP PROJECTION =====
        for poly in polygons:
            coords = np.array(poly.exterior.coords)

            coords_3d = np.hstack((coords, np.zeros((coords.shape[0], 1))))

            pts = world_to_camera(coords_3d, ego_pose, cam_calib)

            # Remove far points
            pts = pts[pts[:, 2] < 50]
            if len(pts) < 3:
                continue

            pts_2d = project_to_image(pts, cam_intrinsic)

            if pts_2d is None or len(pts_2d) < 3:
                continue

            pts_2d = pts_2d.astype(int)

            pts_2d[:, 0] = np.clip(pts_2d[:, 0], 0, w - 1)
            pts_2d[:, 1] = np.clip(pts_2d[:, 1], 0, h - 1)

            cv2.fillPoly(mask, [pts_2d], 255)

        # ===== REMOVE VEHICLES =====
        mask = remove_dynamic_objects(
            nusc,
            sample,
            mask,
            ego_pose,
            cam_calib,
            cam_intrinsic,
            w,
            h
        )

        # ===== SAVE =====
        filename = os.path.basename(img_path)
        cv2.imwrite(os.path.join(OUTPUT_DIR, filename), mask)

        print("Saved:", filename)

    except Exception as e:
        print("Error:", e)
        continue

print("DONE")