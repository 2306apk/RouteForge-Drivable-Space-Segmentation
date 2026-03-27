import numpy as np


def quat_to_rot(q):
    w, x, y, z = q

    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
    ])


def build_transform(rotation, translation):
    R = quat_to_rot(rotation)
    t = np.array(translation).reshape(3, 1)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3:] = t

    return T


def project_point(point_world, ego, calib):
    point = np.array([*point_world, 1.0]).reshape(4, 1)

    # WORLD → EGO
    T_ego = build_transform(ego["rotation"], ego["translation"])
    point_ego = np.linalg.inv(T_ego) @ point

    # EGO → CAMERA
    T_cam = build_transform(calib["rotation"], calib["translation"])
    point_cam = np.linalg.inv(T_cam) @ point_ego

    x, y, z = point_cam[:3].flatten()

    if z <= 0:
        return None

    K = np.array(calib["camera_intrinsic"])

    pixel = K @ np.array([x, y, z])
    u = pixel[0] / pixel[2]
    v = pixel[1] / pixel[2]

    return int(u), int(v)