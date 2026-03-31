import numpy as np
from pyquaternion import Quaternion

def transform_matrix(translation, rotation, inverse=False):
    T = np.eye(4)
    T[:3, :3] = Quaternion(rotation).rotation_matrix
    T[:3, 3] = translation

    if inverse:
        T = np.linalg.inv(T)
    return T


def project_points(points, intrinsic):
    points = points / points[2, :]
    points = intrinsic @ points
    return points[:2, :]