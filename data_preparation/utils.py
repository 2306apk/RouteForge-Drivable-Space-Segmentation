# utils.py

import numpy as np
import cv2

def project_points(points, intrinsic):
    """
    Project 3D points to 2D image plane
    """
    points = np.array(points)
    points = points.T

    proj = intrinsic @ points
    proj = proj / proj[2]

    return proj[:2].T


def create_mask(image_shape, polygons):
    """
    Create binary mask from polygons
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)

    for poly in polygons:
        pts = np.array(poly, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 1)

    return mask