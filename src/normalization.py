"""
Iris Normalization Module
Implements Daugman's Rubber Sheet Model to unwrap the annular iris disc
into a fixed-size rectangular polar coordinate strip.
"""

import cv2
import numpy as np
from typing import Tuple


def daugman_rubber_sheet(gray_img: np.ndarray,
                         pupil_circle: Tuple[int, int, int],
                         iris_circle: Tuple[int, int, int],
                         radial_res: int = 64,
                         angular_res: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transforms the annular iris region into a dimensionless rectangular polar strip.

    Args:
        gray_img: Grayscale eye image (2D numpy array)
        pupil_circle: (xp, yp, rp)
        iris_circle: (xi, yi, ri)
        radial_res: Height of normalized strip (radial samples)
        angular_res: Width of normalized strip (angular samples, 0 to 2*pi)

    Returns:
        normalized_iris: 2D array of size (radial_res, angular_res)
        noise_mask: 2D boolean array (True = valid iris pixel, False = noise/invalid)
    """
    xp, yp, rp = pupil_circle
    xi, yi, ri = iris_circle

    # Theta goes from 0 to 2*pi
    theta = np.linspace(0, 2 * np.pi, angular_res, endpoint=False)
    # r goes from 0 (inner pupil boundary) to 1 (outer iris boundary)
    r = np.linspace(0, 1, radial_res)

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Pupil boundary coordinates along theta
    xp_theta = xp + rp * cos_t
    yp_theta = yp + rp * sin_t

    # Iris boundary coordinates along theta
    xi_theta = xi + ri * cos_t
    yi_theta = yi + ri * sin_t

    # Grid of r and theta using broadcasting
    # r: (radial_res, 1), x_theta: (1, angular_res)
    r_grid = r[:, np.newaxis]

    # Linear interpolation between pupil boundary (r=0) and iris boundary (r=1)
    map_x = (1.0 - r_grid) * xp_theta[np.newaxis, :] + r_grid * xi_theta[np.newaxis, :]
    map_y = (1.0 - r_grid) * yp_theta[np.newaxis, :] + r_grid * yi_theta[np.newaxis, :]

    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    # Remap with bilinear interpolation
    normalized_iris = cv2.remap(gray_img, map_x, map_y, 
                                interpolation=cv2.INTER_LINEAR, 
                                borderMode=cv2.BORDER_REFLECT)

    # Enhance contrast with CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    normalized_iris = clahe.apply(normalized_iris)

    # Compute noise mask
    # Mask out coordinates out-of-bounds or extreme specular highlights (reflections)
    h, w = gray_img.shape[:2]
    valid_coords = (map_x >= 0) & (map_x < w) & (map_y >= 0) & (map_y < h)
    not_specular = normalized_iris < 250
    not_eyelash = normalized_iris > 15
    noise_mask = (valid_coords & not_specular & not_eyelash).astype(np.uint8)

    return normalized_iris, noise_mask
