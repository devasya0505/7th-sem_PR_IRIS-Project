"""
Iris Boundary Localization Module
Locates the inner (pupil) and outer (iris/limbic) circular boundaries
using Circular Hough Transform and edge detection techniques.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict


def preprocess_image(gray_img: np.ndarray) -> np.ndarray:
    """Apply median/gaussian blur to suppress specular reflections and noise."""
    blurred = cv2.medianBlur(gray_img, 5)
    blurred = cv2.GaussianBlur(blurred, (5, 5), 0)
    return blurred


def detect_pupil_hough(gray_img: np.ndarray, 
                       min_radius: int = 15, 
                       max_radius: int = 80) -> Tuple[int, int, int]:
    """
    Detects the inner pupil boundary using Circular Hough Transform.
    The pupil is the dark central aperture of the eye.
    """
    h, w = gray_img.shape[:2]
    blurred = cv2.GaussianBlur(gray_img, (7, 7), 0)

    # Adaptive / binary threshold to isolate dark pupil region
    # Compute 10th percentile intensity as adaptive threshold reference
    p15 = np.percentile(blurred, 15)
    thresh_val = max(35, min(80, int(p15 * 1.4)))
    _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)

    # Morphological opening to clean eyelash noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh_clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Canny on the thresholded pupil mask gives pristine pupil boundary edges
    edges = cv2.Canny(thresh_clean, 40, 120)

    # Circular Hough Transform on pupil edge map
    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=h // 4,
        param1=50,
        param2=11,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        best_circle = None
        min_cost = float('inf')

        img_cx, img_cy = w // 2, h // 2

        for (x, y, r) in circles:
            if 0 <= x < w and 0 <= y < h and r >= min_radius:
                # Calculate mean intensity inside circle
                mask = np.zeros_like(gray_img, dtype=np.uint8)
                cv2.circle(mask, (x, y), max(2, int(r * 0.7)), 255, -1)
                mean_int = cv2.mean(gray_img, mask=mask)[0]

                # Center distance penalty (pupil is roughly central in eye image)
                dist_center = np.sqrt((x - img_cx)**2 + (y - img_cy)**2) / (w * 0.5)

                cost = mean_int + dist_center * 40
                if cost < min_cost:
                    min_cost = cost
                    best_circle = (int(x), int(y), int(r))

        if best_circle is not None:
            return best_circle

    # Direct Hough on grayscale blurred as secondary attempt
    circles_gray = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=h // 4,
        param1=60,
        param2=20,
        minRadius=min_radius,
        maxRadius=max_radius
    )
    if circles_gray is not None:
        circles_gray = np.round(circles_gray[0, :]).astype(int)
        best_c = None
        min_int = float('inf')
        for (x, y, r) in circles_gray:
            mask = np.zeros_like(gray_img, dtype=np.uint8)
            cv2.circle(mask, (x, y), max(2, int(r * 0.7)), 255, -1)
            mean_int = cv2.mean(gray_img, mask=mask)[0]
            if mean_int < min_int:
                min_int = mean_int
                best_c = (int(x), int(y), int(r))
        if best_c:
            return best_c

    # Contour-based fallback
    contours, _ = cv2.findContours(thresh_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        valid_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            if perimeter > 0 and (np.pi * (min_radius**2) * 0.4) < area < (np.pi * (max_radius**2) * 1.6):
                (cx, cy), radius = cv2.minEnclosingCircle(c)
                valid_contours.append((area, int(cx), int(cy), int(radius)))

        if valid_contours:
            valid_contours.sort(key=lambda item: item[0], reverse=True)
            _, cx, cy, radius = valid_contours[0]
            return (cx, cy, max(min_radius, min(radius, max_radius)))

    return (w // 2, h // 2, 34)


def detect_iris_hough(gray_img: np.ndarray, 
                      pupil_circle: Tuple[int, int, int],
                      min_ratio: float = 1.7, 
                      max_ratio: float = 3.6) -> Tuple[int, int, int]:
    """
    Detects the outer iris/limbic boundary using Circular Hough Transform
    constrained by the pupil location and biological iris-to-pupil radius ratio.
    """
    px, py, pr = pupil_circle
    h, w = gray_img.shape[:2]

    min_iris_r = int(pr * min_ratio)
    max_iris_r = int(min(pr * max_ratio, min(h, w) // 2 - 5))

    if min_iris_r >= max_iris_r:
        max_iris_r = min_iris_r + 25

    # Gaussian blur for outer boundary
    blurred = cv2.GaussianBlur(gray_img, (9, 9), 0)

    # Edge detection emphasizing vertical/radial gradients to avoid eyelids
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    edge_mag = np.sqrt(sobel_x**2 + 0.4 * (sobel_y**2))
    edge_mag = np.uint8(np.clip(edge_mag / (edge_mag.max() + 1e-6) * 255, 0, 255))

    edges = cv2.Canny(edge_mag, 20, 60)

    # Suppress pupil interior edges so only outer iris boundary is evaluated
    cv2.circle(edges, (px, py), int(pr * 1.3), 0, -1)

    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=h // 4,
        param1=45,
        param2=22,
        minRadius=min_iris_r,
        maxRadius=max_iris_r
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        best_circle = None
        min_cost = float('inf')

        for (x, y, r) in circles:
            dist_to_pupil = np.sqrt((x - px)**2 + (y - py)**2)
            # Must be roughly concentric with pupil
            if dist_to_pupil < pr * 0.45 and min_iris_r <= r <= max_iris_r:
                cost = dist_to_pupil + abs(r - pr * 2.5) * 0.2
                if cost < min_cost:
                    min_cost = cost
                    best_circle = (int(x), int(y), int(r))

        if best_circle is not None:
            return best_circle

    # Analytical fallback: iris center is nearly coincident with pupil
    estimated_r = int(np.clip(pr * 2.45, min_iris_r, max_iris_r))
    return (px, py, estimated_r)


def localize_iris(image: np.ndarray) -> Dict[str, Tuple[int, int, int]]:
    """
    Full localization pipeline for iris and pupil boundaries.

    Args:
        image: BGR or Grayscale input eye image

    Returns:
        Dictionary containing:
            - 'pupil': (x, y, r)
            - 'iris': (x, y, r)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    pupil = detect_pupil_hough(gray)
    iris = detect_iris_hough(gray, pupil)

    # Ensure iris radius is strictly larger than pupil radius
    if iris[2] <= pupil[2]:
        iris = (iris[0], iris[1], int(pupil[2] * 2.3))

    return {
        'pupil': pupil,
        'iris': iris
    }


def draw_boundaries(image: np.ndarray, 
                    pupil_circle: Tuple[int, int, int], 
                    iris_circle: Tuple[int, int, int],
                    draw_labels: bool = True) -> np.ndarray:
    """
    Draws inner pupil and outer iris boundaries on the image.
    Pupil is drawn in Green, Iris in Bright Cyan/Blue.
    """
    if len(image.shape) == 2:
        out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        out = image.copy()

    px, py, pr = pupil_circle
    ix, iy, ir = iris_circle

    # Draw outer iris boundary (Cyan: 255, 200, 0 in BGR)
    cv2.circle(out, (ix, iy), ir, (255, 215, 0), 2, cv2.LINE_AA)
    cv2.circle(out, (ix, iy), 3, (255, 215, 0), -1)

    # Draw inner pupil boundary (Green: 0, 255, 0 in BGR)
    cv2.circle(out, (px, py), pr, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.circle(out, (px, py), 3, (0, 255, 0), -1)

    if draw_labels:
        # Pupil text
        cv2.putText(out, f"Pupil: ({px},{py}) r={pr}", 
                    (max(10, px - pr - 10), max(20, py - pr - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

        # Iris text
        cv2.putText(out, f"Iris: ({ix},{iy}) r={ir}", 
                    (max(10, ix - ir - 10), max(40, iy - ir - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 215, 0), 1, cv2.LINE_AA)

    return out
