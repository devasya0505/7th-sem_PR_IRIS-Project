"""
Iris Feature Extraction Module
Applies 2D Gabor Wavelet Filters and Phase Quantization to extract
compact binary Iris Codes (Daugman's Algorithm).
"""

import cv2
import numpy as np
from typing import Tuple, Dict


def create_gabor_filter(ksize: int = 15, 
                        sigma: float = 3.0, 
                        theta: float = 0.0, 
                        lambd: float = 8.0, 
                        gamma: float = 0.5, 
                        psi: float = 0.0) -> np.ndarray:
    """Creates a spatial Gabor filter kernel."""
    return cv2.getGaborKernel(
        (ksize, ksize),
        sigma=sigma,
        theta=theta,
        lambd=lambd,
        gamma=gamma,
        psi=psi,
        ktype=cv2.CV_32F
    )


def extract_iris_code(normalized_iris: np.ndarray,
                      noise_mask: np.ndarray = None,
                      wavelengths: Tuple[int, ...] = (8, 16)) -> Dict[str, np.ndarray]:
    """
    Extracts binary iris codes by applying quadrature 2D Gabor filters
    and performing 2-bit phase quantization (Real and Imaginary parts).

    Args:
        normalized_iris: 2D array of normalized iris strip (H x W)
        noise_mask: 2D array of same shape (1 = valid, 0 = noise/occluded)
        wavelengths: List of spatial frequencies / wavelengths to sample

    Returns:
        Dictionary containing:
            - 'iris_code': 2D binary numpy array (uint8) of shape (H, W * 2 * num_filters)
            - 'code_mask': 2D binary numpy array of same shape
            - 'flat_code': 1D binary vector
            - 'flat_mask': 1D binary vector
    """
    h, w = normalized_iris.shape[:2]
    if noise_mask is None:
        noise_mask = np.ones((h, w), dtype=np.uint8)

    iris_float = normalized_iris.astype(np.float32)
    # Subtract local mean to eliminate DC bias
    mean_val = cv2.boxFilter(iris_float, -1, (15, 15))
    iris_diff = iris_float - mean_val

    code_bands = []
    mask_bands = []

    for wl in wavelengths:
        # Real part: psi = 0 (even-symmetric cosine filter)
        kernel_real = create_gabor_filter(ksize=17, sigma=3.5, theta=0.0, lambd=wl, gamma=0.5, psi=0.0)
        # Imaginary part: psi = -pi/2 (odd-symmetric sine filter)
        kernel_imag = create_gabor_filter(ksize=17, sigma=3.5, theta=0.0, lambd=wl, gamma=0.5, psi=-np.pi / 2.0)

        # Convolve with normalized iris
        resp_real = cv2.filter2D(iris_diff, cv2.CV_32F, kernel_real)
        resp_imag = cv2.filter2D(iris_diff, cv2.CV_32F, kernel_imag)

        # 2-bit Phase Quantization:
        # Quadrant 1: Real > 0, Imag > 0  -> (1, 1)
        # Quadrant 2: Real <= 0, Imag > 0 -> (0, 1)
        # Quadrant 3: Real <= 0, Imag <= 0 -> (0, 0)
        # Quadrant 4: Real > 0, Imag <= 0 -> (1, 0)
        bit_real = (resp_real > 0).astype(np.uint8)
        bit_imag = (resp_imag > 0).astype(np.uint8)

        code_bands.append(bit_real)
        code_bands.append(bit_imag)

        # Replicate noise mask for both bits
        mask_bands.append(noise_mask)
        mask_bands.append(noise_mask)

    # Interleave / horizontally stack the feature bands
    iris_code = np.hstack(code_bands)
    code_mask = np.hstack(mask_bands)

    return {
        'iris_code': iris_code,
        'code_mask': code_mask,
        'flat_code': iris_code.flatten(),
        'flat_mask': code_mask.flatten()
    }
