"""
Utility Module for Iris Recognition Project
Provides image I/O, plotting, and visualization functions.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional


def load_image(file_path: str, as_gray: bool = True) -> np.ndarray:
    """Loads an image safely from disk."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    img = cv2.imread(file_path)
    if img is None:
        raise ValueError(f"Could not decode image at {file_path}")
        
    if as_gray and len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def save_image(file_path: str, img: np.ndarray) -> None:
    """Saves image to disk, ensuring directory exists."""
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    cv2.imwrite(file_path, img)


def visualize_pipeline_steps(original_img: np.ndarray,
                             localized_img: np.ndarray,
                             normalized_iris: np.ndarray,
                             iris_code: np.ndarray,
                             title: str = "Iris Recognition Pipeline",
                             save_path: Optional[str] = None) -> None:
    """
    Creates a comprehensive 4-stage pipeline visualization figure.
    Stage 1: Input Eye Image
    Stage 2: Inner Pupil & Outer Iris Boundary Localization (Hough Transform)
    Stage 3: Daugman's Rubber Sheet Normalization (Unwrapped Strip)
    Stage 4: Binary Iris Code (Gabor Wavelet Phase Quantization)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    # 1. Original Image
    if len(original_img.shape) == 3:
        axes[0, 0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    else:
        axes[0, 0].imshow(original_img, cmap='gray')
    axes[0, 0].set_title("1. Original Eye Image", fontsize=12, fontweight='semibold')
    axes[0, 0].axis('off')

    # 2. Localized Boundaries
    axes[0, 1].imshow(cv2.cvtColor(localized_img, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("2. Inner Pupil & Outer Iris Localization\n(Circular Hough Transform)", 
                         fontsize=12, fontweight='semibold')
    axes[0, 1].axis('off')

    # 3. Normalized Iris Strip
    axes[1, 0].imshow(normalized_iris, cmap='gray', aspect='auto')
    axes[1, 0].set_title("3. Daugman's Rubber Sheet Normalization\n(Polar Unwrapped Strip)", 
                         fontsize=12, fontweight='semibold')
    axes[1, 0].set_xlabel("Angular Dimension (0 to 2*pi)")
    axes[1, 0].set_ylabel("Radial Dimension (Pupil to Iris)")

    # 4. Binary Iris Code
    axes[1, 1].imshow(iris_code, cmap='binary', aspect='auto')
    axes[1, 1].set_title(f"4. Binary Iris Code ({iris_code.shape[0]}x{iris_code.shape[1]} bits)\n(2D Gabor Wavelet Phase Quantization)", 
                         fontsize=12, fontweight='semibold')
    axes[1, 1].set_xlabel("Filter Bands & Quadrature Bits")
    axes[1, 1].set_ylabel("Radial Tracks")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
