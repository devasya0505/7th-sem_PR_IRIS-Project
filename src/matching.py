"""
Iris Matching Module
Calculates Normalized Hamming Distance with rotational shift compensation
for biometric verification and identification.
"""

import numpy as np
from typing import Dict, Any


def cyclic_shift_2d(arr: np.ndarray, shift: int) -> np.ndarray:
    """Circulary shifts 2D array horizontally along angular axis."""
    return np.roll(arr, shift, axis=1)


def compute_hamming_distance(code1: np.ndarray,
                            code2: np.ndarray,
                            mask1: np.ndarray = None,
                            mask2: np.ndarray = None) -> float:
    """
    Computes fractional Hamming Distance between two binary arrays.
    HD = sum( (code1 XOR code2) AND mask1 AND mask2 ) / sum( mask1 AND mask2 )
    """
    if mask1 is None:
        mask1 = np.ones_like(code1, dtype=np.uint8)
    if mask2 is None:
        mask2 = np.ones_like(code2, dtype=np.uint8)

    xor_diff = np.bitwise_xor(code1, code2)
    common_mask = np.bitwise_and(mask1, mask2)

    total_bits = np.count_nonzero(common_mask)
    if total_bits == 0:
        return 1.0  # Max distance if no overlapping valid bits

    differing_bits = np.count_nonzero(np.bitwise_and(xor_diff, common_mask))
    return float(differing_bits / total_bits)


def match_iris_codes(code1: np.ndarray,
                     code2: np.ndarray,
                     mask1: np.ndarray = None,
                     mask2: np.ndarray = None,
                     max_shift: int = 8,
                     threshold: float = 0.38) -> Dict[str, Any]:
    """
    Compares two binary iris codes considering cyclic shifts (head tilt / eye rotation).

    Args:
        code1: 2D binary array of reference iris
        code2: 2D binary array of query iris
        mask1: 2D binary mask of reference iris
        mask2: 2D binary mask of query iris
        max_shift: Maximum cyclic column shifts to evaluate (+/- max_shift)
        threshold: Decision boundary for match (default 0.38)

    Returns:
        Dictionary with:
            - 'min_hamming_distance': Minimum HD found
            - 'best_shift': Optimal angular shift (in bits/columns)
            - 'is_match': Boolean decision (True = same subject)
            - 'similarity_pct': Percentage similarity (0-100%)
            - 'threshold': Threshold used
    """
    if mask1 is None:
        mask1 = np.ones_like(code1, dtype=np.uint8)
    if mask2 is None:
        mask2 = np.ones_like(code2, dtype=np.uint8)

    min_hd = 1.0
    best_shift = 0

    # Test shifts in range [-max_shift, max_shift]
    for s in range(-max_shift, max_shift + 1):
        shifted_c2 = cyclic_shift_2d(code2, s)
        shifted_m2 = cyclic_shift_2d(mask2, s)

        hd = compute_hamming_distance(code1, shifted_c2, mask1, shifted_m2)
        if hd < min_hd:
            min_hd = hd
            best_shift = s

    is_match = bool(min_hd < threshold)
    # Similarity percentage scaled so 0.50 (random) = 0% and 0.00 = 100%
    similarity = max(0.0, min(100.0, (1.0 - min_hd) * 100.0))

    return {
        'min_hamming_distance': round(min_hd, 4),
        'best_shift': best_shift,
        'is_match': is_match,
        'similarity_pct': round(similarity, 2),
        'threshold': threshold
    }
