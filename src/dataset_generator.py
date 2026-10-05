"""
Iris Dataset Generator Module
Generates high-fidelity synthetic eye images with distinctive iris textures,
pupil, sclera, collarette, and natural intra-class variations (dilations, rotations, noise).
Enables out-of-the-box self-contained evaluation and demonstrations.
"""

import os
import cv2
import numpy as np
from typing import List


def generate_synthetic_iris_image(subject_id: int, 
                                  capture_id: int, 
                                  img_size: int = 320,
                                  pupil_dilation: float = 1.0,
                                  rotation_deg: float = 0.0,
                                  noise_level: float = 8.0) -> np.ndarray:
    """
    Generates a realistic iris/eye image with unique identity-based features.
    """
    # Deterministic base seed per subject
    rng = np.random.RandomState(subject_id * 1000 + 42)
    
    # Capture specific random variation
    cap_rng = np.random.RandomState(subject_id * 1000 + capture_id * 100 + 7)

    h = w = img_size
    cx = w // 2 + int(cap_rng.uniform(-3, 3))
    cy = h // 2 + int(cap_rng.uniform(-3, 3))

    # Base pupil & iris dimensions
    base_pr = 34 + int(rng.uniform(-4, 6))
    pr = int(base_pr * pupil_dilation)
    ir = int(pr * 2.5 + rng.uniform(-2, 4))

    # Create coordinate grid
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    theta = np.arctan2(y - cy, x - cx) + np.radians(rotation_deg)

    # Canvas initialization (Sclera - off-white with slight gradient)
    sclera_base = 220 + 15 * np.cos(theta) - 10 * (r / (w / 2))
    img = np.clip(sclera_base, 180, 240).astype(np.float32)

    # Add slight sclera blood vessel / texture streaks
    vessel_noise = cap_rng.normal(0, 3, (h, w))
    img += vessel_noise

    # Subject specific texture frequencies
    num_spokes = rng.randint(45, 75)
    freq1 = rng.uniform(8.0, 15.0)
    freq2 = rng.uniform(20.0, 35.0)
    phase1 = rng.uniform(0, 2 * np.pi)
    phase2 = rng.uniform(0, 2 * np.pi)
    collarette_r = pr + (ir - pr) * rng.uniform(0.35, 0.45)

    # Iris mask
    iris_mask = (r >= pr) & (r <= ir)

    # Normalized radius inside iris [0, 1]
    r_norm = np.clip((r - pr) / (ir - pr + 1e-5), 0, 1)

    # Iris pattern synthesis:
    # 1. Radial fibers (trabecular meshwork)
    fiber_pattern = np.cos(num_spokes * theta + phase1) * 0.5 + 0.5
    # 2. Concentric contraction furrows
    furrow_pattern = np.sin(freq1 * r_norm * 2 * np.pi + phase2) * 0.5 + 0.5
    # 3. Collarette boundary modulation
    collarette = np.exp(-((r - collarette_r)**2) / (2.0 * (6.0**2))) * 0.6
    # 4. Crypts and freckles (high frequency speckle)
    speckle = np.sin(freq2 * theta * 5 + r_norm * 12) * 0.3

    # Composite iris texture
    iris_tex = 70 + 75 * fiber_pattern + 30 * furrow_pattern - 40 * collarette + 20 * speckle

    # Inner collarette zone is slightly darker/different tone
    inner_zone = (r < collarette_r) & iris_mask
    iris_tex[inner_zone] = iris_tex[inner_zone] * 0.85 + 10

    # Smooth transition at iris outer limbus
    limbic_edge = np.exp(-((r - ir)**2) / (2.0 * (4.5**2)))
    iris_tex = iris_tex * (1.0 - 0.4 * limbic_edge)

    # Apply iris texture to image
    img[iris_mask] = iris_tex[iris_mask]

    # Dark Pupil region
    pupil_mask = r < pr
    pupil_val = 25 + cap_rng.normal(0, 2, (h, w))
    img[pupil_mask] = pupil_val[pupil_mask]

    # Soft transition at pupil boundary
    pupil_edge = (r >= pr - 2) & (r <= pr + 2)
    alpha = (r[pupil_edge] - (pr - 2)) / 4.0
    img[pupil_edge] = (1 - alpha) * 25 + alpha * img[pupil_edge]

    # Specular corneal reflection (bright glint)
    glint_x, glint_y = cx + int(pr * 0.35), cy - int(pr * 0.35)
    glint_dist = np.sqrt((x - glint_x)**2 + (y - glint_y)**2)
    glint = np.exp(-(glint_dist**2) / (2.0 * (2.8**2))) * 230
    img = np.clip(img + glint, 0, 255)

    # Add realistic sensor noise
    sensor_noise = cap_rng.normal(0, noise_level, (h, w))
    img = np.clip(img + sensor_noise, 0, 255).astype(np.uint8)

    return img


def create_sample_dataset(output_dir: str, num_subjects: int = 5, captures_per_subject: int = 3) -> List[str]:
    """
    Creates a sample dataset of test eyes with genuine and impostor pairs.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_paths = []

    dilation_factors = [1.0, 1.08, 0.94, 1.05]
    rotations = [0.0, 2.5, -3.0, 1.5]

    for subj in range(1, num_subjects + 1):
        for cap in range(1, captures_per_subject + 1):
            dil = dilation_factors[(cap - 1) % len(dilation_factors)]
            rot = rotations[(cap - 1) % len(rotations)]
            
            img = generate_synthetic_iris_image(
                subject_id=subj,
                capture_id=cap,
                pupil_dilation=dil,
                rotation_deg=rot,
                noise_level=6.0
            )

            filename = f"subject_{subj:02d}_eye_{cap:02d}.png"
            path = os.path.join(output_dir, filename)
            cv2.imwrite(path, img)
            file_paths.append(path)

    return file_paths


if __name__ == "__main__":
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "samples")
    paths = create_sample_dataset(test_dir, num_subjects=5, captures_per_subject=3)
    print(f"Generated {len(paths)} iris samples in {test_dir}")
