"""
Main CLI Application for Iris Recognition and Boundary Localization
Pattern Recognition Micro-Project (Course Code: 3171613)

Usage:
    python main.py
    python main.py --image1 path/to/eye1.png --image2 path/to/eye2.png
    python main.py --generate-dataset
"""

import os
import sys
import argparse
import cv2
import numpy as np

from src.localization import localize_iris, draw_boundaries
from src.normalization import daugman_rubber_sheet
from src.feature_extraction import extract_iris_code
from src.matching import match_iris_codes
from src.dataset_generator import create_sample_dataset
from src.utils import load_image, save_image, visualize_pipeline_steps


def process_iris_pipeline(image_path: str, save_visuals: bool = True, output_prefix: str = "demo"):
    """Runs complete recognition pipeline for a single iris image."""
    print(f"\n[+] Processing image: {image_path}")
    gray_img = load_image(image_path, as_gray=True)

    # 1. Boundary Localization using Hough Transform
    loc_results = localize_iris(gray_img)
    pupil = loc_results['pupil']
    iris = loc_results['iris']

    print(f"    - Pupil localized (Hough): Center=({pupil[0]}, {pupil[1]}), Radius={pupil[2]} px")
    print(f"    - Iris  localized (Hough): Center=({iris[0]}, {iris[1]}), Radius={iris[2]} px")

    # 2. Daugman's Rubber Sheet Normalization
    norm_iris, mask = daugman_rubber_sheet(gray_img, pupil, iris, radial_res=64, angular_res=256)
    print(f"    - Iris Normalized: Unwrapped polar strip size = {norm_iris.shape[0]}x{norm_iris.shape[1]}")

    # 3. Feature Extraction (Gabor Wavelets & Phase Quantization)
    features = extract_iris_code(norm_iris, mask, wavelengths=(8, 16))
    iris_code = features['iris_code']
    code_mask = features['code_mask']
    print(f"    - Binary Iris Code: Generated {iris_code.shape[0]}x{iris_code.shape[1]} bits ({iris_code.size} bits total)")

    # 4. Visualization & Saving
    if save_visuals:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "outputs")
        os.makedirs(out_dir, exist_ok=True)

        # Draw localized boundaries
        annotated_img = draw_boundaries(cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR), pupil, iris)
        save_image(os.path.join(out_dir, f"{output_prefix}_localized.png"), annotated_img)
        save_image(os.path.join(out_dir, f"{output_prefix}_normalized.png"), norm_iris)
        
        # Save visual iris code (0 -> 0, 1 -> 255)
        code_vis = (iris_code * 255).astype(np.uint8)
        save_image(os.path.join(out_dir, f"{output_prefix}_iriscode.png"), code_vis)

        # Complete 4-step diagram
        pipeline_plot_path = os.path.join(out_dir, f"{output_prefix}_pipeline.png")
        visualize_pipeline_steps(gray_img, annotated_img, norm_iris, iris_code, 
                                 title=f"Iris Recognition Pipeline - {os.path.basename(image_path)}",
                                 save_path=pipeline_plot_path)
        print(f"    - Saved pipeline visualization figure to: {pipeline_plot_path}")

    return {
        'image': gray_img,
        'pupil': pupil,
        'iris': iris,
        'norm_iris': norm_iris,
        'iris_code': iris_code,
        'code_mask': code_mask
    }


def compare_two_irises(path1: str, path2: str, threshold: float = 0.38):
    """Executes localization, feature extraction and matching between two images."""
    print("=" * 65)
    print(" IRIS RECOGNITION COMPARISON & BIOMETRIC VERIFICATION")
    print("=" * 65)

    res1 = process_iris_pipeline(path1, save_visuals=False)
    res2 = process_iris_pipeline(path2, save_visuals=False)

    match_res = match_iris_codes(
        res1['iris_code'], 
        res2['iris_code'], 
        res1['code_mask'], 
        res2['code_mask'],
        max_shift=8,
        threshold=threshold
    )

    hd = match_res['min_hamming_distance']
    is_match = match_res['is_match']
    sim = match_res['similarity_pct']
    shift = match_res['best_shift']

    print("\n" + "-" * 65)
    print(" MATCHING RESULTS:")
    print(f" Image 1: {os.path.basename(path1)}")
    print(f" Image 2: {os.path.basename(path2)}")
    print(f" Normalized Hamming Distance: {hd:.4f}  (Threshold = {threshold:.2f})")
    print(f" Best Rotational Bit Shift   : {shift} columns")
    print(f" Biometric Similarity Score  : {sim:.2f}%")
    if is_match:
        print(" VERDICT                    : [MATCH] Samples belong to the SAME SUBJECT")
    else:
        print(" VERDICT                    : [NO MATCH] Samples belong to DIFFERENT SUBJECTS")
    print("-" * 65 + "\n")

    return match_res


def run_default_demonstration():
    """Default self-contained demo that runs when no args are provided."""
    print("=" * 70)
    print(" PATTERN RECOGNITION (PR) MICRO-PROJECT: IRIS RECOGNITION")
    print(" Topic 22: Iris Recognition & Boundary Localization [Biometrics & Security]")
    print("=" * 70)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    samples_dir = os.path.join(base_dir, "data", "samples")

    # Generate sample dataset if not present
    if not os.path.exists(samples_dir) or len(os.listdir(samples_dir)) == 0:
        print("\n[*] Initializing sample iris benchmark dataset...")
        create_sample_dataset(samples_dir, num_subjects=5, captures_per_subject=3)
        print(f"[+] Created 15 sample iris images in: {samples_dir}")

    s1_e1 = os.path.join(samples_dir, "subject_01_eye_01.png")
    s1_e2 = os.path.join(samples_dir, "subject_01_eye_02.png")
    s2_e1 = os.path.join(samples_dir, "subject_02_eye_01.png")

    # 1. Pipeline demonstration on Subject 1 Eye 1
    print("\n>>> STAGE 1: Full Pipeline Demonstration (Subject 01, Eye 01)")
    process_iris_pipeline(s1_e1, save_visuals=True, output_prefix="demo_s01_e01")

    # 2. Genuine Match Comparison (Same Person)
    print("\n>>> STAGE 2: Genuine Match Test (Same Subject - Subj 01 Eye 01 vs Eye 02)")
    compare_two_irises(s1_e1, s1_e2, threshold=0.38)

    # 3. Impostor Match Comparison (Different Persons)
    print("\n>>> STAGE 3: Impostor Match Test (Different Subjects - Subj 01 Eye 01 vs Subj 02 Eye 01)")
    compare_two_irises(s1_e1, s2_e1, threshold=0.38)

    print("\n[V] Demo completed successfully!")
    print("    - Visual outputs saved to: data/outputs/")
    print("    - To evaluate full benchmark & ROC curves: python benchmark.py")
    print("    - To launch the interactive GUI dashboard:  python app_gui.py\n")


def main():
    parser = argparse.ArgumentParser(description="Iris Recognition and Boundary Localization System")
    parser.add_argument("--image1", type=str, help="Path to first iris image")
    parser.add_argument("--image2", type=str, help="Path to second iris image for matching")
    parser.add_argument("--generate-dataset", action="store_true", help="Generate synthetic sample dataset")
    parser.add_argument("--threshold", type=float, default=0.38, help="Hamming distance decision threshold")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    if args.generate_dataset:
        out_dir = os.path.join(base_dir, "data", "samples")
        paths = create_sample_dataset(out_dir, num_subjects=5, captures_per_subject=3)
        print(f"Generated {len(paths)} iris samples in: {out_dir}")
        return

    if args.image1 and args.image2:
        compare_two_irises(args.image1, args.image2, threshold=args.threshold)
    elif args.image1:
        process_iris_pipeline(args.image1, save_visuals=True)
    else:
        run_default_demonstration()


if __name__ == "__main__":
    main()
