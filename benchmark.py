"""
Biometric Evaluation and Benchmark Module
Evaluates Iris Recognition system performance across the dataset.
Computes Genuine vs Impostor Hamming Distance distributions,
FAR (False Acceptance Rate), FRR (False Rejection Rate),
EER (Equal Error Rate), and ROC Curve.
"""

import os
import glob
import itertools
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

from src.localization import localize_iris
from src.normalization import daugman_rubber_sheet
from src.feature_extraction import extract_iris_code
from src.matching import match_iris_codes
from src.dataset_generator import create_sample_dataset
from src.utils import load_image


def run_benchmark(dataset_dir: str, output_plot_path: str = None) -> Dict[str, float]:
    """
    Performs all-to-all matching across dataset images to evaluate
    biometric distribution and error rates.
    """
    image_files = sorted(glob.glob(os.path.join(dataset_dir, "*.png")) + 
                        glob.glob(os.path.join(dataset_dir, "*.jpg")) +
                        glob.glob(os.path.join(dataset_dir, "*.bmp")))

    if len(image_files) < 4:
        print("[!] Too few samples found. Generating sample benchmark dataset...")
        image_files = create_sample_dataset(dataset_dir, num_subjects=6, captures_per_subject=3)

    print(f"\n[+] Extracting iris templates for {len(image_files)} images...")
    templates = []
    labels = []

    for path in image_files:
        filename = os.path.basename(path)
        # Parse subject identity from filename, e.g. "subject_01_eye_01.png"
        parts = filename.split('_')
        subject_id = parts[1] if len(parts) >= 2 else filename[:8]
        labels.append(subject_id)

        gray = load_image(path, as_gray=True)
        loc = localize_iris(gray)
        norm, mask = daugman_rubber_sheet(gray, loc['pupil'], loc['iris'])
        feat = extract_iris_code(norm, mask)
        templates.append({
            'file': filename,
            'code': feat['iris_code'],
            'mask': feat['code_mask']
        })

    print("[+] Computing pairwise matching scores (Genuine vs Impostor pairs)...")
    genuine_scores = []
    impostor_scores = []

    n = len(templates)
    pairs = list(itertools.combinations(range(n), 2))

    for idx1, idx2 in pairs:
        t1 = templates[idx1]
        t2 = templates[idx2]

        res = match_iris_codes(t1['code'], t2['code'], t1['mask'], t2['mask'], max_shift=6)
        hd = res['min_hamming_distance']

        if labels[idx1] == labels[idx2]:
            genuine_scores.append(hd)
        else:
            impostor_scores.append(hd)

    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)

    # Statistical metrics
    mu_g, std_g = float(np.mean(genuine_scores)), float(np.std(genuine_scores))
    mu_i, std_i = float(np.mean(impostor_scores)), float(np.std(impostor_scores))

    # Daugman's Decidability Index d-prime:
    # d' = |mu_i - mu_g| / sqrt(0.5 * (std_i^2 + std_g^2))
    d_prime = abs(mu_i - mu_g) / np.sqrt(0.5 * (std_i**2 + std_g**2) + 1e-8)

    # Threshold sweep for FAR & FRR
    thresholds = np.linspace(0.15, 0.55, 100)
    far_list = []
    frr_list = []

    for th in thresholds:
        # FAR = False Acceptance = Impostors with HD < th
        far = np.mean(impostor_scores < th) if len(impostor_scores) > 0 else 0.0
        # FRR = False Rejection = Genuines with HD >= th
        frr = np.mean(genuine_scores >= th) if len(genuine_scores) > 0 else 0.0
        far_list.append(far)
        frr_list.append(frr)

    far_arr = np.array(far_list)
    frr_arr = np.array(frr_list)

    # Equal Error Rate (EER) where FAR == FRR
    eer_idx = np.argmin(np.abs(far_arr - frr_arr))
    eer = float((far_arr[eer_idx] + frr_arr[eer_idx]) / 2.0)
    optimal_threshold = float(thresholds[eer_idx])

    # Plotting Benchmark Results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle("Pattern Recognition Micro-Project: Biometric Iris Verification Benchmark", 
                 fontsize=14, fontweight='bold')

    # 1. Genuine vs Impostor Score Distributions
    bins = np.linspace(0.1, 0.6, 25)
    ax1.hist(genuine_scores, bins=bins, alpha=0.65, color='forestgreen', edgecolor='black', 
             label=f'Genuine (n={len(genuine_scores)}, $\\mu$={mu_g:.3f}, $\\sigma$={std_g:.3f})', density=True)
    ax1.hist(impostor_scores, bins=bins, alpha=0.65, color='crimson', edgecolor='black', 
             label=f'Impostor (n={len(impostor_scores)}, $\\mu$={mu_i:.3f}, $\\sigma$={std_i:.3f})', density=True)
    ax1.axvline(optimal_threshold, color='blue', linestyle='--', linewidth=2, 
                label=f'Optimal Threshold = {optimal_threshold:.2f}')
    ax1.set_title("Hamming Distance Distributions", fontsize=12, fontweight='semibold')
    ax1.set_xlabel("Normalized Hamming Distance")
    ax1.set_ylabel("Probability Density")
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. FAR vs FRR Curves (EER plot)
    ax2.plot(thresholds, far_arr * 100, label='FAR (False Acceptance Rate %)', color='crimson', linewidth=2)
    ax2.plot(thresholds, frr_arr * 100, label='FRR (False Rejection Rate %)', color='forestgreen', linewidth=2)
    ax2.scatter([optimal_threshold], [eer * 100], color='blue', s=70, zorder=5, 
                label=f'EER = {eer*100:.2f}% @ Thresh={optimal_threshold:.2f}')
    ax2.set_title("FAR vs. FRR Trade-off Curves", fontsize=12, fontweight='semibold')
    ax2.set_xlabel("Decision Threshold (Hamming Distance)")
    ax2.set_ylabel("Error Rate (%)")
    ax2.legend(loc='center right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_plot_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_plot_path)), exist_ok=True)
        plt.savefig(output_plot_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"[+] Benchmark figure saved to: {output_plot_path}")
    else:
        plt.show()

    results = {
        'total_images': len(image_files),
        'genuine_pairs': len(genuine_scores),
        'impostor_pairs': len(impostor_scores),
        'mu_genuine': round(mu_g, 4),
        'std_genuine': round(std_g, 4),
        'mu_impostor': round(mu_i, 4),
        'std_impostor': round(std_i, 4),
        'decidability_d_prime': round(d_prime, 4),
        'optimal_threshold': round(optimal_threshold, 4),
        'eer_percentage': round(eer * 100, 2)
    }

    print("\n" + "=" * 55)
    print(" BIOMETRIC EVALUATION SUMMARY")
    print("=" * 55)
    for k, v in results.items():
        print(f" {k:<25} : {v}")
    print("=" * 55 + "\n")

    return results


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data", "samples")
    plot_path = os.path.join(base_dir, "data", "outputs", "benchmark_results.png")
    run_benchmark(data_dir, plot_path)
