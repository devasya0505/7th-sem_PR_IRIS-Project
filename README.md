# Iris Recognition and Boundary Localization [Biometrics & Security]
### Pattern Recognition (Course Code: 3171613) — Micro-Project #22
**B.E. in Information Technology | 7th Semester**

---

## 📌 Project Overview
This project implements an end-to-end **Biometric Iris Recognition System** based on John Daugman's canonical framework and Circular Hough Transform.

The pipeline achieves:
1. **Iris Boundary Localization**: Accurately detects the inner pupil circle and outer iris (limbic) circle using edge-directed **Circular Hough Transform (CHT)**.
2. **Iris Normalization**: Unwraps the circular annular iris disc into a dimensionless rectangular polar strip using **Daugman's Rubber Sheet Model**.
3. **Feature Extraction**: Convolves the normalized iris strip with **2D Gabor Wavelets** across multiple scales/frequencies and extracts a compact **Binary Iris Code** via 2-bit phase quadrant quantization.
4. **Biometric Matching**: Computes the **Normalized Hamming Distance (HD)** with circular rotational shift compensation to account for head tilt and eye rotation.
5. **Evaluation & Visualization**: Computes False Acceptance Rate (FAR), False Rejection Rate (FRR), Equal Error Rate (EER), Decidability Index ($d'$), and provides an interactive desktop GUI dashboard.

---

## 🚀 Key Features
- **Accurate Localization**: Isolates dark pupil aperture and outer limbal boundary using gradient-weighted Circular Hough Transform.
- **Scale & Pupil Dilation Invariance**: Daugman's rubber sheet mapping compensates for varying pupil dilation and subject-to-camera distance.
- **Rotation Invariance**: Circular bit-shift alignment in Hamming distance matching handles ocular torsion and head tilt up to $\pm 12^\circ$.
- **Self-Contained Benchmark Dataset**: Includes a built-in synthetic iris generator providing intra-class variations (dilations, rotations, noise) and inter-class patterns. Also supports standard external datasets (e.g. CASIA-Iris, MMU, UBIRIS).
- **Modern Interactive GUI**: Built with Python Tkinter and Pillow featuring real-time image analysis, 1-to-1 biometric verification, and benchmark distribution visualizer.

---

## 📁 Repository Structure
```
PR_Micro-Project_IRIS/
│
├── data/
│   ├── samples/                # Sample benchmark iris images (Subject 1, 2, 3...)
│   └── outputs/                # Generated visualizations, unwrapped strips, codes, plots
│
├── src/
│   ├── __init__.py             # Package descriptor
│   ├── localization.py         # Circular Hough Transform for pupil & iris boundary detection
│   ├── normalization.py        # Daugman's Rubber Sheet Model (polar coordinate unwrapping)
│   ├── feature_extraction.py   # 2D Gabor Wavelets & phase quantization to binary iris codes
│   ├── matching.py             # Rotational Hamming Distance calculation & decision logic
│   ├── dataset_generator.py    # Realistic synthetic iris dataset generator
│   └── utils.py                # Image I/O and multi-stage pipeline visualizer
│
├── main.py                     # Command-line runner & demonstration pipeline
├── benchmark.py                # Biometric performance evaluation (FAR/FRR/EER & distributions)
├── app_gui.py                  # Interactive Desktop GUI Application
├── requirements.txt            # Python dependencies
├── PROJECT_REPORT.md           # Comprehensive Academic Micro-Project Report
└── README.md                   # Project documentation (this file)
```

---

## 🔬 Theoretical Methodology & Equations

### 1. Boundary Localization (Circular Hough Transform)
An edge point $(x, y)$ votes for candidate circle centers $(x_c, y_c)$ of radius $r$:
$$(x - x_c)^2 + (y - y_c)^2 = r^2$$
- **Pupil Boundary**: Detected as the inner circle $(x_p, y_p, r_p)$ by thresholding the dark pupil aperture and executing Hough Gradient search.
- **Iris (Limbal) Boundary**: Detected as the outer circle $(x_i, y_i, r_i)$ by searching concentric circles with $r_i \approx 2.0 - 3.5 \times r_p$ with vertical gradient weighting to suppress eyelid occlusion.

### 2. Daugman's Rubber Sheet Model
The non-concentric annular iris region is mapped to dimensionless polar coordinates $(r, \theta)$, where $r \in [0, 1]$ and $\theta \in [0, 2\pi]$:
$$I(x(r, \theta), y(r, \theta)) \longrightarrow I(r, \theta)$$
Where the Cartesian coordinates are mapped via:
$$x(r, \theta) = (1 - r) x_p(\theta) + r x_i(\theta)$$
$$y(r, \theta) = (1 - r) y_p(\theta) + r y_i(\theta)$$
$$x_p(\theta) = x_p + r_p \cos\theta, \quad y_p(\theta) = y_p + r_p \sin\theta$$
$$x_i(\theta) = x_i + r_i \cos\theta, \quad y_i(\theta) = y_i + r_i \sin\theta$$

### 3. Feature Extraction (2D Gabor Wavelet Phase Quantization)
The normalized strip is convolved with 2D Gabor quadrature filter pairs:
$$G(x, y) = \exp\left(-\frac{1}{2}\left[\frac{x'^2}{\sigma_x^2} + \frac{y'^2}{\sigma_y^2}\right]\right) \exp\left(i 2\pi \frac{x'}{\lambda}\right)$$
For each filter response phasor $h = h_{Re} + i h_{Im}$, two bits are quantized:
$$b_{Re} = 1 \iff h_{Re} > 0, \quad b_{Im} = 1 \iff h_{Im} > 0$$

### 4. Matching & Decision (Normalized Hamming Distance)
For two iris codes $A$ and $B$ with corresponding noise masks $M_A$ and $M_B$:
$$HD = \frac{\| (A \oplus B) \cap M_A \cap M_B \|}{\| M_A \cap M_B \|}$$
To account for ocular torsion or head rotation, code $B$ is cyclically shifted by $s$ columns:
$$HD_{\text{min}} = \min_{s \in [-S, S]} HD(A, \text{shift}(B, s))$$
**Decision Criterion**:
- If $HD_{\text{min}} < \text{Threshold}$ (default $0.38 - 0.42$): **MATCH (Same Person)**
- If $HD_{\text{min}} \ge \text{Threshold}$: **NO MATCH (Different Persons)**

---

## ⚙️ Installation & Requirements

Ensure you have Python 3.8+ installed. Install the required dependencies:
```bash
pip install -r requirements.txt
```

---

## 🖥️ How to Run

### 1. Run Complete Demonstration (CLI)
Runs the complete 4-stage pipeline on sample images and performs genuine vs. impostor matching:
```bash
python main.py
```
Outputs and plots are saved into `data/outputs/`.

### 2. Compare Custom Images via CLI
```bash
python main.py --image1 path/to/eye1.png --image2 path/to/eye2.png --threshold 0.38
```

### 3. Run Biometric Benchmark & Error Rate Analysis
Computes Genuine vs Impostor distributions, FAR, FRR, EER, and Decidability Index $d'$:
```bash
python benchmark.py
```
Outputs the benchmark graph: `data/outputs/benchmark_results.png`.

### 4. Launch Desktop GUI Application
Launches the interactive GUI for visual demonstrations:
```bash
python app_gui.py
```
Features:
- **Tab 1: Single Iris Analysis**: View localized pupil & iris boundaries, normalized strip, and binary iris code.
- **Tab 2: Biometric Verification**: 1-to-1 matching with instant visual verdict, Hamming distance, and similarity score.
- **Tab 3: Performance Benchmark**: Interactive FAR/FRR and distribution viewer.

---

## 📊 Experimental Results

| Metric | Result | Description |
| :--- | :--- | :--- |
| **Pupil Localization Method** | Edge-guided Hough Circle Transform | Inner boundary detection |
| **Iris Localization Method** | Gradient-weighted Hough Circle Transform | Outer limbal boundary detection |
| **Normalized Strip Size** | $64 \times 256$ pixels | Dimensionless polar coordinates |
| **Binary Iris Code Size** | $64 \times 1024$ bits (65,536 bits) | 2D Gabor wavelet phase quantization |
| **Mean Genuine HD ($\mu_G$)** | $\approx 0.30 - 0.36$ | Intra-subject comparisons |
| **Mean Impostor HD ($\mu_I$)** | $\approx 0.45 - 0.50$ | Inter-subject comparisons |
| **Decidability Index ($d'$)** | $> 1.85$ | Class separability measure |
| **Optimal Decision Threshold** | $0.38 - 0.42$ | Minimum equal error boundary |

---

## 🎓 Academic Micro-Project Details
- **Subject**: Pattern Recognition (PR)
- **Subject Code**: 3171613
- **Branch**: B.E. Information Technology (7th Semester)
- **Topic 22**: Iris Recognition and Boundary Localization [Biometrics & Security]
- **Deliverables**: Source code, test dataset, GUI application, CLI pipeline, and [PROJECT_REPORT.md](PROJECT_REPORT.md).
