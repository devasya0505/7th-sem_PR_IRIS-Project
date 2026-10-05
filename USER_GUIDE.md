# First-Time User Guide: Iris Recognition & Boundary Localization
### Pattern Recognition Micro-Project (Course Code: 3171613) — Topic 22

Welcome to the **Iris Recognition & Boundary Localization** project! This guide walks you through setup, running the interactive GUI, executing CLI pipelines, evaluating biometric benchmarks, and re-compiling the PDF report.

---

## 📋 Table of Contents
1. [Prerequisites & System Requirements](#1-prerequisites--system-requirements)
2. [Step-by-Step Installation](#2-step-by-step-installation)
3. [Running the Project (3 Easy Methods)](#3-running-the-project-3-easy-methods)
   - [Method 1: Interactive Desktop GUI (Recommended)](#method-1-interactive-desktop-gui-recommended)
   - [Method 2: Command Line Interface (CLI)](#method-2-command-line-interface-cli)
   - [Method 3: Biometric Benchmark & ROC Evaluator](#method-3-biometric-benchmark--roc-evaluator)
4. [Using Custom or External Iris Datasets](#4-using-custom-or-external-iris-datasets)
5. [Re-Generating the Academic PDF Report](#5-re-generating-the-academic-pdf-report)
6. [Understanding the Output Metrics](#6-understanding-the-output-metrics)
7. [Troubleshooting & FAQs](#7-troubleshooting--faqs)

---

## 1. Prerequisites & System Requirements
- **Operating System:** Windows 10/11, macOS, or Linux.
- **Python:** Python 3.8 to 3.14 installed.
- **Hardware:** Standard CPU (runs in real-time, no GPU required).

Verify your Python installation in your terminal:
```powershell
python --version
pip --version
```

---

## 2. Step-by-Step Installation

### Step 2.1: Open Terminal in Project Directory
Open PowerShell, Command Prompt, or your IDE terminal inside the project root folder:
```powershell
cd "e:\1_B.E. in IT\7th Semester\3171613_Pattern Recognition (PR)\PR_Micro-Project_IRIS"
```

### Step 2.2: (Optional but Recommended) Create Virtual Environment
```powershell
# Create virtual environment named 'venv'
python -m venv venv

# Activate it on Windows:
.\venv\Scripts\Activate.ps1
# (or in CMD: .\venv\Scripts\activate.bat)
```

### Step 2.3: Install Dependencies
Install all required libraries (`opencv-python`, `numpy`, `matplotlib`, `scipy`, `pillow`, `reportlab`, `markdown`):
```powershell
pip install -r requirements.txt
```

---

## 3. Running the Project (3 Easy Methods)

### Method 1: Interactive Desktop GUI (Recommended)
The GUI is the easiest and most visually impressive way to demonstrate the project to evaluators.

Run:
```powershell
python app_gui.py
```

#### What you can do inside the GUI:
1. **Tab 1: Single Iris Analysis & Localization**:
   - Select an iris image from the sample dropdown (e.g. `subject_01_eye_01.png`), or click **"Browse Custom Image..."** to pick any photo.
   - Click **"Run Analysis"**.
   - **What you will see:**
     - **Card 1:** Original eye image showing the detected inner pupil circle (Green) and outer iris limbus circle (Cyan) with center coordinates and radii.
     - **Card 2:** Unwrapped rectangular strip (Daugman's Rubber Sheet Normalization).
     - **Card 3:** Binary Iris Code bitmap ($64 \times 1024$ bits).
2. **Tab 2: Biometric Verification (1-to-1 Match)**:
   - Choose **Probe (Image 1)** and **Gallery (Image 2)**.
   - Try selecting **Same Subject** (e.g. `subject_01_eye_01.png` and `subject_01_eye_02.png`) $\to$ Click **"Compare & Verify"**.
     - Notice the green badge: `[✓ BIOMETRIC MATCH: SAMPLES BELONG TO THE SAME SUBJECT]` with Hamming Distance $\approx 0.30$.
   - Try selecting **Different Subjects** (e.g. `subject_01_eye_01.png` and `subject_02_eye_01.png`) $\to$ Click **"Compare & Verify"**.
     - Notice the red badge: `[✗ NO MATCH: SAMPLES BELONG TO DIFFERENT SUBJECTS]` with Hamming Distance $\approx 0.45$.
   - Adjust the **Decision Threshold** slider (default: `0.38`) to test strict vs. lenient security policies.
3. **Tab 3: Dataset Benchmark & Performance**:
   - Click **"Run Biometric Benchmark (FAR/FRR & ROC)"**.
   - View genuine vs. impostor Hamming Distance histograms and FAR/FRR trade-off curves loaded directly on screen.

---

### Method 2: Command Line Interface (CLI)

#### 2.1 Default Automated Demonstration
Runs the full 4-stage pipeline and executes genuine and impostor comparisons automatically:
```powershell
python main.py
```
*Outputs are saved to `data/outputs/`:*
- `demo_s01_e01_localized.png`: Boundaries drawn on image.
- `demo_s01_e01_normalized.png`: Unwrapped polar strip.
- `demo_s01_e01_iriscode.png`: Binary code bitmap.
- `demo_s01_e01_pipeline.png`: 4-in-1 multi-stage summary figure.

#### 2.2 Compare Any Two Custom Images
```powershell
python main.py --image1 "data/samples/subject_01_eye_01.png" --image2 "data/samples/subject_01_eye_02.png" --threshold 0.38
```

#### 2.3 Process a Single Image & Save Visualizations
```powershell
python main.py --image1 "path/to/eye.jpg"
```

#### 2.4 Regenerate the Sample Benchmark Dataset
```powershell
python main.py --generate-dataset
```

---

### Method 3: Biometric Benchmark & ROC Evaluator

To compute statistical biometric curves and error rates across the entire dataset:
```powershell
python benchmark.py
```

**Key Metrics Output:**
- Total pairwise comparisons (Genuine pairs vs. Impostor pairs).
- Mean and standard deviation for Genuine ($\mu_G$) and Impostor ($\mu_I$).
- **Daugman's Decidability Index ($d'$)**: Measures class separability.
- **Equal Error Rate (EER)** and **Optimal Threshold**.
- Saves the publication-quality graph to:
  `data/outputs/benchmark_results.png`

---

## 4. Using Custom or External Iris Datasets
You can use external public datasets such as **CASIA-Iris**, **MMU**, or **UBIRIS**:
1. Copy your `.png`, `.jpg`, or `.bmp` eye images into:
   `data/samples/`
2. Name your images with an identifier prefix, for example:
   - `subject_01_1.jpg`, `subject_01_2.jpg` (Subject 1)
   - `subject_02_1.jpg`, `subject_02_2.jpg` (Subject 2)
3. Launch `app_gui.py` or run `benchmark.py` — they will automatically load your new images!

---

## 5. Re-Generating the Academic PDF Report

The project includes an automatic PDF generator that builds a report containing all math, figures, and benchmark graphs:

```powershell
python generate_pdf_report.py
```

The compiled report is saved at:
[`PROJECT_REPORT.pdf`](PROJECT_REPORT.pdf)

You can print or submit this PDF directly for your college evaluation.

---

## 6. Understanding the Output Metrics

| Metric | What It Means | Typical Value Range |
| :--- | :--- | :--- |
| **Hamming Distance (HD)** | Fraction of differing bits between two iris codes ($0.0$ = identical, $0.5$ = random noise). | **Genuine:** $0.15 - 0.36$<br/>**Impostor:** $0.44 - 0.52$ |
| **Decision Threshold ($\tau$)** | Cutoff below which two images are classified as the same person. | **$0.38 - 0.42$** |
| **Rotational Bit Shift** | Cyclic column shift compensating for ocular torsion or head tilt ($\pm 11.25^\circ$). | $-8$ to $+8$ columns |
| **Biometric Similarity %** | Scaled match percentage: $(1 - HD) \times 100\%$. | **Genuine:** $> 65\%$<br/>**Impostor:** $< 55\%$ |
| **Decidability ($d'$)** | Statistical separation between genuine and impostor score distributions. | $> 1.8$ indicates solid biometric discriminability. |
| **EER (Equal Error Rate)** | Operating point where False Acceptance Rate equals False Rejection Rate. | Ideal: Lowest possible |

---

## 7. Troubleshooting & FAQs

### Q1: `ModuleNotFoundError: No module named 'cv2'`
**Fix:** Run `pip install opencv-python`.

### Q2: `ModuleNotFoundError: No module named 'reportlab'`
**Fix:** Run `pip install reportlab markdown`.

### Q3: My custom image failed to detect the pupil. Why?
**Fix:**
- Ensure the image has reasonable contrast (pupil should be darker than the iris).
- Check that the pupil is not completely occluded by eyelids or strong glare.
- The algorithm automatically applies adaptive thresholding, but extremely dark or overexposed images should be pre-cropped to focus on the eye.

### Q4: How do I change the matching threshold?
- In the GUI: Drag the **"Decision Threshold"** slider in Tab 2.
- In CLI: Pass `--threshold 0.40` to `python main.py`.

---

## 📂 Project Summary Reference

| File | Description |
| :--- | :--- |
| [`app_gui.py`](app_gui.py) | **Desktop Graphical Application** (Tkinter dark theme) |
| [`main.py`](main.py) | **CLI Pipeline** runner and pairwise comparator |
| [`benchmark.py`](benchmark.py) | **FAR/FRR/EER and ROC Curve** biometric evaluator |
| [`generate_pdf_report.py`](generate_pdf_report.py) | **Academic PDF Generator** |
| [`PROJECT_REPORT.pdf`](PROJECT_REPORT.pdf) | **Compiled Academic Project Report** |
| [`PROJECT_REPORT.md`](PROJECT_REPORT.md) | **Markdown Project Report** source |
| [`README.md`](README.md) | Technical overview and equations |
| [`requirements.txt`](requirements.txt) | Python dependencies list |
| [`.gitignore`](.gitignore) | Git exclusions for clean repository tracking |
