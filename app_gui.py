"""
Interactive Desktop GUI for Iris Recognition and Boundary Localization
Pattern Recognition Micro-Project (Course Code: 3171613)

Provides an intuitive graphical interface for:
  - Single Iris Analysis & Hough Boundary Localization
  - 1-to-1 Iris Biometric Verification & Matching
  - Performance Benchmark Visualizer (FAR/FRR & ROC)
"""

import os
import glob
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

from src.localization import localize_iris, draw_boundaries
from src.normalization import daugman_rubber_sheet
from src.feature_extraction import extract_iris_code
from src.matching import match_iris_codes
from src.dataset_generator import create_sample_dataset
from src.utils import load_image


class IrisRecognitionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Iris Recognition & Boundary Localization - Pattern Recognition (3171613)")
        self.geometry("1180x840")
        self.minsize(1050, 750)

        # Base directories
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.samples_dir = os.path.join(self.base_dir, "data", "samples")
        self.outputs_dir = os.path.join(self.base_dir, "data", "outputs")
        os.makedirs(self.samples_dir, exist_ok=True)
        os.makedirs(self.outputs_dir, exist_ok=True)

        # Ensure sample dataset exists
        if len(glob.glob(os.path.join(self.samples_dir, "*.png"))) == 0:
            create_sample_dataset(self.samples_dir, num_subjects=5, captures_per_subject=3)

        # Style configuration
        self.setup_styles()

        # Build UI layout
        self.build_header()
        self.build_tabs()

    def setup_styles(self):
        self.configure(bg="#12161f")
        self.style = ttk.Style(self)
        self.style.theme_use('clam')

        # Colors
        self.c_bg = "#12161f"
        self.c_card = "#1a2233"
        self.c_card_border = "#2b3952"
        self.c_text = "#e6edf3"
        self.c_text_dim = "#8b949e"
        self.c_accent = "#2f81f7"
        self.c_success = "#2ea043"
        self.c_danger = "#f85149"

        # Tab style
        self.style.configure("TNotebook", background=self.c_bg, borderwidth=0)
        self.style.configure("TNotebook.Tab", background=self.c_card, foreground=self.c_text, 
                             padding=[18, 8], font=("Segoe UI", 10, "bold"), borderwidth=0)
        self.style.map("TNotebook.Tab", 
                       background=[("selected", self.c_accent)],
                       foreground=[("selected", "#ffffff")])

    def build_header(self):
        header_frame = tk.Frame(self, bg="#0d1117", height=80, padx=20, pady=12)
        header_frame.pack(fill=tk.X)

        title_lbl = tk.Label(header_frame, 
                             text="22. Iris Recognition and Boundary Localization", 
                             font=("Segoe UI", 16, "bold"), 
                             fg="#58a6ff", bg="#0d1117")
        title_lbl.pack(anchor=tk.W)

        subtitle_lbl = tk.Label(header_frame, 
                                text="GTU B.E. IT | 7th Semester | Pattern Recognition (Course: 3171613) | Biometrics & Security", 
                                font=("Segoe UI", 9), 
                                fg="#8b949e", bg="#0d1117")
        subtitle_lbl.pack(anchor=tk.W)

    def build_tabs(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=15, pady=12)

        # Tab 1: Single Iris Analysis
        self.tab_analysis = tk.Frame(self.notebook, bg=self.c_bg)
        self.notebook.add(self.tab_analysis, text="  Single Iris Analysis & Localization  ")
        self.build_analysis_tab()

        # Tab 2: Pairwise Matching
        self.tab_matching = tk.Frame(self.notebook, bg=self.c_bg)
        self.notebook.add(self.tab_matching, text="  Biometric Verification (1-to-1 Match)  ")
        self.build_matching_tab()

        # Tab 3: Dataset Benchmark
        self.tab_benchmark = tk.Frame(self.notebook, bg=self.c_bg)
        self.notebook.add(self.tab_benchmark, text="  Dataset Benchmark & Performance  ")
        self.build_benchmark_tab()

    # ==========================================================
    # TAB 1: SINGLE IRIS ANALYSIS
    # ==========================================================
    def build_analysis_tab(self):
        # Controls Frame (top)
        top_ctrl = tk.Frame(self.tab_analysis, bg=self.c_card, padx=15, pady=10, highlightbackground=self.c_card_border, highlightthickness=1)
        top_ctrl.pack(fill=tk.X, padx=10, pady=(10, 5))

        tk.Label(top_ctrl, text="Select Iris Image:", font=("Segoe UI", 10, "bold"), fg=self.c_text, bg=self.c_card).pack(side=tk.LEFT, padx=(0, 10))

        # Dropdown of sample files
        sample_files = sorted([os.path.basename(p) for p in glob.glob(os.path.join(self.samples_dir, "*.png"))])
        self.analysis_file_var = tk.StringVar(value=sample_files[0] if sample_files else "")
        self.combo_analysis = ttk.Combobox(top_ctrl, textvariable=self.analysis_file_var, values=sample_files, width=28, state="readonly")
        self.combo_analysis.pack(side=tk.LEFT, padx=5)
        self.combo_analysis.bind("<<ComboboxSelected>>", lambda e: self.run_single_analysis())

        btn_browse = tk.Button(top_ctrl, text="Browse Custom Image...", font=("Segoe UI", 9, "bold"),
                               bg="#238636", fg="white", activebackground="#2ea043", relief=tk.FLAT, padx=12, pady=4,
                               command=self.browse_analysis_image)
        btn_browse.pack(side=tk.LEFT, padx=10)

        btn_run = tk.Button(top_ctrl, text="Run Analysis", font=("Segoe UI", 9, "bold"),
                            bg=self.c_accent, fg="white", activebackground="#388bfd", relief=tk.FLAT, padx=14, pady=4,
                            command=self.run_single_analysis)
        btn_run.pack(side=tk.LEFT, padx=5)

        # Display Panels (Middle)
        panels_frame = tk.Frame(self.tab_analysis, bg=self.c_bg)
        panels_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        panels_frame.columnconfigure(0, weight=1)
        panels_frame.columnconfigure(1, weight=1)
        panels_frame.rowconfigure(0, weight=1)
        panels_frame.rowconfigure(1, weight=1)

        # Panel 1: Localized Boundaries
        self.card_orig = self.create_image_card(panels_frame, "1. Localized Inner Pupil (Green) & Outer Iris (Cyan) Boundary", 0, 0)
        # Panel 2: Unwrapped Strip
        self.card_norm = self.create_image_card(panels_frame, "2. Daugman's Rubber Sheet Normalization (Polar Strip)", 0, 1)
        # Panel 3: Binary Iris Code
        self.card_code = self.create_image_card(panels_frame, "3. 2D Gabor Wavelet Binary Iris Code (Bit Feature Matrix)", 1, 0, columnspan=2)

        # Status Info Bar (Bottom)
        self.lbl_analysis_status = tk.Label(self.tab_analysis, text="Status: Ready", 
                                            font=("Segoe UI", 9), fg=self.c_text_dim, bg=self.c_bg, anchor=tk.W)
        self.lbl_analysis_status.pack(fill=tk.X, padx=15, pady=(0, 5))

        # Initial trigger
        self.after(200, self.run_single_analysis)

    def create_image_card(self, parent, title, row, col, columnspan=1):
        frame = tk.Frame(parent, bg=self.c_card, padx=10, pady=8, 
                         highlightbackground=self.c_card_border, highlightthickness=1)
        frame.grid(row=row, column=col, columnspan=columnspan, sticky="nsew", padx=6, pady=6)

        lbl_title = tk.Label(frame, text=title, font=("Segoe UI", 9, "bold"), fg="#58a6ff", bg=self.c_card)
        lbl_title.pack(anchor=tk.W, pady=(0, 6))

        canvas = tk.Label(frame, bg="#0d1117", text="No image loaded", fg=self.c_text_dim)
        canvas.pack(fill=tk.BOTH, expand=True)

        return {'frame': frame, 'label': canvas}

    def browse_analysis_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Iris Image",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")]
        )
        if file_path:
            self.custom_analysis_path = file_path
            self.analysis_file_var.set(os.path.basename(file_path))
            self.run_single_analysis(override_path=file_path)

    def run_single_analysis(self, override_path=None):
        if override_path:
            path = override_path
        else:
            filename = self.analysis_file_var.get()
            if not filename:
                return
            path = os.path.join(self.samples_dir, filename)

        if not os.path.exists(path):
            messagebox.showerror("Error", f"File does not exist: {path}")
            return

        try:
            gray = load_image(path, as_gray=True)
            loc = localize_iris(gray)
            pupil = loc['pupil']
            iris = loc['iris']

            annotated = draw_boundaries(gray, pupil, iris)
            norm, mask = daugman_rubber_sheet(gray, pupil, iris, radial_res=64, angular_res=256)
            feat = extract_iris_code(norm, mask)
            code = feat['iris_code']

            # Render onto cards
            self.display_on_card(self.card_orig, annotated, max_w=480, max_h=240)
            self.display_on_card(self.card_norm, norm, max_w=480, max_h=240)
            
            # Binary code visualization
            code_vis = (code * 255).astype(np.uint8)
            self.display_on_card(self.card_code, code_vis, max_w=1000, max_h=160)

            self.lbl_analysis_status.config(
                text=f"Image: {os.path.basename(path)} | Pupil: Center=({pupil[0]},{pupil[1]}), Radius={pupil[2]}px | "
                     f"Iris: Center=({iris[0]},{iris[1]}), Radius={iris[2]}px | Code: {code.shape[0]}x{code.shape[1]} bits ({code.size} total)"
            )
        except Exception as e:
            messagebox.showerror("Localization Error", f"An error occurred during analysis: {str(e)}")

    def display_on_card(self, card, cv_img, max_w, max_h):
        if len(cv_img.shape) == 2:
            rgb = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

        h, w = rgb.shape[:2]
        scale = min(max_w / w, max_h / h)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        pil_img = Image.fromarray(resized)
        tk_img = ImageTk.PhotoImage(image=pil_img)

        card['label'].config(image=tk_img, text="")
        card['label'].image = tk_img

    # ==========================================================
    # TAB 2: BIOMETRIC VERIFICATION (1-TO-1 MATCH)
    # ==========================================================
    def build_matching_tab(self):
        sample_files = sorted([os.path.basename(p) for p in glob.glob(os.path.join(self.samples_dir, "*.png"))])

        # Selector Top Frame
        top_ctrl = tk.Frame(self.tab_matching, bg=self.c_card, padx=15, pady=12, 
                            highlightbackground=self.c_card_border, highlightthickness=1)
        top_ctrl.pack(fill=tk.X, padx=10, pady=10)

        # Image 1 (Reference)
        f_img1 = tk.Frame(top_ctrl, bg=self.c_card)
        f_img1.pack(side=tk.LEFT, padx=10)
        tk.Label(f_img1, text="Probe (Image 1):", font=("Segoe UI", 9, "bold"), fg=self.c_text, bg=self.c_card).pack(anchor=tk.W)
        self.match_img1_var = tk.StringVar(value=sample_files[0] if sample_files else "")
        self.combo_m1 = ttk.Combobox(f_img1, textvariable=self.match_img1_var, values=sample_files, width=22, state="readonly")
        self.combo_m1.pack()

        # Image 2 (Query)
        f_img2 = tk.Frame(top_ctrl, bg=self.c_card)
        f_img2.pack(side=tk.LEFT, padx=10)
        tk.Label(f_img2, text="Gallery (Image 2):", font=("Segoe UI", 9, "bold"), fg=self.c_text, bg=self.c_card).pack(anchor=tk.W)
        default_val2 = sample_files[1] if len(sample_files) > 1 else (sample_files[0] if sample_files else "")
        self.match_img2_var = tk.StringVar(value=default_val2)
        self.combo_m2 = ttk.Combobox(f_img2, textvariable=self.match_img2_var, values=sample_files, width=22, state="readonly")
        self.combo_m2.pack()

        # Threshold slider
        f_thresh = tk.Frame(top_ctrl, bg=self.c_card)
        f_thresh.pack(side=tk.LEFT, padx=15)
        self.lbl_thresh = tk.Label(f_thresh, text="Decision Threshold: 0.38", font=("Segoe UI", 9, "bold"), fg=self.c_text, bg=self.c_card)
        self.lbl_thresh.pack(anchor=tk.W)
        self.scale_thresh = tk.Scale(f_thresh, from_=0.20, to=0.50, resolution=0.01, orient=tk.HORIZONTAL, 
                                     bg=self.c_card, fg=self.c_text, highlightthickness=0, length=160,
                                     command=lambda v: self.lbl_thresh.config(text=f"Decision Threshold: {float(v):.2f}"))
        self.scale_thresh.set(0.38)
        self.scale_thresh.pack()

        # Compare Button
        btn_compare = tk.Button(top_ctrl, text="  Compare & Verify  ", font=("Segoe UI", 11, "bold"),
                                bg=self.c_accent, fg="white", activebackground="#388bfd", relief=tk.FLAT, padx=16, pady=6,
                                command=self.run_biometric_matching)
        btn_compare.pack(side=tk.LEFT, padx=15)

        # Results Display Area
        res_area = tk.Frame(self.tab_matching, bg=self.c_bg)
        res_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        res_area.columnconfigure(0, weight=1)
        res_area.columnconfigure(1, weight=1)
        res_area.rowconfigure(0, weight=1)

        # Image 1 display card
        self.card_m1 = self.create_image_card(res_area, "Probe Iris (Image 1)", 0, 0)
        # Image 2 display card
        self.card_m2 = self.create_image_card(res_area, "Gallery Iris (Image 2)", 0, 1)

        # Verdict Badge & Metrics Box (Bottom)
        self.f_verdict = tk.Frame(self.tab_matching, bg=self.c_card, padx=20, pady=12,
                                  highlightbackground=self.c_card_border, highlightthickness=1)
        self.f_verdict.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.lbl_verdict_badge = tk.Label(self.f_verdict, text="PRESS 'COMPARE & VERIFY' TO RUN MATCHING",
                                          font=("Segoe UI", 14, "bold"), fg="#8b949e", bg=self.c_card)
        self.lbl_verdict_badge.pack(anchor=tk.CENTER, pady=(0, 8))

        self.lbl_match_metrics = tk.Label(self.f_verdict, 
                                          text="Hamming Distance: -- | Similarity: -- | Rotational Shift: --",
                                          font=("Segoe UI", 11), fg=self.c_text, bg=self.c_card)
        self.lbl_match_metrics.pack(anchor=tk.CENTER)

    def run_biometric_matching(self):
        f1 = self.match_img1_var.get()
        f2 = self.match_img2_var.get()
        threshold = float(self.scale_thresh.get())

        p1 = os.path.join(self.samples_dir, f1)
        p2 = os.path.join(self.samples_dir, f2)

        try:
            # Process Image 1
            g1 = load_image(p1, as_gray=True)
            loc1 = localize_iris(g1)
            ann1 = draw_boundaries(g1, loc1['pupil'], loc1['iris'])
            n1, m1 = daugman_rubber_sheet(g1, loc1['pupil'], loc1['iris'])
            feat1 = extract_iris_code(n1, m1)

            # Process Image 2
            g2 = load_image(p2, as_gray=True)
            loc2 = localize_iris(g2)
            ann2 = draw_boundaries(g2, loc2['pupil'], loc2['iris'])
            n2, m2 = daugman_rubber_sheet(g2, loc2['pupil'], loc2['iris'])
            feat2 = extract_iris_code(n2, m2)

            # Display images
            self.display_on_card(self.card_m1, ann1, max_w=480, max_h=260)
            self.display_on_card(self.card_m2, ann2, max_w=480, max_h=260)

            # Match
            match_res = match_iris_codes(
                feat1['iris_code'], 
                feat2['iris_code'], 
                feat1['code_mask'], 
                feat2['code_mask'],
                max_shift=8,
                threshold=threshold
            )

            hd = match_res['min_hamming_distance']
            is_match = match_res['is_match']
            sim = match_res['similarity_pct']
            shift = match_res['best_shift']

            if is_match:
                self.lbl_verdict_badge.config(
                    text="✓ BIOMETRIC MATCH: SAMPLES BELONG TO THE SAME SUBJECT",
                    fg="#3fb950"
                )
            else:
                self.lbl_verdict_badge.config(
                    text="✗ NO MATCH: SAMPLES BELONG TO DIFFERENT SUBJECTS",
                    fg="#f85149"
                )

            self.lbl_match_metrics.config(
                text=f"Hamming Distance: {hd:.4f} (Threshold = {threshold:.2f})  |  "
                     f"Biometric Similarity: {sim:.2f}%  |  Rotational Shift Compensation: {shift} cols"
            )
        except Exception as e:
            messagebox.showerror("Matching Error", f"Failed to match iris images: {str(e)}")

    # ==========================================================
    # TAB 3: DATASET BENCHMARK & PERFORMANCE
    # ==========================================================
    def build_benchmark_tab(self):
        top_ctrl = tk.Frame(self.tab_benchmark, bg=self.c_card, padx=15, pady=10, 
                            highlightbackground=self.c_card_border, highlightthickness=1)
        top_ctrl.pack(fill=tk.X, padx=10, pady=10)

        btn_run_bench = tk.Button(top_ctrl, text="  Run Biometric Benchmark (FAR/FRR & ROC)  ", 
                                  font=("Segoe UI", 10, "bold"),
                                  bg=self.c_accent, fg="white", activebackground="#388bfd", relief=tk.FLAT, padx=15, pady=6,
                                  command=self.run_gui_benchmark)
        btn_run_bench.pack(side=tk.LEFT, padx=10)

        self.lbl_bench_status = tk.Label(top_ctrl, text="Click to evaluate dataset and plot error distributions.",
                                         font=("Segoe UI", 9), fg=self.c_text_dim, bg=self.c_card)
        self.lbl_bench_status.pack(side=tk.LEFT, padx=15)

        # Plot Display Card
        plot_frame = tk.Frame(self.tab_benchmark, bg=self.c_card, padx=10, pady=10,
                              highlightbackground=self.c_card_border, highlightthickness=1)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.lbl_plot = tk.Label(plot_frame, bg="#0d1117", text="Benchmark figure will appear here.")
        self.lbl_plot.pack(fill=tk.BOTH, expand=True)

        # Check if benchmark plot already exists
        bench_plot_path = os.path.join(self.outputs_dir, "benchmark_results.png")
        if os.path.exists(bench_plot_path):
            self.load_benchmark_plot(bench_plot_path)

    def load_benchmark_plot(self, path):
        try:
            img = cv2.imread(path)
            if img is not None:
                h, w = img.shape[:2]
                target_w, target_h = 1000, 420
                scale = min(target_w / w, target_h / h)
                resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                tk_img = ImageTk.PhotoImage(image=pil_img)
                self.lbl_plot.config(image=tk_img, text="")
                self.lbl_plot.image = tk_img
        except Exception:
            pass

    def run_gui_benchmark(self):
        from benchmark import run_benchmark
        self.lbl_bench_status.config(text="Computing pairwise matches... Please wait...", fg=self.c_accent)
        self.update_idletasks()

        plot_path = os.path.join(self.outputs_dir, "benchmark_results.png")
        try:
            res = run_benchmark(self.samples_dir, plot_path)
            self.load_benchmark_plot(plot_path)
            self.lbl_bench_status.config(
                text=f"Benchmark Complete! Total pairs: {res['genuine_pairs'] + res['impostor_pairs']} | "
                     f"Decidability d': {res['decidability_d_prime']} | EER: {res['eer_percentage']}% @ Thresh={res['optimal_threshold']}",
                fg=self.c_success
            )
        except Exception as e:
            messagebox.showerror("Benchmark Error", str(e))
            self.lbl_bench_status.config(text="Benchmark failed.", fg=self.c_danger)


if __name__ == "__main__":
    app = IrisRecognitionApp()
    app.mainloop()
