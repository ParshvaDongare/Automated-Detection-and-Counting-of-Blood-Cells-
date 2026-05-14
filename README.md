# 🔬 Automated Detection and Counting of Blood Cells

A state-of-the-art, fully polished **Digital Image Processing and Morphometric Analyzer Suite** designed to isolate, detect, and quantify blood cell structures from microscopic slide images using optimized **OpenCV Watershed segmentation algorithms**.

---

## ✨ Key Features

### 🖥️ 1. Interactive Diagnostic Viewport
* **Multiple Display Modes:** Seamlessly toggle between **Detected View** (annotated bounding contours and cell indices), **Original Raw View**, and an elegant **🌓 Split Comparison Wipe Mode** with live draggable vertical boundary sliders.
* **Hardware Canvas Transformations:** Full mouse integration supporting scroll-wheel zooming, fluid coordinate panning, and sub-millisecond layer compositing.

### 🛠️ 2. Multi-Step Execution Pipeline Matrix
Inspect intermediate computational image stages directly within the tabbed interface:
* **Step 1:** Contrast-Limited Adaptive Histogram Equalization (**CLAHE**) & Bilateral Smoothing.
* **Step 2:** Specialized **HSV Nuclei Thresholding** to separate chromatically dense regions.
* **Step 3:** Morphological cleaning to isolate crisp Red Blood Cell (**RBC**) binary masks.
* **Step 4:** Distance transform markers leading to the final **Watershed boundaries**.

### 📈 3. Real-Time Analytics & Proportions
* Embedded **Matplotlib canvas visualizers** rendering live structural breakdowns.
* **Class Proportions:** Dynamic pie charts showing structural composition.
* **Surface Area Histograms:** Red Blood Cell pixel area distribution mapping to identify extreme scaling deviations.

### 📄 4. High-Fidelity PDF Clinical Export
* Built-in metadata dialog modal to input custom **Patient Identifiers**, **Referring Clinical Officers**, and **Diagnostic Notes**.
* Dynamically compiles a beautiful, multi-page document via **ReportLab** embedding segmented previews, proportional pie figures, and professional data-grid breakdowns.

### 🛡️ 5. Extreme System Stability
* Features a custom, central **timer garbage collection engine** that tracks and preemptively terminates active Tkinter loop events upon exit.
* Zero traceback leakages during multi-threaded background processing or application termination.

---

## ⚙️ Installation & Requirements

Ensure you have Python 3.9+ installed along with the requisite dependencies:

```bash
pip install opencv-python numpy pillow matplotlib reportlab
```

---

## ▶️ Usage

Simply run the central graphical user interface script:

```bash
python blood_cell_gui.py
```

1. **Load Image:** Click the **📂 Load Image** control or use `Ctrl + O` to select a microscopic blood specimen field.
2. **Analyze:** Click **▶ Analyze** or press `Enter` to execute the automated watershed verification pipeline.
3. **Explore Workspace:** Navigate across the three main diagnostic workspaces (**Detection**, **Tuning Pipeline**, and **Analytics**).
4. **Export Report:** Click **💾 Export PDF Report** or press `Ctrl + S` to input clinical data and save professional PDF deliverables.

---

## ⚖️ Disclaimer
This software suite is intended strictly for **educational, demonstration, and digital image processing research applications**. Bounding contours and morphological statistical deliverables do not constitute definitive or automated clinical medical diagnoses.
