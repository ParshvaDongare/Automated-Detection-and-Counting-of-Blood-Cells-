# 🔬 Automated Detection and Counting of Blood Cells
> **Digital Signal & Image Processing (DSIP) Morphometric Suite**

A state-of-the-art, high-fidelity **Digital Image Processing Desktop Suite** designed to isolate, segment, and quantify blood cell structures from microscopic blood smear slides using advanced **OpenCV Watershed Morphological pipelines**.

---

## 📥 Download Standalone Application (.exe)

You do not need to install Python or source code dependencies to use this application. A fully pre-compiled standalone executable is available for immediate desktop use.

### 👉 **[Download the Executable from GitHub Releases](https://github.com/ParshvaDongare/Automated-Detection-and-Counting-of-Blood-Cells-/releases)**

**Instructions for end-users:**
1. Navigate to the **Releases** section on the right-hand sidebar of this repository.
2. Download the latest **`BloodCellAnalyzerSuite.exe`** file.
3. Double-click to run natively on Windows. All computational libraries (OpenCV, Matplotlib, ReportLab) are completely bundled inside.

---

## 🧠 Core DSIP Techniques Utilized

This software serves as a real-world implementation of complex **Digital Signal & Image Processing (DSIP)** algorithms to resolve real-world clinical microscopy challenges:

### 1. Contrast-Limited Adaptive Histogram Equalization (CLAHE)
* **Purpose:** Standardizes unpredictable slide illuminations and sub-surface backlighting.
* **Mechanism:** Operates on localized image tiles rather than the global histogram, preventing the over-amplification of noise while uncovering subtle cytoplasmic boundaries.

### 2. Edge-Preserving Bilateral Filtering
* **Purpose:** Spatial domain noise suppression.
* **Mechanism:** Replaces pixel values with a weighted average of nearby pixels using both a **spatial Gaussian kernel** and a **range (intensity) Gaussian kernel**. This obliterates background plasma noise without blurring critical cell membranes.

### 3. Chromatic Transformation (BGR to HSV Space)
* **Purpose:** Isolates complex cellular components based on staining properties.
* **Mechanism:** Decouples pure color information (**Hue**) and depth (**Saturation**) from brightness (**Value**). This allows highly robust segmentation of deep purple White Blood Cell (WBC) nuclei regardless of slide brightness.

### 4. Otsu's Automated Global Thresholding
* **Purpose:** Objective binary binarization.
* **Mechanism:** Iteratively searches for the threshold value that minimizes intra-class variance (and maximizes inter-class variance) between foreground cellular components and background serum.

### 5. Structured Morphological Operations
* **Opening (Erosion followed by Dilation):** Eliminates isolated salt-and-pepper pixel artifacts.
* **Closing (Dilation followed by Erosion):** Fills internal cellular vacuole gaps to establish solid connected components.

### 6. Euclidean Distance Transforms & Watershed Segmentation
* **Purpose:** Separating dense, clumped, or overlapping cell clusters.
* **Mechanism:** Calculates the distance of every foreground pixel to the nearest background boundary. Peaks in the distance map form definitive internal markers. The **Watershed algorithm** then treats the intensity profiles as topographical maps, flooding catchments to accurately delineate precise cellular borders.

---

## ✨ Interface & Features Matrix

### 🖥️ 1. Multi-Mode Diagnostic Viewports
* **Detected View:** Real-time overlaid bounding contours, calculated surface areas, and enumerated target indices.
* **Original View:** Raw specimen inspection.
* **🌓 Interactive Split-Screen Wipe Mode:** An elegant visual inspector featuring a fluid, mouse-draggable vertical boundary bar comparing raw input against watershed segmentations in real-time.

### 🛠️ 2. Live Multi-Step Execution Pipeline
Inspect intermediate DSIP mathematical matrix outputs directly inside the workspace tabs:
* **Step 1 Output:** Filtered & CLAHE equalized domain.
* **Step 2 Output:** Nucleus HSV isolated masks.
* **Step 3 Output:** Cleaned binary Red Blood Cell (RBC) masks.
* **Step 4 Output:** Final regional markers feeding the watershed topology.

### 📈 3. Quantitative Analytics Dashboard
* Embedded real-time **Matplotlib canvas visualizers**.
* **Structural Breakdowns:** Interactive dynamic pie charts mapping complete specimen compositions.
* **Morphometric Histograms:** Surface area pixel distribution charts identifying highly skewed macrocytic or microcytic scaling deviations.

### 📄 4. Professional Laboratory PDF Export
* Built-in clinical input modal dialog capturing **Patient Identifiers**, **Referring Practitioners**, and **Custom Comments**.
* Dynamically compiles pristine laboratory documentation via **ReportLab**, embedding visual slice figures, tabular data arrays, and metric summaries.

---

## ⚙️ Source Code Installation (For Developers)

To run or edit the codebase directly using Python:

```bash
# 1. Clone the repository
git clone https://github.com/ParshvaDongare/Automated-Detection-and-Counting-of-Blood-Cells-.git
cd Automated-Detection-and-Counting-of-Blood-Cells-

# 2. Install external library dependencies
pip install opencv-python numpy pillow matplotlib reportlab

# 3. Launch the Premium User Interface
python blood_cell_gui.py
```

---

## 📦 Building the Executable Locally

Developers can independently bundle the full application suite into a high-performance binary:

```bash
python build_executable.py
```
* Automatically verifies PyInstaller hooks.
* Aggressively prunes global machine learning libraries (`torch`, `ultralytics`) to guarantee an incredibly optimized, lightweight binary artifact bundle output inside the `./dist/` folder.

---

## ⚖️ Legal & Medical Disclaimer
This analytical suite is created exclusively for **educational, algorithmic research, and digital image processing demonstrations**. Metrics, pixel area calculations, and structural boundary annotations generated by the automated DSIP pipelines do not constitute approved automated clinical diagnoses or medical advisory protocols.
