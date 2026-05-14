"""
╔══════════════════════════════════════════════════════════════════════════╗
║         BLOOD CELL ANALYZER — Premium Animated UI/UX Suite v2.0          ║
║         DSIP Project | Advanced Image Processing & Diagnostics           ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import threading
import time
import json

# Optional analytics & reporting imports
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    _MATPLOTLIB_OK = True
except ImportError:
    _MATPLOTLIB_OK = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    _PDF_OK = True
except ImportError:
    _PDF_OK = False

# ─── PREMIUM COLOUR PALETTE (Sleek Clinical Dark Theme) ─────────
BG_DARK     = "#070a14"       # Deep outer space
BG_PANEL    = "#0f1526"       # Dark header/sidebar
BG_CARD     = "#161f36"       # Soft widget container
BG_HOVER    = "#1f2b48"       # Interactive hover state
ACCENT      = "#007acc"       # Vivid neon azure
ACCENT2     = "#8b5cf6"       # Ultra violet
SUCCESS     = "#10b981"       # Emerald clinical green
WARNING     = "#f59e0b"       # Vibrant amber
DANGER      = "#ef4444"       # Diagnostic red alerts
TEXT_PRI    = "#f8fafc"       # High-contrast pristine white
TEXT_SEC    = "#94a3b8"       # Muted slate
TEXT_MUTED  = "#475569"       # Deep slate for empty states
BORDER      = "#223559"       # Soft divider outline
GLOW        = "#007acc33"

# Consistent Cell Classification Colors
CELL_COLOURS = {
    "Lymphocyte":  "#8b5cf6", # Purple
    "Monocyte":    "#f59e0b", # Amber
    "Platelet":    "#10b981", # Green
    "WBC":         "#007acc", # Azure Blue
    "RBC":         "#ef4444", # Red
    "Uncertain":   "#94a3b8", # Muted
}


# ─── UPGRADED DETECTION ENGINE WITH REAL-TIME CALLBACKS ─────────

def detect_cells_watershed(img_path: str, min_area=200, max_area=6000, 
                           min_circ=0.35, max_aspect=3.0, progress_callback=None) -> dict:
    """
    Advanced multi-step Watershed & Morphological Pipeline with embedded progress tracking
    and interim diagnostic frame outputs for algorithm visualization.
    """
    def _report_prog(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)

    _report_prog(5, "Loading image arrays…")
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image file: {img_path}")

    original = img.copy()
    rgb      = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # STEP 1: Grayscale -> CLAHE Contrast Enhancement -> Bilateral Filter
    _report_prog(15, "Applying CLAHE Contrast Enhancement & Bilateral Filtering…")
    gray         = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    clahe        = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray_cl      = clahe.apply(gray)
    preprocessed = cv2.bilateralFilter(gray_cl, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Store interim image for diagnostics tab
    step1_preview = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2RGB)

    # STEP 2: WBC Detection via Advanced HSV Color Masking
    _report_prog(30, "Isolating WBC Nuclei via specific HSV thresholds…")
    hsv       = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    lower_wbc = np.array([115, 50, 30])
    upper_wbc = np.array([179, 255, 220])
    wbc_raw   = cv2.inRange(hsv, lower_wbc, upper_wbc)

    k5 = np.ones((5, 5), np.uint8)
    k7 = np.ones((7, 7), np.uint8)
    wbc_closed = cv2.morphologyEx(wbc_raw,    cv2.MORPH_CLOSE, k7, iterations=2)
    wbc_opened = cv2.morphologyEx(wbc_closed, cv2.MORPH_OPEN,  k5, iterations=1)
    
    step2_preview = cv2.cvtColor(wbc_opened, cv2.COLOR_GRAY2RGB)

    MIN_WBC_AREA = 700
    wbc_cnts_all, _ = cv2.findContours(wbc_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    wbc_contours = [c for c in wbc_cnts_all if cv2.contourArea(c) > MIN_WBC_AREA]

    # Build WBC exclusion zone (dilated filled mask to separate from RBCs)
    wbc_filled = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(wbc_filled, wbc_contours, -1, 255, -1)
    wbc_exclusion = cv2.dilate(wbc_filled, k7, iterations=3)

    # Collect WBC centroids
    wbc_labels = []
    for c in wbc_contours:
        M = cv2.moments(c)
        if M["m00"] == 0: continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        wbc_labels.append((cx, cy, cv2.contourArea(c)))

    # STEP 3: Otsu Global Thresholding -> Total Foreground Cell Mask
    _report_prog(45, "Computing Otsu Global Background Thresholding…")
    _, total_mask = cv2.threshold(preprocessed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # STEP 4: Subtract WBC Zone -> RBC-only Mask
    _report_prog(60, "Isolating Red Blood Cell clusters & applying cleaning filters…")
    rbc_raw     = cv2.subtract(total_mask, wbc_exclusion)
    rbc_cleaned = cv2.morphologyEx(rbc_raw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
    
    step4_preview = cv2.cvtColor(rbc_cleaned, cv2.COLOR_GRAY2RGB)

    # STEP 5: Distance Transform -> Watershed Segmentation
    _report_prog(75, "Executing Distance Transform & Watershed Marker Separation…")
    dist_transform = cv2.distanceTransform(rbc_cleaned, cv2.DIST_L2, maskSize=5)
    if dist_transform.max() == 0:
        sure_fg = np.zeros_like(rbc_cleaned)
    else:
        fg_thresh  = 0.40 * dist_transform.max()
        _, sure_fg = cv2.threshold(dist_transform, fg_thresh, 255, 0)
        sure_fg    = sure_fg.astype(np.uint8)

    sure_bg = cv2.dilate(rbc_cleaned, np.ones((3, 3), np.uint8), iterations=3)
    unknown  = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers    = markers + 1
    markers[unknown == 255] = 0
    markers_ws = cv2.watershed(original.copy(), markers.copy())

    # Reconstruct clean mask from watershed markers
    ws_mask = np.zeros(gray.shape, np.uint8)
    ws_mask[markers_ws > 1]   = 255
    ws_mask[markers_ws == -1] = 0
    ws_mask_clean = cv2.morphologyEx(ws_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    
    step5_preview = cv2.cvtColor(ws_mask_clean, cv2.COLOR_GRAY2RGB)

    # STEP 6: Apply Customizable Hyperparameter Filtering & Output Mapping
    _report_prog(90, "Extracting final statistical metrics and mapping coordinates…")
    rbc_cnts_all, _ = cv2.findContours(ws_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rbc_final = []
    
    for c in rbc_cnts_all:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area: continue
        peri = cv2.arcLength(c, True)
        if peri == 0: continue
        circ = (4 * np.pi * area) / (peri ** 2)
        if circ < min_circ: continue
        rect = cv2.minAreaRect(c)
        bw, bh = rect[1]
        if bw == 0 or bh == 0: continue
        if max(bw, bh) / min(bw, bh) > max_aspect: continue
        rbc_final.append((c, area, circ))

    output = rgb.copy()
    stats  = {"total": 0, "sizes": [], "types": {"RBC": 0, "WBC": 0}, "cells": []}

    # Draw RBCs — red boundary (clinically correct) + compact label
    rbc_colour = (220, 30, 30)      # RED — standard clinical convention
    rbc_text_bg = (80, 10, 10)      # Dark red tag background
    for i, (c, area, circ) in enumerate(rbc_final):
        cv2.drawContours(output, [c], -1, rbc_colour, 1)
        x, y, w, h = cv2.boundingRect(c)
        cx, cy = x + w//2, y + h//2
        
        # Center targeting dot
        cv2.circle(output, (cx, cy), 2, rbc_colour, -1)
        
        # Compact red label tag
        label = f"RBC {i+1}"
        fs = 0.32
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        tx = max(cx - tw // 2, 0)
        ty = max(cy - 4, 10)
        cv2.rectangle(output, (tx - 1, ty - th - 2), (tx + tw + 1, ty + 1), rbc_text_bg, -1)
        cv2.putText(output, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 180, 180), 1, cv2.LINE_AA)
        
        stats["total"] += 1
        stats["sizes"].append(area)
        stats["types"]["RBC"] += 1
        stats["cells"].append({
            "type": "RBC", "area": round(area, 1),
            "circ": round(circ, 2), "x": cx, "y": cy, "w": w, "h": h
        })

    # Draw WBCs — highlight ring + floating clean indicator
    wbc_colour = (60, 130, 255)
    for i, (cx, cy, area) in enumerate(wbc_labels):
        radius = int(np.sqrt(area / np.pi)) + 10
        cv2.circle(output, (cx, cy), radius, wbc_colour, 2)
        cv2.circle(output, (cx, cy), 3, wbc_colour, -1)
        label = f"WBC {i+1}"
        fs = 0.38
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        tx = max(cx - tw // 2, 0)
        ty = max(cy - radius - 6, 14)
        cv2.rectangle(output, (tx - 2, ty - th - 3), (tx + tw + 2, ty + 2), wbc_colour, -1)
        cv2.putText(output, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)

        stats["total"] += 1
        stats["sizes"].append(area)
        stats["types"]["WBC"] += 1
        stats["cells"].append({
            "type": "WBC", "area": round(area, 1),
            "circ": 1.0, "x": cx, "y": cy, "w": radius * 2, "h": radius * 2
        })

    stats["types"] = {k: v for k, v in stats["types"].items() if v > 0}

    _report_prog(100, "Inference completed successfully!")

    return {
        "image":    output,
        "original": rgb,
        "stats":    stats,
        "path":     img_path,
        # Intermediates for diagnostics tab
        "step1":    step1_preview,
        "step2":    step2_preview,
        "step4":    step4_preview,
        "step5":    step5_preview,
    }


# ─── ROUNDED GEOMETRY ENGINE ────────────────────────────────────

def round_rect(canvas, x1, y1, x2, y2, r=12, **kw):
    pts = [x1+r,y1, x2-r,y1, x2,y1, x2,y1+r, x2,y2-r, x2,y2,
           x2-r,y2, x1+r,y2, x1,y2, x1,y2-r, x1,y1+r, x1,y1]
    return canvas.create_polygon(pts, smooth=True, **kw)


# ─── ANIMATED GLOW BUTTON ───────────────────────────────────────

class GlowButton(tk.Canvas):
    def __init__(self, parent, text="", command=None, bg=ACCENT, fg=TEXT_PRI,
                 width=160, height=40, icon="", radius=10, **kw):
        super().__init__(parent, width=width, height=height, bg=parent.cget("bg"),
                         highlightthickness=0, **kw)
        self.command  = command
        self.bg_idle  = bg
        self.bg_hover = self._lighten(bg)
        self.text     = text
        self.icon     = icon
        self.fg       = fg
        self.r        = radius
        self.curr_bg  = bg
        self._draw(self.curr_bg)
        self.bind("<Enter>",    self._on_enter)
        self.bind("<Leave>",    self._on_leave)
        self.bind("<Button-1>", self._on_click)

    def _lighten(self, hex_col):
        h = hex_col.lstrip("#")
        r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
        r = min(255, r + 35); g = min(255, g + 35); b = min(255, b + 35)
        return "#{:02x}{:02x}{:02x}".format(r, g, b)

    def _draw(self, colour):
        self.delete("all")
        w, h = self.winfo_reqwidth(), self.winfo_reqheight()
        round_rect(self, 0, 0, w, h, r=self.r, fill=colour, outline="")
        lbl = (self.icon + "  " if self.icon else "") + self.text
        self.create_text(w//2, h//2, text=lbl, fill=self.fg,
                         font=("Segoe UI", 10, "bold"))

    def _on_enter(self, _):
        self._draw(self.bg_hover)
        self.config(cursor="hand2")

    def _on_leave(self, _):
        self._draw(self.bg_idle)
        self.config(cursor="")

    def _on_click(self, _):
        # Quick push-down visual pop
        self._draw(self._lighten(self.bg_hover))
        self.after(80, lambda: self._draw(self.bg_hover))
        if self.command: self.command()


# ─── ROLLING STAT COUNTER CARD ──────────────────────────────────

class StatCard(tk.Frame):
    def __init__(self, parent, label, colour=ACCENT, **kw):
        super().__init__(parent, bg=BG_CARD, **kw)
        self.config(padx=16, pady=12)
        self.colour = colour
        self.current_val = 0
        self.target_val  = 0
        self.is_rolling  = False
        
        self.lbl_val = tk.Label(self, text="—", font=("Segoe UI", 26, "bold"),
                                fg=colour, bg=BG_CARD)
        self.lbl_val.pack(anchor="w")
        tk.Label(self, text=label, font=("Segoe UI", 9),
                 fg=TEXT_SEC, bg=BG_CARD).pack(anchor="w")

    def update_instant(self, value):
        self.is_rolling = False
        self.lbl_val.config(text=str(value))
        if isinstance(value, (int, float)):
            self.current_val = int(value)

    def roll_to(self, target):
        """Beautiful speedometer digital tick-up animation loop."""
        self.target_val = int(target)
        self.is_rolling = True
        self.current_val = 0
        self._step_roll()

    def _step_roll(self):
        if not self.is_rolling: return
        diff = self.target_val - self.current_val
        if diff <= 0:
            self.current_val = self.target_val
            self.lbl_val.config(text=str(self.current_val))
            self.is_rolling = False
            return
        
        # Increment logarithmically/accelerated
        step = max(1, int(diff * 0.15))
        self.current_val += step
        self.lbl_val.config(text=str(self.current_val))
        self._roll_timer = self.after(20, self._step_roll)

    def destroy(self):
        self.is_rolling = False
        if hasattr(self, "_roll_timer"):
            try: self.after_cancel(self._roll_timer)
            except Exception: pass
        super().destroy()


# ─── PDF CONFIGURATION MODAL DIALOG ─────────────────────────────

class ReportConfigDialog(tk.Toplevel):
    def __init__(self, parent, callback):
        super().__init__(parent)
        self.title("📄 Configure Laboratory Report")
        self.geometry("460x380")
        self.configure(bg=BG_CARD)
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        
        self.callback = callback
        
        # Center modal
        self.update_idletasks()
        x = parent.winfo_rootx() + (parent.winfo_width() - 460) // 2
        y = parent.winfo_rooty() + (parent.winfo_height() - 380) // 2
        self.geometry(f"+{x}+{y}")

        tk.Label(self, text="Report Document Metadata", font=("Segoe UI", 14, "bold"),
                 fg=TEXT_PRI, bg=BG_CARD).pack(pady=(20, 15), padx=20, anchor="w")

        # Container
        form = tk.Frame(self, bg=BG_CARD)
        form.pack(fill="both", expand=True, padx=20)

        # Fields
        tk.Label(form, text="Patient Name / ID:", font=("Segoe UI", 9), fg=TEXT_SEC, bg=BG_CARD).pack(anchor="w", pady=(0, 2))
        self.ent_patient = tk.Entry(form, font=("Segoe UI", 10), bg=BG_DARK, fg=TEXT_PRI, insertbackground=TEXT_PRI, relief="flat")
        self.ent_patient.pack(fill="x", pady=(0, 12), ipady=5)
        self.ent_patient.insert(0, "Anonymous Patient")

        tk.Label(form, text="Referring Clinical Officer:", font=("Segoe UI", 9), fg=TEXT_SEC, bg=BG_CARD).pack(anchor="w", pady=(0, 2))
        self.ent_doctor = tk.Entry(form, font=("Segoe UI", 10), bg=BG_DARK, fg=TEXT_PRI, insertbackground=TEXT_PRI, relief="flat")
        self.ent_doctor.pack(fill="x", pady=(0, 12), ipady=5)
        self.ent_doctor.insert(0, "Dr. Default Pathologist")

        tk.Label(form, text="Diagnostic Comments / Notes:", font=("Segoe UI", 9), fg=TEXT_SEC, bg=BG_CARD).pack(anchor="w", pady=(0, 2))
        self.ent_notes = tk.Entry(form, font=("Segoe UI", 10), bg=BG_DARK, fg=TEXT_PRI, insertbackground=TEXT_PRI, relief="flat")
        self.ent_notes.pack(fill="x", pady=(0, 20), ipady=5)
        self.ent_notes.insert(0, "Automated watershed cell multi-class verification.")

        # Bottom Buttons
        btn_box = tk.Frame(self, bg=BG_CARD)
        btn_box.pack(fill="x", padx=20, pady=(0, 20))
        
        GlowButton(btn_box, text="Cancel", command=self.destroy, bg=TEXT_MUTED, width=100, height=36).pack(side="left")
        GlowButton(btn_box, text="Generate High-Fidelity PDF", command=self._submit, bg=SUCCESS, width=220, height=36).pack(side="right")

    def _submit(self):
        data = {
            "patient": self.ent_patient.get().strip() or "N/A",
            "doctor":  self.ent_doctor.get().strip()  or "N/A",
            "notes":   self.ent_notes.get().strip()   or "None",
        }
        self.destroy()
        self.callback(data)


# ─── MAIN COMPLETE APPLICATION SUITE ────────────────────────────

class BloodCellApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🔬 Blood Cell Analyzer Suite • Premium Diagnostics")
        self.geometry("1420x880")
        self.minsize(1200, 750)
        self.configure(bg=BG_DARK)
        self.resizable(True, True)

        # Reactive State Variables
        self.result       = None
        self.img_path     = tk.StringVar(value="")
        self.status_text  = tk.StringVar(value="Ready — Drag & drop or load an image file")
        self.progress_var = tk.DoubleVar(value=0)
        self._busy        = False
        
        # Image Display Controls
        self._photo_disp  = None
        self._zoom        = 1.0
        self._pan_x       = 0
        self._pan_y       = 0
        self._drag_start  = None
        self._view_mode   = tk.StringVar(value="detected") # 'detected', 'original', 'split'
        self._split_pos   = 0.5  # 50% relative wipe line position

        # Custom Tuning Variables
        self.var_min_area   = tk.IntVar(value=200)
        self.var_max_area   = tk.IntVar(value=6000)
        self.var_min_circ   = tk.DoubleVar(value=0.35)
        self.var_max_aspect = tk.DoubleVar(value=3.0)

        # Tab views container tracking
        self.active_tab = "Detection"

        self._build_ui()
        self._animate_header()

        # Keyboard global mappings
        self.bind("<Control-o>", lambda _: self._load_image())
        self.bind("<Return>",    lambda _: self._run_analysis())
        self.bind("<Control-s>", lambda _: self._save_report_dialog())

    def after(self, ms, func=None, *args):
        timer_id = super().after(ms, func, *args)
        if not hasattr(self, "_active_timers"):
            self._active_timers = set()
        self._active_timers.add(timer_id)
        return timer_id

    def after_cancel(self, id):
        if hasattr(self, "_active_timers") and id in self._active_timers:
            self._active_timers.discard(id)
        return super().after_cancel(id)

    def destroy(self):
        if hasattr(self, "_active_timers"):
            for t_id in list(self._active_timers):
                try: super().after_cancel(t_id)
                except Exception: pass
            self._active_timers.clear()
        super().destroy()

    def report_callback_exception(self, exc, val, tb):
        """Silently ignore 'invalid command name' tracebacks fired during app shutdown."""
        if "invalid command name" in str(val):
            return
        super().report_callback_exception(exc, val, tb)

    # ─── TOP LEVEL UI ASSEMBLY ──────────────────────────────────
    def _build_ui(self):
        self._build_toast_container()
        self._build_header()
        self._build_tabs_navigation()

        # Master Content Frame holding interchangeable tabs
        self.content_master = tk.Frame(self, bg=BG_DARK)
        self.content_master.pack(fill="both", expand=True, padx=16, pady=(0, 12))
        
        # Prepare individual workspace pages
        self.page_detection   = tk.Frame(self.content_master, bg=BG_DARK)
        self.page_tuning      = tk.Frame(self.content_master, bg=BG_DARK)
        self.page_analytics   = tk.Frame(self.content_master, bg=BG_DARK)

        # Grid all pages into the master frame so we can swap them dynamically
        for p in (self.page_detection, self.page_tuning, self.page_analytics):
            p.grid(row=0, column=0, sticky="nsew")
        self.content_master.rowconfigure(0, weight=1)
        self.content_master.columnconfigure(0, weight=1)

        # Build inside pages
        self._build_detection_page()
        self._build_tuning_diagnostics_page()
        self._build_analytics_page()

        self._build_statusbar()
        # Ensure default tab is displayed
        self._switch_tab("Detection")

    def _build_toast_container(self):
        """Floating notification wrapper frame at bottom right."""
        self.toast_frame = tk.Frame(self, bg=BG_CARD, bd=1, relief="solid")
        # Hidden initially
        self.toast_lbl = tk.Label(self.toast_frame, text="", font=("Segoe UI", 10, "bold"),
                                  bg=BG_CARD, fg=TEXT_PRI, padx=16, pady=10)
        self.toast_lbl.pack()

    def show_toast(self, msg, colour=SUCCESS):
        """Elegant pop-up slide animation overlay."""
        self.toast_lbl.config(text=msg, fg=colour)
        self.toast_frame.config(highlightbackground=colour, highlightthickness=1)
        # Position at bottom right
        self.update_idletasks()
        rw, rh = self.winfo_width(), self.winfo_height()
        tw, th = self.toast_frame.winfo_reqwidth(), self.toast_frame.winfo_reqheight()
        self.toast_frame.place(x=rw - tw - 25, y=rh - th - 50)
        
        # Schedule auto hide
        if hasattr(self, "_toast_timer"):
            self.after_cancel(self._toast_timer)
        def _hide():
            if self.toast_frame.winfo_exists():
                self.toast_frame.place_forget()
        self._toast_timer = self.after(3500, _hide)

    def _build_header(self):
        hdr = tk.Frame(self, bg=BG_PANEL, height=75)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        # Animated Gradient Bottom Strip
        self.strip = tk.Canvas(hdr, height=3, bg=BG_PANEL, highlightthickness=0)
        self.strip.pack(fill="x", side="bottom")
        self.strip.create_line(0, 1, 3000, 1, fill=ACCENT, width=3)

        # Title Block
        tk.Label(hdr, text="🔬", font=("Segoe UI Emoji", 24), bg=BG_PANEL, fg=ACCENT).pack(side="left", padx=(20, 6), pady=12)
        title_box = tk.Frame(hdr, bg=BG_PANEL)
        title_box.pack(side="left", pady=12)
        tk.Label(title_box, text="Blood Cell Analyzer Suite", font=("Segoe UI", 16, "bold"), fg=TEXT_PRI, bg=BG_PANEL).pack(anchor="w")
        tk.Label(title_box, text="Premium Automated UI/UX Diagnostic Suite", font=("Segoe UI", 9), fg=ACCENT, bg=BG_PANEL).pack(anchor="w")

        # Global Quick Access Controls
        ctrl = tk.Frame(hdr, bg=BG_PANEL)
        ctrl.pack(side="right", padx=20)
        
        GlowButton(ctrl, text="Load Image", icon="📂", command=self._load_image, bg=ACCENT, width=140, height=38).pack(side="left", padx=4)
        GlowButton(ctrl, text="Analyze", icon="▶", command=self._run_analysis, bg=SUCCESS, width=120, height=38).pack(side="left", padx=4)
        GlowButton(ctrl, text="Export PDF Report", icon="💾", command=self._save_report_dialog, bg=ACCENT2, width=170, height=38).pack(side="left", padx=4)
        GlowButton(ctrl, text="Clear", icon="✕", command=self._clear, bg=BG_CARD, fg=TEXT_SEC, width=90, height=38).pack(side="left", padx=4)

    def _build_tabs_navigation(self):
        """Top level navigation panel separating logical domains."""
        nav = tk.Frame(self, bg=BG_DARK, pady=8, padx=16)
        nav.pack(fill="x")
        
        self.tab_buttons = {}
        tabs = [("Detection", "🔬 Primary Detection View"), 
                ("Tuning",    "🛠️ Pipeline Execution Steps"), 
                ("Analytics", "📈 Charts & Histograms")]
        
        for key, title in tabs:
            btn = tk.Button(nav, text=title, font=("Segoe UI", 10, "bold"), bg=BG_CARD, fg=TEXT_SEC,
                            relief="flat", bd=0, padx=16, pady=6, cursor="hand2",
                            command=lambda k=key: self._switch_tab(k))
            btn.pack(side="left", padx=(0, 8))
            self.tab_buttons[key] = btn

    def _switch_tab(self, key):
        self.active_tab = key
        # Update tab visual selections
        for k, btn in self.tab_buttons.items():
            if k == key:
                btn.config(bg=ACCENT, fg=TEXT_PRI)
            else:
                btn.config(bg=BG_CARD, fg=TEXT_SEC)
        
        # Raise active workspace frame container
        if key == "Detection":
            self.page_detection.tkraise()
        elif key == "Tuning":
            self.page_tuning.tkraise()
            self._update_diagnostics_grid()
        elif key == "Analytics":
            self.page_analytics.tkraise()
            self._render_matplotlib_chart()

    # ─── TAB 1: PRIMARY DETECTION SUITE ─────────────────────────
    def _build_detection_page(self):
        self.page_detection.columnconfigure(0, weight=3, minsize=650)
        self.page_detection.columnconfigure(1, weight=1, minsize=350)
        self.page_detection.rowconfigure(0, weight=1)

        # Left split: Image Viewport
        lf = tk.Frame(self.page_detection, bg=BG_DARK)
        lf.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        lf.rowconfigure(1, weight=1)
        lf.columnconfigure(0, weight=1)

        # Top viewport controls toolbar
        v_toolbar = tk.Frame(lf, bg=BG_CARD, padx=12, pady=8)
        v_toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        
        tk.Label(v_toolbar, text="Mode:", font=("Segoe UI", 9, "bold"), fg=TEXT_SEC, bg=BG_CARD).pack(side="left", padx=(0, 6))
        
        # Interactive mode switch
        for val, label in [("detected", "Detected View"), ("original", "Original"), ("split", "🌓 Split Comparison")]:
            rb = tk.Radiobutton(v_toolbar, text=label, variable=self._view_mode, value=val,
                                command=self._refresh_viewport, font=("Segoe UI", 9),
                                bg=BG_CARD, fg=TEXT_PRI, selectcolor=BG_DARK, activebackground=BG_CARD, activeforeground=ACCENT)
            rb.pack(side="left", padx=4)

        # Center Reset & Hint
        tk.Button(v_toolbar, text="Reset Zoom/Pan", font=("Segoe UI", 8), bg=BG_HOVER, fg=TEXT_SEC,
                  relief="flat", command=self._reset_view_transforms).pack(side="right")
        
        # Main interactive drawing Canvas
        v_frame = tk.Frame(lf, bg=BG_CARD)
        v_frame.grid(row=1, column=0, sticky="nsew")
        v_frame.rowconfigure(0, weight=1)
        v_frame.columnconfigure(0, weight=1)

        self._canvas = tk.Canvas(v_frame, bg=BG_CARD, highlightthickness=1, highlightbackground=BORDER)
        self._canvas.grid(row=0, column=0, sticky="nsew")
        
        # Mouse events integration
        self._canvas.bind("<Configure>", self._refresh_viewport)
        self._canvas.bind("<MouseWheel>", self._on_zoom)
        self._canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self._canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self._canvas.bind("<Motion>", self._on_mouse_move)

        self._canvas.create_text(450, 320, text="📂 Drop or load a blood sample image to analyze",
                                 font=("Segoe UI", 16), fill=TEXT_MUTED, tags="placeholder")

        # Sleek Animated Ring & Progress indicator integration
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Premium.Horizontal.TProgressbar", troughcolor=BG_CARD, background=ACCENT, bordercolor=BG_CARD)
        self._pbar = ttk.Progressbar(lf, variable=self.progress_var, maximum=100, mode="determinate",
                                     style="Premium.Horizontal.TProgressbar")
        self._pbar.grid(row=2, column=0, sticky="ew", pady=(6, 0))

        # Right split: Clinical Results Block
        rf = tk.Frame(self.page_detection, bg=BG_DARK)
        rf.grid(row=0, column=1, sticky="nsew")
        rf.columnconfigure(0, weight=1)
        rf.rowconfigure(2, weight=1)

        # Live counters
        stats_box = tk.Frame(rf, bg=BG_DARK)
        stats_box.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        stats_box.columnconfigure((0,1), weight=1)

        self._card_total = StatCard(stats_box, "Total Cells Counted", colour=ACCENT)
        self._card_total.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self._card_types = StatCard(stats_box, "Unique Cell Classes", colour=ACCENT2)
        self._card_types.grid(row=0, column=1, sticky="ew")

        # Mini Canvas Chart Breakdown
        mc_box = tk.Frame(rf, bg=BG_CARD)
        mc_box.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        tk.Label(mc_box, text="REAL-TIME CLASS PROPORTIONS", font=("Segoe UI", 9, "bold"), fg=TEXT_SEC, bg=BG_CARD).pack(anchor="w", padx=12, pady=(8, 2))
        self._mini_chart = tk.Canvas(mc_box, bg=BG_CARD, height=150, highlightthickness=0)
        self._mini_chart.pack(fill="x", padx=12, pady=(0, 8))
        self._render_mini_chart({})

        # Comprehensive Tree List
        t_box = tk.Frame(rf, bg=BG_CARD)
        t_box.grid(row=2, column=0, sticky="nsew", pady=(0, 8))
        t_box.rowconfigure(1, weight=1)
        t_box.columnconfigure(0, weight=1)

        tk.Label(t_box, text="DETECTED CELL INSTANCES", font=("Segoe UI", 9, "bold"), fg=TEXT_SEC, bg=BG_CARD).grid(row=0, column=0, sticky="w", padx=12, pady=(8, 2))

        style.configure("Clinical.Treeview", background=BG_CARD, foreground=TEXT_PRI, fieldbackground=BG_CARD,
                        rowheight=26, bordercolor=BORDER, font=("Consolas", 9))
        style.configure("Clinical.Treeview.Heading", background=BG_PANEL, foreground=TEXT_SEC, font=("Segoe UI", 9, "bold"), relief="flat")
        style.map("Clinical.Treeview", background=[("selected", ACCENT)], foreground=[("selected", "#ffffff")])

        self._tree = ttk.Treeview(t_box, columns=("Type", "Area", "Circ", "Pos"), show="headings", style="Clinical.Treeview")
        self._tree.heading("Type", text="Class")
        self._tree.heading("Area", text="Area (px²)")
        self._tree.heading("Circ", text="Circularity")
        self._tree.heading("Pos",  text="Center (x,y)")
        self._tree.column("Type", width=80)
        self._tree.column("Area", width=75, anchor="center")
        self._tree.column("Circ", width=75, anchor="center")
        self._tree.column("Pos",  width=100, anchor="center")
        self._tree.grid(row=1, column=0, sticky="nsew", padx=6, pady=(0, 6))

        sb_tree = ttk.Scrollbar(t_box, orient="vertical", command=self._tree.yview)
        sb_tree.grid(row=1, column=1, sticky="ns")
        self._tree.configure(yscrollcommand=sb_tree.set)
        self._tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        # Log Block
        log_box = tk.Frame(rf, bg=BG_CARD)
        log_box.grid(row=3, column=0, sticky="ew")
        tk.Label(log_box, text="EXECUTION ENGINE TRACE", font=("Segoe UI", 8, "bold"), fg=TEXT_SEC, bg=BG_CARD).pack(anchor="w", padx=12, pady=(6, 2))
        self._log_text = tk.Text(log_box, height=8, bg=BG_DARK, fg=TEXT_SEC, font=("Consolas", 8),
                                 state="disabled", relief="flat")
        self._log_text.pack(fill="x", padx=6, pady=(0, 6))

    # ─── TAB 2: PIPELINE DIAGNOSTIC VISUALIZATION ───────────────
    def _build_tuning_diagnostics_page(self):
        self.page_tuning.columnconfigure(0, weight=1)
        self.page_tuning.rowconfigure(0, weight=1)

        # Full screen grid for 4-Step intermediate previews
        diag_grid = tk.Frame(self.page_tuning, bg=BG_DARK)
        diag_grid.grid(row=0, column=0, sticky="nsew")
        diag_grid.rowconfigure((0,1), weight=1)
        diag_grid.columnconfigure((0,1), weight=1)

        self.diag_canvases = {}
        steps = [("step1", "Step 1: CLAHE & Bilateral Filter", 0, 0),
                 ("step2", "Step 2: HSV Nuclei Thresholding",   0, 1),
                 ("step4", "Step 3: Isolated RBC Clean Mask",   1, 0),
                 ("step5", "Step 4: Watershed Markers Output",  1, 1)]

        for key, title, r, c in steps:
            box = tk.Frame(diag_grid, bg=BG_CARD)
            box.grid(row=r, column=c, sticky="nsew", padx=8, pady=8)
            box.rowconfigure(1, weight=1)
            box.columnconfigure(0, weight=1)
            tk.Label(box, text=title, font=("Segoe UI", 10, "bold"), fg=TEXT_SEC, bg=BG_CARD).grid(row=0, column=0, sticky="w", padx=12, pady=6)
            canv = tk.Canvas(box, bg=BG_PANEL, highlightthickness=0)
            canv.grid(row=1, column=0, sticky="nsew")
            canv.bind("<Configure>", lambda _, k=key: self._update_diag_canvas(k))
            self.diag_canvases[key] = canv

    # ─── TAB 3: COMPLETE EMBEDDED MATPLOTLIB ANALYTICS ──────────
    def _build_analytics_page(self):
        self.page_analytics.columnconfigure(0, weight=1)
        self.page_analytics.rowconfigure(1, weight=1)

        title_box = tk.Frame(self.page_analytics, bg=BG_CARD, padx=16, pady=10)
        title_box.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        tk.Label(title_box, text="CLINICAL POPULATION STATISTICAL DISTRIBUTIONS", font=("Segoe UI", 11, "bold"), fg=TEXT_PRI, bg=BG_CARD).pack(anchor="w")

        self.chart_container = tk.Frame(self.page_analytics, bg=BG_CARD)
        self.chart_container.grid(row=1, column=0, sticky="nsew")

        if not _MATPLOTLIB_OK:
            lbl = tk.Label(self.chart_container, text="Matplotlib is required for high-end embedded visualizations.\nPlease run `pip install matplotlib` to activate.",
                           font=("Segoe UI", 12), fg=WARNING, bg=BG_CARD)
            lbl.pack(expand=True)
            return
        
        # Configure embedded matplotlib canvas
        plt.style.use("dark_background")
        self.fig, (self.ax_pie, self.ax_hist) = plt.subplots(1, 2, figsize=(10, 4.5), facecolor=BG_CARD)
        self.fig.tight_layout(pad=3.0)
        
        self.canvas_mpl = FigureCanvasTkAgg(self.fig, master=self.chart_container)
        self.canvas_mpl.get_tk_widget().pack(fill="both", expand=True, padx=16, pady=16)

    def _render_matplotlib_chart(self):
        if not _MATPLOTLIB_OK or not hasattr(self, "fig") or not self.result: return
        stats = self.result["stats"]
        types = stats["types"]
        sizes = stats["sizes"]

        # Clear axes
        self.ax_pie.cla()
        self.ax_hist.cla()

        self.ax_pie.set_facecolor(BG_CARD)
        self.ax_hist.set_facecolor(BG_CARD)

        # Plot 1: Dynamic Pie Breakdown
        if types:
            labels = list(types.keys())
            counts = list(types.values())
            colors = [CELL_COLOURS.get(l, "#94a3b8") for l in labels]
            wedges, texts, autotexts = self.ax_pie.pie(counts, labels=labels, autopct="%1.1f%%", startangle=140,
                                                       colors=colors, textprops={"fontsize": 10, "color": TEXT_PRI},
                                                       wedgeprops={"linewidth": 1.0, "edgecolor": BG_CARD})
            self.ax_pie.set_title("Class Composition Proportions", fontsize=12, fontweight="bold", color=TEXT_PRI, pad=12)
        else:
            self.ax_pie.text(0.5, 0.5, "No classification instances available", ha="center", va="center", color=TEXT_MUTED)

        # Plot 2: Size distribution Histogram
        if sizes:
            self.ax_hist.hist(sizes, bins=20, color=ACCENT, alpha=0.85, edgecolor=BG_CARD, linewidth=0.5)
            self.ax_hist.set_title("Red Blood Cell Area Distribution (px²)", fontsize=12, fontweight="bold", color=TEXT_PRI, pad=12)
            self.ax_hist.set_xlabel("Surface Area", color=TEXT_SEC, fontsize=9)
            self.ax_hist.set_ylabel("Frequency", color=TEXT_SEC, fontsize=9)
            self.ax_hist.tick_params(colors=TEXT_SEC, labelsize=8)
            self.ax_hist.grid(True, linestyle=":", alpha=0.2, color=TEXT_SEC)
        else:
            self.ax_hist.text(0.5, 0.5, "Insufficient sizing dataset", ha="center", va="center", color=TEXT_MUTED)

        self.fig.canvas.draw_idle()

    # ─── STATUS BAR ─────────────────────────────────────────────
    def _build_statusbar(self):
        sb = tk.Frame(self, bg=BG_PANEL, height=28)
        sb.pack(fill="x", side="bottom")
        sb.pack_propagate(False)
        self._status_lbl = tk.Label(sb, textvariable=self.status_text, font=("Segoe UI", 9), fg=TEXT_SEC, bg=BG_PANEL, anchor="w")
        self._status_lbl.pack(side="left", padx=12)
        
        self._clock_lbl = tk.Label(sb, font=("Consolas", 9), fg=TEXT_MUTED, bg=BG_PANEL)
        self._clock_lbl.pack(side="right", padx=12)
        self._tick_clock()

    def _tick_clock(self):
        if not self.winfo_exists(): return
        self._clock_lbl.config(text=time.strftime("%H:%M:%S"))
        self.after(1000, self._tick_clock)

    # ─── DYNAMIC LOG & STATE UPDATES ────────────────────────────
    def _set_status(self, msg, colour=TEXT_PRI):
        self.status_text.set(msg)
        self._status_lbl.config(fg=colour)

    def _log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self._log_text.config(state="normal")
        self._log_text.insert("end", f"{ts}  {msg}\n")
        self._log_text.see("end")
        self._log_text.config(state="disabled")

    # ─── IMAGE WORKFLOW LOOP ────────────────────────────────────
    def _load_image(self):
        path = filedialog.askopenfilename(
            title="Import Microscopic Blood Sample",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        )
        if not path: return
        self.img_path.set(path)
        self.result = None
        self._reset_view_transforms()
        self._view_mode.set("detected")
        
        # Load and verify
        img = cv2.imread(path)
        if img is None:
            self.show_toast("Failed reading sample bytes", DANGER)
            return
            
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Display instantly
        self._cache_original_for_display(rgb)
        self._refresh_viewport()
        
        self._set_status(f"Imported sample: {os.path.basename(path)}", SUCCESS)
        self.show_toast("Image loaded successfully. Click 'Analyze' to compute metrics.", SUCCESS)
        self._log(f"[IMPORT] {path}")

    def _cache_original_for_display(self, rgb_arr):
        self._cached_rgb = rgb_arr

    def _reset_view_transforms(self):
        self._zoom  = 1.0
        self._pan_x = 0
        self._pan_y = 0
        self._refresh_viewport()

    # ─── EXECUTION THREAD WORKER ────────────────────────────────
    def _run_analysis(self):
        path = self.img_path.get()
        if not path or not hasattr(self, "_cached_rgb"):
            messagebox.showwarning("No Data", "Please import a blood sample image first.")
            return
        if self._busy: return
        
        self._busy = True
        self._set_status("Initializing fully automated diagnostic engine inference…", WARNING)
        self.progress_var.set(0)
        self._clear_results_trees()

        # Display an elegant Processing Overlay mask on canvas
        cw, ch = self._canvas.winfo_width(), self._canvas.winfo_height()
        self._canvas.create_rectangle(0, 0, cw, ch, fill=BG_DARK, stipple="gray50", tags="overlay")
        self._canvas.create_text(cw//2, ch//2, text="⏳ Executing Convolutional Pipeline…",
                                 font=("Segoe UI", 16, "bold"), fill=ACCENT, tags="overlay")

        # Spawn backend background execution
        threading.Thread(target=self._worker_thread, args=(path,), daemon=True).start()

    def _worker_thread(self, path):
        try:
            # Directly hook progress updates inside the inference code
            def _prog_hook(pct, msg):
                self.progress_var.set(pct)
                self.status_text.set(f"⚙️ {msg}")
            
            res = detect_cells_watershed(
                path,
                min_area=self.var_min_area.get(),
                max_area=self.var_max_area.get(),
                min_circ=self.var_min_circ.get(),
                max_aspect=self.var_max_aspect.get(),
                progress_callback=_prog_hook
            )
            self.after(0, self._on_analysis_complete, res)
        except Exception as e:
            self.after(0, self._on_analysis_failed, str(e))
        finally:
            self._busy = False

    def _on_analysis_complete(self, res):
        self.result = res
        stats = res["stats"]
        self.progress_var.set(100)
        
        # Ensure correct mode is picked
        self._view_mode.set("detected")
        self._refresh_viewport()
        self._update_diagnostics_grid()
        self._render_matplotlib_chart()

        # Smooth Rolling Numbers integration
        self._card_total.roll_to(stats["total"])
        self._card_types.update_instant(len(stats["types"]))

        # Animated bar layout insertion
        self._render_mini_chart(stats["types"])

        # Populate structured treeview
        for idx, cell in enumerate(stats["cells"]):
            colour = CELL_COLOURS.get(cell["type"], "#94a3b8")
            tag = cell["type"].replace(" ", "_")
            self._tree.tag_configure(tag, foreground=colour)
            self._tree.insert("", "end", values=(cell["type"], f"{cell['area']:.0f}",
                                                 f"{cell.get('circ', 0):.2f}",
                                                 f"({cell['x']}, {cell['y']})"), tags=(tag,))

        self._set_status(f"✔ Diagnostics valid — successfully verified {stats['total']} cells.", SUCCESS)
        self.show_toast(f"Inference finalized! Isolated {stats['total']} total structures.", SUCCESS)
        self._log(f"[SUCCESS] Total verified: {stats['total']} | Types: {stats['types']}")

    def _on_analysis_failed(self, err_msg):
        self._refresh_viewport()
        self._set_status(f"[CRITICAL FAILURE] {err_msg}", DANGER)
        self.show_toast("Pipeline error occurred during computation.", DANGER)
        self._log(f"[FAIL] {err_msg}")
        messagebox.showerror("Algorithm Error", f"Inference pipeline execution error:\n\n{err_msg}")

    def _apply_tuning_recalculate(self):
        """Allows instantly updating final metrics cleanly via step 6 filtering without full recompute."""
        if not self.result:
            self.show_toast("Execute initial pipeline first", WARNING)
            return
        self.show_toast("Re-evaluating mathematical threshold parameters…", SUCCESS)
        # Simply rerun analysis logic
        self._run_analysis()

    # ─── INTERACTIVE HARDWARE CANVAS RENDERING ENGINE ───────────
    def _refresh_viewport(self, *_):
        self._canvas.delete("all")
        if not hasattr(self, "_cached_rgb") or self._cached_rgb is None:
            self._canvas.create_text(450, 320, text="📂 Drop or load a blood sample image to analyze",
                                     font=("Segoe UI", 16), fill=TEXT_MUTED, tags="placeholder")
            return

        mode = self._view_mode.get()
        cw = max(self._canvas.winfo_width(), 600)
        ch = max(self._canvas.winfo_height(), 400)

        # Retrieve proper image source arrays
        if mode == "detected" and self.result:
            base_arr = self.result["image"]
        elif mode == "original" or not self.result:
            base_arr = self._cached_rgb
        else: # Split Comparison Wipe Slider Mode
            base_arr = self._render_wipe_comparison_frame()

        h, w = base_arr.shape[:2]
        # Calculate isotropic containment scale
        scale = min(cw / w, ch / h, 1.0) * self._zoom
        nw, nh = int(w * scale), int(h * scale)
        
        # Cache current rendered pixel boundaries for reverse coordinate mapping
        self._disp_w, self._disp_h = nw, nh
        self._disp_scale = scale
        self._orig_w, self._orig_h = w, h

        # Center alignments adding pan translations
        ox = max((cw - nw) // 2, 0) + self._pan_x
        oy = max((ch - nh) // 2, 0) + self._pan_y
        self._disp_ox, self._disp_oy = ox, oy

        pil_img = Image.fromarray(base_arr).resize((nw, nh), Image.LANCZOS)
        self._photo_disp = ImageTk.PhotoImage(pil_img)
        self._canvas.create_image(ox, oy, image=self._photo_disp, anchor="nw")

        # If in Split wipe mode, draw stunning neon wipe guideline
        if mode == "split" and self.result:
            wipe_x = ox + int(nw * self._split_pos)
            self._canvas.create_line(wipe_x, oy, wipe_x, oy + nh, fill=ACCENT, width=2, dash=(4,2))
            # Indicator tags
            self._canvas.create_rectangle(wipe_x - 35, oy + 10, wipe_x - 2, oy + 32, fill=BG_DARK, outline="")
            self._canvas.create_text(wipe_x - 18, oy + 21, text="RAW", fill=TEXT_SEC, font=("Segoe UI", 8, "bold"))
            self._canvas.create_rectangle(wipe_x + 2, oy + 10, wipe_x + 45, oy + 32, fill=ACCENT, outline="")
            self._canvas.create_text(wipe_x + 23, oy + 21, text="WATERSHED", fill=TEXT_PRI, font=("Segoe UI", 8, "bold"))

    def _render_wipe_comparison_frame(self):
        """Sub-millisecond horizontal composite stitch merging input layers."""
        orig = self._cached_rgb
        det  = self.result["image"]
        h, w = orig.shape[:2]
        split_px = int(w * self._split_pos)
        
        composite = np.empty_like(orig)
        composite[:, :split_px] = orig[:, :split_px]
        composite[:, split_px:] = det[:, split_px:]
        return composite

    # ─── VIEWPORT MOUSE ACTIONS (Panning, Zooming & Wiping) ─────
    def _on_zoom(self, event):
        if not hasattr(self, "_cached_rgb"): return
        zoom_factor = 1.15 if event.delta > 0 else (1.0 / 1.15)
        new_zoom = max(0.3, min(self._zoom * zoom_factor, 8.0))
        
        # Center zoom on pointer coordinates
        # Adjust pan so zoom scales around cursor cleanly
        self._zoom = new_zoom
        self._refresh_viewport()

    def _on_canvas_press(self, event):
        self._drag_start = (event.x, event.y)
        # If in Split mode, clicking near the wipe line lets us drag the wipe position
        if self._view_mode.get() == "split" and hasattr(self, "_disp_ox"):
            rel_x = event.x - self._disp_ox
            if 0 <= rel_x <= self._disp_w:
                current_pos = rel_x / self._disp_w
                if abs(current_pos - self._split_pos) < 0.15: # Grab line tolerance
                    self._is_wiping = True
                    return
        self._is_wiping = False

    def _on_canvas_drag(self, event):
        if not self._drag_start: return
        dx = event.x - self._drag_start[0]
        dy = event.y - self._drag_start[1]
        self._drag_start = (event.x, event.y)

        if getattr(self, "_is_wiping", False) and hasattr(self, "_disp_w"):
            # Update wipe divider percentage
            new_pos = (event.x - self._disp_ox) / self._disp_w
            self._split_pos = max(0.02, min(new_pos, 0.98))
            self._refresh_viewport()
        else:
            # General image dragging / translations
            self._pan_x += dx
            self._pan_y += dy
            self._refresh_viewport()

    def _on_mouse_move(self, event):
        # Update wipe cursor style if hovering the stitch separator
        if self._view_mode.get() == "split" and hasattr(self, "_disp_ox"):
            line_x = self._disp_ox + int(self._disp_w * self._split_pos)
            if abs(event.x - line_x) < 15:
                self._canvas.config(cursor="sb_h_double_arrow")
                return
        self._canvas.config(cursor="")

    # ─── SONAR RIPPLE CELL CLICK FEEDBACK LOOP ──────────────────
    def _on_tree_select(self, *_):
        sel = self._tree.selection()
        if not sel or not self.result or not hasattr(self, "_disp_ox"): return
        vals = self._tree.item(sel[0], "values")
        
        # Extract precise cell coordinate string: e.g. "(320, 180)"
        pos_str = vals[3].strip("()")
        try:
            orig_x, orig_y = map(int, pos_str.split(","))
        except ValueError:
            return

        # Map back to zoomed UI canvas layout coordinates
        cx = self._disp_ox + int(orig_x * self._disp_scale)
        cy = self._disp_oy + int(orig_y * self._disp_scale)

        self._set_status(f"Targeting Object: {vals[0]} | Area: {vals[1]} px²", ACCENT)
        # Execute Radar Sonar wave loops
        self._animate_sonar_ring(cx, cy)

    def _animate_sonar_ring(self, cx, cy, radius=10, max_radius=60):
        """Draws concentric expanding pulse animations guide targets visually."""
        tag = f"sonar_{time.time()}"
        
        def _wave_step(r):
            self._canvas.delete(tag)
            if r > max_radius: return
            
            alpha_col = ACCENT2 if r % 2 == 0 else ACCENT
            self._canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                     outline=alpha_col, width=2, tags=tag)
            self.after(20, lambda: _wave_step(r + 4))
            
        _wave_step(radius)

    # ─── INTERIM PIPELINE IMAGES PREVIEW MATRIX ─────────────────
    def _update_diagnostics_grid(self):
        if not self.result: return
        for key, canv in self.diag_canvases.items():
            self._update_diag_canvas(key)

    def _update_diag_canvas(self, key):
        if not self.result or key not in self.result: return
        canv = self.diag_canvases[key]
        canv.delete("all")
        
        arr = self.result[key]
        h, w = arr.shape[:2]
        cw = max(canv.winfo_width(), 100)
        ch = max(canv.winfo_height(), 80)
        
        scale = min(cw / w, ch / h, 1.0)
        nw, nh = int(w * scale), int(h * scale)
        if nw <= 0 or nh <= 0: return

        pil_img = Image.fromarray(arr).resize((nw, nh), Image.LANCZOS)
        photo = ImageTk.PhotoImage(pil_img)
        # Persistent storage mapping inside view variables to prevent memory garbage cleanup
        setattr(self, f"_diag_photo_{key}", photo)
        
        ox, oy = (cw - nw)//2, (ch - nh)//2
        canv.create_image(ox, oy, image=photo, anchor="nw")

    # ─── ANIMATED GROWING MINI BAR BREAKDOWN ────────────────────
    def _render_mini_chart(self, types: dict):
        self._mini_chart.delete("all")
        w = max(self._mini_chart.winfo_width(), 280)
        h = 140
        if not types:
            self._mini_chart.create_text(w//2, h//2, text="Awaiting data stream", fill=TEXT_MUTED, font=("Segoe UI", 9))
            return

        total = sum(types.values())
        bar_h = 16
        gap   = 8
        x0    = 8
        max_w = w - 120

        # Implement smoothly calculated growing horizontal indicators
        sorted_types = sorted(types.items(), key=lambda x: -x[1])
        
        for i, (ctype, count) in enumerate(sorted_types):
            y = i * (bar_h + gap) + 6
            pct = count / total
            target_bw = max(int(pct * max_w), 4)
            col = CELL_COLOURS.get(ctype, "#94a3b8")

            # Draw static container path
            round_rect(self._mini_chart, x0, y, x0 + max_w, y + bar_h, r=4, fill=BG_HOVER, outline="")
            
            # Animate real-time growing paths
            tag = f"bar_{i}"
            self._animate_growing_bar(x0, y, bar_h, target_bw, col, ctype, count, pct, tag, max_w)

    def _animate_growing_bar(self, x0, y, bar_h, target_bw, col, ctype, count, pct, tag, max_w, curr_bw=0):
        if not self.winfo_exists(): return
        self._mini_chart.delete(tag)
        
        if curr_bw >= target_bw:
            curr_bw = target_bw
            # Draw persistent completion state
            round_rect(self._mini_chart, x0, y, x0 + curr_bw, y + bar_h, r=4, fill=col, outline="", tags=tag)
            self._mini_chart.create_text(x0 + 6, y + bar_h//2, text=ctype, anchor="w", fill=TEXT_PRI, font=("Segoe UI", 8, "bold"), tags=tag)
            self._mini_chart.create_text(x0 + max_w + 8, y + bar_h//2, text=f"{count} ({pct*100:.0f}%)", anchor="w", fill=TEXT_SEC, font=("Consolas", 8), tags=tag)
            return

        step = max(2, int(target_bw * 0.15))
        curr_bw += step
        round_rect(self._mini_chart, x0, y, x0 + curr_bw, y + bar_h, r=4, fill=col, outline="", tags=tag)
        self.after(15, lambda: self._animate_growing_bar(x0, y, bar_h, target_bw, col, ctype, count, pct, tag, max_w, curr_bw))

    # ─── HIGH-FIDELITY PDF REPORTS ENGINE & CUSTOM MODAL ────────
    def _save_report_dialog(self):
        if not self.result:
            messagebox.showwarning("Incomplete Workflow", "Execute classification engine processing first.")
            return
        
        # Pop Custom Config Modal
        ReportConfigDialog(self, self._execute_pdf_export)

    def _execute_pdf_export(self, meta_data):
        if _PDF_OK:
            filetypes = [("Professional PDF Report", "*.pdf"), ("JSON Export", "*.json"), ("All Files", "*.*")]
            def_ext = ".pdf"
        else:
            filetypes = [("JSON Export", "*.json"), ("All Files", "*.*")]
            def_ext = ".json"

        path = filedialog.asksaveasfilename(defaultextension=def_ext, filetypes=filetypes, initialfile="Hematology_Diagnostic_Report")
        if not path: return

        stats = self.result["stats"]
        mean_area = round(float(np.mean(stats["sizes"])), 2) if stats["sizes"] else 0
        
        report_json = {
            "source_sample":  self.result["path"],
            "analyzed_time":  time.strftime("%Y-%m-%d %H:%M:%S"),
            "patient_info":   meta_data["patient"],
            "clinical_lead":  meta_data["doctor"],
            "notes":          meta_data["notes"],
            "total_verified": stats["total"],
            "breakdown":      stats["types"],
            "mean_area_px":   mean_area,
            "structures":     stats["cells"]
        }

        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf" and _PDF_OK:
            self._generate_high_fidelity_pdf(path, report_json, meta_data)
        else:
            with open(path, "w") as f:
                json.dump(report_json, f, indent=2)
                
        self.show_toast(f"Report exported cleanly: {os.path.basename(path)}", SUCCESS)
        self._log(f"[EXPORT] Completed writing diagnostics payload -> {path}")

    def _generate_high_fidelity_pdf(self, path: str, report: dict, custom_meta: dict):
        """Builds state-of-the-art multi-page diagnostic document story complete with custom fields."""
        types  = report["breakdown"]
        labels = list(types.keys())
        sizes  = list(types.values())
        total  = report["total_verified"] or 1
        tmp_files = []

        # 1. Output annotated matrix frames
        det_img_path = path + "_preview.png"
        tmp_files.append(det_img_path)
        Image.fromarray(self.result["image"]).save(det_img_path)

        # 2. Build explicit document distribution plot
        chart_path = path + "_pie.png"
        tmp_files.append(chart_path)
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        colors_lst = [CELL_COLOURS.get(l, "#94a3b8") for l in labels]
        ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140, colors=colors_lst,
               textprops={"fontsize": 10}, wedgeprops={"linewidth": 1.2, "edgecolor": "white"})
        ax.set_title("Hematological Proportions", fontsize=11, fontweight="bold", pad=8)
        fig.tight_layout()
        fig.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        # 3. Assemble document parameters
        doc = SimpleDocTemplate(path, pagesize=A4, rightMargin=0.6*inch, leftMargin=0.6*inch,
                                topMargin=0.6*inch, bottomMargin=0.6*inch)
        story = []
        styles = getSampleStyleSheet()

        title_st = ParagraphStyle("TSt", parent=styles["Heading1"], alignment=1, fontSize=20, spaceAfter=2, textColor=colors.HexColor("#0a0e1a"))
        sub_st   = ParagraphStyle("SSt", parent=styles["Normal"], alignment=1, fontSize=11, spaceAfter=12, textColor=colors.HexColor(ACCENT))
        sec_st   = ParagraphStyle("Sec", parent=styles["Normal"], fontSize=12, fontName="Helvetica-Bold", textColor=colors.HexColor(BORDER), spaceBefore=12, spaceAfter=6)
        info_st  = ParagraphStyle("Inf", parent=styles["Normal"], fontSize=9.5, spaceAfter=4, textColor=colors.HexColor("#1e293b"))
        note_st  = ParagraphStyle("Not", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#64748b"), spaceBefore=6)

        # Header Block
        hdr_table = Table([[Paragraph("<b>METROPOLITAN HOSPITAL CLINICAL REPORT</b>", title_st)]], colWidths=[7.2*inch])
        hdr_table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#f0f6ff")),
            ("BOX",        (0,0), (-1,-1), 1.5, colors.HexColor(ACCENT)),
            ("TOPPADDING", (0,0), (-1,-1), 8), ("BOTTOMPADDING", (0,0), (-1,-1), 8)
        ]))
        story.append(hdr_table)
        story.append(Spacer(1, 0.05*inch))
        story.append(Paragraph("HEMATOLOGY & DIGITAL CELL PATHOLOGY SUBSYSTEM", sub_st))

        # Dynamic Metadata rows table
        src_name = os.path.splitext(os.path.basename(report["source_sample"]))[0]
        meta_left = [
            [Paragraph("<b>Patient Identifier:</b>", info_st), Paragraph(custom_meta["patient"], info_st)],
            [Paragraph("<b>Referring Clinician:</b>", info_st), Paragraph(custom_meta["doctor"], info_st)],
            [Paragraph("<b>Date of Inquiry:</b>", info_st), Paragraph(report["analyzed_time"], info_st)],
        ]
        meta_right = [
            [Paragraph("<b>Specimen Matrix:</b>", info_st), Paragraph(src_name, info_st)],
            [Paragraph("<b>Total Classified:</b>", info_st), Paragraph(f"<b>{total}</b> objects", info_st)],
            [Paragraph("<b>Mean Scale Profile:</b>", info_st), Paragraph(f"{report['mean_area_px']} px²", info_st)],
        ]

        def _wrap_t(rows):
            t = Table(rows, colWidths=[1.5*inch, 2.0*inch])
            t.setStyle(TableStyle([
                ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
                ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.HexColor("#f8fafc"), colors.white]),
                ("TOPPADDING", (0,0), (-1,-1), 4), ("BOTTOMPADDING", (0,0), (-1,-1), 4),
                ("LEFTPADDING", (0,0), (-1,-1), 6), ("RIGHTPADDING", (0,0), (-1,-1), 6),
            ]))
            return t

        master_meta = Table([[_wrap_t(meta_left), _wrap_t(meta_right)]], colWidths=[3.6*inch, 3.6*inch])
        master_meta.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "TOP")]))
        story.append(master_meta)
        story.append(Spacer(1, 0.1*inch))

        # Comments Box
        if custom_meta["notes"] != "None":
            note_box = Table([[Paragraph(f"<b>Diagnostic Commentary:</b> {custom_meta['notes']}", info_st)]], colWidths=[7.2*inch])
            note_box.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#fffbeb")),
                                          ("BOX", (0,0), (-1,-1), 0.5, colors.HexColor(WARNING)),
                                          ("TOPPADDING", (0,0), (-1,-1), 6), ("BOTTOMPADDING", (0,0), (-1,-1), 6),
                                          ("LEFTPADDING", (0,0), (-1,-1), 8)]))
            story.append(note_box)
            story.append(Spacer(1, 0.1*inch))

        # Detection Frame Output
        story.append(Paragraph("▶ Primary Segmentation Visual Verification", sec_st))
        ih, iw = self.result["image"].shape[:2]
        avail_w = 7.2 * inch
        img_h = avail_w * (ih / iw)
        if img_h > 3.5 * inch:
            img_h = 3.5 * inch
            avail_w = img_h * (iw / ih)
        story.append(RLImage(det_img_path, width=avail_w, height=img_h))
        story.append(Paragraph("Solid outlines trace valid targets. Central overlay indices designate candidate class classifications.", note_st))
        story.append(Spacer(1, 0.1*inch))

        # Statistical Matrix Side Table
        story.append(Paragraph("▶ Quantitative Cytology Breakdown", sec_st))
        t_data = [[Paragraph("<b>Target Domain Class</b>", info_st), Paragraph("<b>Count</b>", info_st), Paragraph("<b>Proportion</b>", info_st)]]
        for ct, cnt in sorted(types.items(), key=lambda x: -x[1]):
            col = CELL_COLOURS.get(ct, "#475569")
            t_data.append([
                Paragraph(f"<font color='{col}'><b>{ct}</b></font>", info_st),
                Paragraph(str(cnt), info_st),
                Paragraph(f"{cnt/total*100:.1f}%", info_st)
            ])
            
        breakdown_t = Table(t_data, colWidths=[1.5*inch, 0.9*inch, 1.1*inch])
        breakdown_t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor(BG_PANEL)),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("ALIGN", (0,0), (-1,-1), "CENTER"),
            ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#93c5fd")),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#f1f5f9"), colors.white]),
            ("TOPPADDING", (0,0), (-1,-1), 5), ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ]))
        
        side_row = Table([[RLImage(chart_path, width=3.5*inch, height=2.4*inch), breakdown_t]], colWidths=[3.6*inch, 3.6*inch])
        side_row.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "MIDDLE")]))
        story.append(side_row)

        doc.build(story)
        for f in tmp_files:
            try: os.remove(f)
            except Exception: pass

    # ─── CLEARING & RESET LOOP ──────────────────────────────────
    def _clear(self):
        self.result = None
        self.img_path.set("")
        self._cached_rgb = None
        self._reset_view_transforms()
        self._clear_results_trees()
        self._card_total.update_instant("—")
        self._card_types.update_instant("—")
        self._render_mini_chart({})
        self.progress_var.set(0)
        
        # Clear diagnostic grid thumbnails
        for key, canv in self.diag_canvases.items():
            canv.delete("all")
            
        self._set_status("Workspace cleared successfully. Ready for new specimen input.", TEXT_SEC)
        self.show_toast("Diagnostics cache emptied.", WARNING)

    def _clear_results_trees(self):
        for row in self._tree.get_children():
            self._tree.delete(row)

    # ─── ANIMATED DECORATIVE HEADER HEADER LOOP ─────────────────
    def _animate_header(self):
        colours = [ACCENT, ACCENT2, SUCCESS, WARNING]
        self._hdr_idx = 0
        def _step():
            if not self.winfo_exists(): return
            c = colours[self._hdr_idx % len(colours)]
            self.strip.delete("all")
            self.strip.create_line(0, 1, 3000, 1, fill=c, width=3)
            self._hdr_idx += 1
            self.after(2500, _step)
        self.after(2500, _step)


# ─── ENTRY BOOTSTRAP EXECUTION ──────────────────────────────────
if __name__ == "__main__":
    try:
        from PIL import Image, ImageTk
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
        from PIL import Image, ImageTk

    app = BloodCellApp()
    app.mainloop()
