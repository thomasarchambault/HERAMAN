# Rôle : Fournir les widgets Qt réutilisables, dont le canevas de visualisation des images brutes.

from __future__ import annotations

import contextlib
import datetime
import io
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from subprocess import PIPE, STDOUT, Popen

import cv2
import joblib
import numpy as np
import pyqtgraph as pg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import QEvent, QObject, Qt, QTimer
from PyQt5.QtGui import QColor, QFont, QImage, QIntValidator, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QColorDialog, QComboBox, QFileDialog, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLayout, QLineEdit, QMessageBox,
    QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy, QSlider, QTabWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from shared.paths import PROJECT_ROOT


class ResizablePixmapLabel(QLabel):
    """A QLabel that scales its pixmap to fit while maintaining aspect ratio."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._orig_pix = None
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAlignment(Qt.AlignCenter)
        self.setScaledContents(False)  # on gère nous-mêmes le scale

    def setPixmap(self, pixmap: QPixmap):
        # mémorise l’original pour les prochains resize
        self._orig_pix = pixmap
        if pixmap is not None and not self.size().isEmpty():
            # scale immédiat à la taille actuelle du widget
            scaled = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            super().setPixmap(scaled)
        else:
            super().setPixmap(pixmap)
        self.update()  # force un repaint immédiat

    def clear(self):
        self._orig_pix = None
        super().clear()

    def resizeEvent(self, event):
        # re-scale à chaque resize à partir de l’original
        if self._orig_pix is not None and not self.size().isEmpty():
            scaled = self._orig_pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            super().setPixmap(scaled)
        super().resizeEvent(event)

class DNLiveCanvas(FigureCanvas):
    """Matplotlib canvas for live raw-DN histogram.

    The camera frames are kept in their native DN scale for statistics:
    - uint8  -> 0..255
    - uint16 -> 0..65535

    For 16-bit images, the histogram uses a reduced number of bins for speed,
    but the x-axis and mean/target values remain true raw DN.
    """
    def __init__(self, parent=None, max_dn=65535, target_fraction=0.70, n_bins=128, max_hist_samples=100_000):
        fig = Figure(figsize=(4, 1.5), tight_layout=True)
        super().__init__(fig)
        self.setParent(parent)
        self.ax = fig.add_subplot(111)

        self.max_dn = float(max_dn)
        self.target_fraction = float(target_fraction)
        self.target_dn = self.target_fraction * self.max_dn
        self.n_bins = int(n_bins)
        self.max_hist_samples = int(max_hist_samples)

        self.sat_count = 0
        self.mean_line = None
        self.target_line = None
        self.hist = None
        self.bin_edges = None
        self.bin_centers = None
        self.last_mean = None
        self._init_plot()

    def set_dynamic_range(self, max_dn: float, target_fraction: float | None = None):
        """Update histogram dynamic range when the camera bit depth is detected."""
        self.max_dn = float(max_dn)
        if target_fraction is not None:
            self.target_fraction = float(target_fraction)
        self.target_dn = self.target_fraction * self.max_dn
        self._init_plot()

    def _init_plot(self):
        self.ax.clear()
        self.ax.set_facecolor("none")
        self.figure.set_facecolor("none")

        # Keep the histogram light enough for live display.
        # Fewer bins still display true 16-bit DN on the x-axis; only the bin width changes.
        n_bins = self.n_bins if self.max_dn > 255 else 256
        self.bin_edges = np.linspace(0.0, self.max_dn, n_bins + 1)
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        widths = np.diff(self.bin_edges)

        colors = ["#42A5F5"] * n_bins
        colors[-1] = "#FF1744"

        self.hist = self.ax.bar(
            self.bin_centers,
            np.zeros(n_bins),
            width=widths,
            align="center",
            color=colors,
            edgecolor="white",
            linewidth=0.15,
            alpha=0.95,
        )

        self.ax.set_xlim(0, self.max_dn)
        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel("Raw DN", color="white")
        self.ax.set_ylabel("Nb pixels", color="white")
        self.ax.set_title("Histogramme DN brut", color="white")

        # Saturation zone: last ~1% of the raw range.
        sat_start = 0.99 * self.max_dn
        self.sat_patch = self.ax.axvspan(
            sat_start,
            self.max_dn,
            color="#D32F2F",
            alpha=0.25,
            label="Saturation"
        )

        self.target_line = self.ax.axvline(
            self.target_dn,
            color="orange",
            linestyle="--",
            linewidth=2,
            label=f"Target {self.target_fraction * 100:.0f}% = {self.target_dn:.0f} DN"
        )

        self.ax.tick_params(axis="x", colors="white")
        self.ax.tick_params(axis="y", colors="white")
        for spine in self.ax.spines.values():
            spine.set_color("white")

        leg = self.ax.legend()
        if leg:
            for t in leg.get_texts():
                t.set_color("white")
            leg.get_frame().set_facecolor("none")
            leg.get_frame().set_edgecolor("white")

        self.draw()

    def update_hist(self, img):
        """Update histogram from raw ROI pixels, not display-converted 8-bit pixels."""
        if img is None:
            return

        arr = np.asarray(img)
        if arr.size == 0:
            return

        arr = arr.astype(np.float64, copy=False)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return

        arr = np.clip(arr, 0, self.max_dn)

        # Histogram speed optimization:
        # compute the displayed histogram on a deterministic subsample if the ROI is huge.
        # The mean line still uses all ROI pixels for accuracy.
        hist_arr = arr
        if hist_arr.size > self.max_hist_samples:
            step = int(np.ceil(hist_arr.size / self.max_hist_samples))
            hist_arr = hist_arr[::step]

        h, _ = np.histogram(hist_arr, bins=self.bin_edges)

        for rect, val in zip(self.hist, h):
            rect.set_height(int(val))

        self.ax.set_ylim(0, max(int(h.max()), 1))

        mean_val = float(np.mean(arr))
        self.last_mean = mean_val

        if self.mean_line is not None:
            try:
                self.mean_line.remove()
            except Exception:
                pass
        self.mean_line = self.ax.axvline(mean_val, color="lime", linestyle="-", linewidth=2)

        # Tolerance for the green/red frame: ±1% of the full dynamic range.
        close = abs(mean_val - self.target_dn) <= 0.01 * self.max_dn
        for spine in self.ax.spines.values():
            spine.set_edgecolor("#00E676" if close else "#FF1744")
            spine.set_linewidth(3)

        self.draw_idle()

class SpectraInteractive(QWidget):
    """
    Trace interactif des spectres par classe à partir d'un fichier spectra_data.npz.
    On n'affiche que les classes passées via set_selected_classes().
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        import pyqtgraph as pg
        pg.setConfigOptions(antialias=True)  # tracés plus lisses

        self._plot_items = []   # refs fortes pour courbes/fill (évite GC)
        self._data = None
        self._selected = set()
        self.show_lib = True
        self.show_std = True
        self.show_img = True

        lay = QVBoxLayout(self)

        # --- Graphe + légende externe (hors du tracé) ---
        self.glw = pg.GraphicsLayoutWidget()
        self.plot = self.glw.addPlot(row=0, col=0)

        # Légende dans une colonne séparée (à droite du graphe)
        self.legend = pg.LegendItem(brush=pg.mkBrush(40, 40, 40, 180), labelTextColor='w')
        self.glw.addItem(self.legend, row=0, col=1)

        # --- fond sombre + axes lisibles ---
        self.glw.setBackground('#1e1e1e') 
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        for ax in ('left', 'bottom'):
            self.plot.getAxis(ax).setPen((230, 230, 230))
            self.plot.getAxis(ax).setTextPen((230, 230, 230))
        self.plot.setLabel('bottom', 'Band index', **{'color': 'w'})
        self.plot.setLabel('left', 'Normalized intensity', **{'color': 'w'})

        lay.addWidget(self.glw)


    def load_data(self, spectra_npz_path: str | Path):

        p = Path(spectra_npz_path)
        if not p.exists():
            print("[SPECTRA][ERR] file not found in load_data()")
            self._data = None
            self.plot.clear()
            self.plot.addItem(pg.TextItem(f"Missing: {p.name}", anchor=(0,1)))
            return

        d = np.load(p, allow_pickle=True)
        classes_arr = d.get("class_names", d.get("classes"))
        if classes_arr is None:
            print("[SPECTRA][ERR] no class list in npz")
            self._data = None
            self.plot.clear()
            self.plot.addItem(pg.TextItem("spectra_data.npz invalide (pas de classes)", anchor=(0,1)))
            return
        classes = [str(c) for c in classes_arr.tolist()]

        if "wavelengths" in d.files:
            bands = d["wavelengths"].astype(float)
        elif "bands" in d.files:
            bands = d["bands"].astype(float)
        elif "bands_idx" in d.files:
            bands = d["bands_idx"].astype(float)
        else:
            bands = np.arange(d["lib_median"].shape[1], dtype=np.float64)

        lib_median = d["lib_median"].astype(float)
        lib_std    = d.get("lib_stdev", d.get("lib_std", None))
        img_mean   = d.get("img_mean", None)
        if img_mean is not None: img_mean = img_mean.astype(float)

        if lib_std is not None:
            lib_std = np.clip(np.nan_to_num(lib_std.astype(float), nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)

        colors_bgr = d.get("class_colors_bgr", None)
        if colors_bgr is not None: colors_bgr = colors_bgr.astype(np.uint8)

        # aligner
        Bm = lib_median.shape[1]
        Bs = lib_std.shape[1] if lib_std is not None else Bm
        Bi = img_mean.shape[1] if img_mean is not None else Bm
        Bb = bands.shape[0]
        m = min(Bb, Bm, (Bs if lib_std is not None else Bm), (Bi if img_mean is not None else Bm))

        bands = bands[:m]
        lib_median = lib_median[:, :m]
        if lib_std is not None: lib_std = lib_std[:, :m]
        if img_mean is not None: img_mean = img_mean[:, :m]

        self._data = {
            "classes": classes,
            "bands": bands,
            "lib_med": lib_median,
            "lib_std": lib_std,
            "img_mean": img_mean,
            "colors_bgr": colors_bgr,
        }



    def set_selected_classes(self, selected_names: list[str]):
        self._selected = set(selected_names)
        self.update_plot()

    def update_plot(self):
        # reset
        self.plot.clear()
        # --- visuel/range sûrs ---
        vb = self.plot.getViewBox()
        vb.enableAutoRange(x=True, y=True)
        vb.autoRange()

        try:
            self.legend.clear()
        except Exception:
            pass
        # nettoie refs fortes
        for it in getattr(self, "_plot_items", []):
            try:
                self.plot.removeItem(it)
            except Exception:
                pass
        self._plot_items = []
        self._legend_items = []

        if not getattr(self, "_data", None):
            return

        bands    = np.asarray(self._data["bands"], float)
        classes  = list(self._data["classes"])
        lib_med  = np.asarray(self._data["lib_med"], float)
        lib_std  = self._data.get("lib_std", None)
        img_mean = self._data.get("img_mean", None)
        colors_bgr = self._data.get("colors_bgr", None)

        if img_mean is not None: img_mean = np.asarray(img_mean, float)
        if lib_std  is not None: lib_std  = np.asarray(lib_std,  float)

        selected = set(getattr(self, "_selected", [])) or set(classes)

        def rgb_from_bgr_row(row):
            b, g, r = int(row[0]), int(row[1]), int(row[2])
            return (r, g, b)

        for i, cname in enumerate(classes):
            if cname not in selected:
                continue

            if colors_bgr is not None and i < len(colors_bgr):
                r, g, b = rgb_from_bgr_row(colors_bgr[i])
            else:
                r, g, b = (200, 200, 200)

            x     = bands
            y_med = lib_med[i]
            y_img = (img_mean[i] if img_mean is not None and i < img_mean.shape[0] else None)
            y_std = (lib_std[i]  if lib_std  is not None and i < lib_std.shape[0]  else None)

            
            lib_pen = pg.mkPen((r, g, b), width=2.2)
            img_pen = pg.mkPen((r, g, b), width=2.0, style=Qt.CustomDashLine, dash=[2, 6])


            # médiane
            if getattr(self, "show_lib", True):
                c_lib = self.plot.plot(x, y_med, pen=lib_pen, name=f"{cname} (lib)", connect='finite')
                c_lib.setZValue(10)  # au-dessus du fill
                self._plot_items.append(c_lib)
                try: self.legend.addItem(c_lib, f"{cname} (lib)")
                except Exception: pass

            # img pointillé
            if getattr(self, "show_img", True) and y_img is not None:
                c_img = self.plot.plot(x, y_img, pen=img_pen, name=f"{cname} (img)", connect='finite')
                c_img.setZValue(9)   # au-dessus du fill
                self._plot_items.append(c_img)
                try: self.legend.addItem(c_img, f"{cname} (img)")
                except Exception: pass

            # ±1σ — zone ombrée (version robuste avec PlotCurveItem explicites)
            if getattr(self, "show_std", True) and y_std is not None:
                if np.isfinite(y_std).any() and np.nanmax(y_std) > 0:
                    try:
                        y_up = y_med + y_std
                        y_lo = y_med - y_std

                        # IMPORTANT: utiliser PlotCurveItem explicites (pas self.plot.plot) et éviter pen=None
                        cu = pg.PlotCurveItem(x, y_up, pen=pg.mkPen((r, g, b, 1)))   # trait quasi invisible mais chemin valide
                        cl = pg.PlotCurveItem(x, y_lo, pen=pg.mkPen((r, g, b, 1)))
                        cu.setZValue(8)
                        cl.setZValue(8)
                        self.plot.addItem(cu)
                        self.plot.addItem(cl)

                        fill = pg.FillBetweenItem(cu, cl, brush=pg.mkBrush(r, g, b, 30))
                        fill.setZValue(7)  # juste sous les lignes (lignes à 9/10 plus bas)
                        self.plot.addItem(fill)

                        # garder des refs
                        self._plot_items.extend([cu, cl, fill])

                    except Exception as e:
                        print(f"[SPECTRA][ERR] FillBetween for '{cname}' failed: {e!r}")
        # --- Normalisation du zoom (dézoomer proprement après le tracé) ---
        try:

            vb = self.plot.getViewBox()

            # bornes X
            x = np.asarray(self._data["bands"], float)
            xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))

            # bornes Y = tout ce qui est affiché (médianes, img_mean, et si dispo mu±σ)
            lib_med  = np.asarray(self._data["lib_med"], float)
            lib_std  = self._data.get("lib_std")
            img_mean = self._data.get("img_mean")

            y_vals = []
            # classes sélectionnées uniquement
            selected = set(getattr(self, "_selected", [])) or set(self._data["classes"])
            for i, cname in enumerate(self._data["classes"]):
                if cname not in selected:
                    continue
                y_vals.append(lib_med[i])
                if img_mean is not None and i < np.shape(img_mean)[0]:
                    y_vals.append(np.asarray(img_mean[i], float))
                if lib_std is not None and i < np.shape(lib_std)[0]:
                    y_vals.append(np.asarray(lib_med[i] + lib_std[i], float))
                    y_vals.append(np.asarray(lib_med[i] - lib_std[i], float))

            Y = np.vstack([np.asarray(y, float) for y in y_vals if y is not None])
            ymin, ymax = float(np.nanmin(Y)), float(np.nanmax(Y))

            # petit padding pour aérer
            ypad = 0.05 * (ymax - ymin) if np.isfinite([ymin, ymax]).all() else 0.1
            if not np.isfinite(ypad) or ypad == 0:
                ypad = 0.1

            # appliquer la fenêtre & verrouiller l'autorange
            vb.enableAutoRange(x=False, y=False)
            vb.setXRange(xmin, xmax, padding=0.02)
            vb.setYRange(ymin - ypad, ymax + ypad, padding=0.0)
        except Exception as e:
            print(f"[SPECTRA][WARN] normalize zoom failed: {e!r}")

def color_for(cls: str) -> tuple[int, int, int]:
    """Return the fixed BGR color used by the 64-bit classifier."""
    cls = cls.lower().strip()
    palette = {
        "background": (0, 0, 0),
        "fat":        (255, 0, 0),
        "muscle":     (0, 255, 180),
        "glandular":  (0, 0, 255),
        "stroma":     (255, 0, 255),
        "yellow ink": (0, 255, 255),
        "black ink":  (80, 80, 80),
        "petri":      (255, 255, 255),
        "shade":      (80, 80, 80),
    }
    return palette.get(cls, (128, 128, 128))
