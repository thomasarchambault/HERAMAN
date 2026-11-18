# Rôle : Construire l’onglet d’affichage et d’interaction avec les cartes d’abondance et de classification.

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
from shared.image_io import read_image, write_image
import joblib
import numpy as np
import pyqtgraph as pg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import QEvent, QObject, QSize, Qt, QTimer
from PyQt5.QtGui import QColor, QFont, QIcon, QImage, QIntValidator, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLayout, QLineEdit, QMessageBox,
    QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy, QSlider, QTabWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from shared.paths import PROJECT_ROOT

from gui.widgets import ResizablePixmapLabel
from gui.inspection_widgets import ZoomablePixmapLabel
from gui.features.overlay_spectra import (
    class_references, load_decision_scores, load_platt_probabilities, platt_overlay,
)
from shared.probability_colors import FAT_CLASS_LEGEND
from shared.svm_calibration import spatial_fat_membership_labels, fat_membership_labels, DEFAULT_FAT_THRESHOLD, DEFAULT_MEAT_THRESHOLD
from shared.probability_colors import probability_rgb


class OverlayMixin:
    def _make_abundance_overlay_tab(self, *, embedded: bool = False) -> QWidget:

        tab = QWidget()
        root = QVBoxLayout(tab)
        root.setContentsMargins(8 if embedded else 10, 8 if embedded else 10, 8 if embedded else 10, 8 if embedded else 10)
        root.setSpacing(6 if embedded else 8)

        title = QLabel(
            "<b>Image · live pixel spectrum</b>"
            if embedded else "<b>Overlay · explore tissue predictions</b>"
        )
        root.addWidget(title, 0)

        # -------------------------
        # State
        # -------------------------
        self._ab_base_bgr = None            # (H,W,3) uint8 BGR
        self._ab_abund = None               # (H,W,C) float32
        self._ab_names = None               # list[str]
        self._ab_colors_bgr = None          # (C,3), same palette as mixture_side_by_side.png
        self._ab_cube = None                # (H,W,B), native values used for spectra
        self._ab_tissue_mask = None
        self._ab_wavelengths = None         # (B,) wavelengths really present in the loaded cube
        self._ab_cube_ymax = 1.0             # y-axis max for spectra, inferred from cube actual values
        self._ab_cube_dtype = "n/a"          # original cube dtype shown in spectrum info
        self._ab_cube_vmax = 0.0             # original cube max shown in spectrum info
        self._ab_last_folder = getattr(self, "_ab_last_folder", "")
        self.ab_class_checks = {}

        self._ab_hover_xy = None            # (x,y) coords image
        self._ab_hover_radius = 0
        self._ab_pinned = False
        self._ab_class_map = None
        self._ab_references = {}
        self._ab_decision = None
        self._ab_probability = None
        self._ab_cleanup_cache = None
        self._ab_probability_native = None
        self._ab_spectrum_limits = (0., 1.)
        self._ab_spectrum_context = None
        self._ab_normalized_refs = None
        self._ab_hover_color = (0, 255, 0)  # BGR
        self._ab_show_hover_box = True

        # mapping label coords -> image coords
        self._ab_last_img_shape = None      # (H,W)
        self._ab_last_pix_size = None       # (pw,ph)
        self._ab_last_offsets = None        # (ox,oy)

        # -------------------------
        # Palette per class (BGR)
        # -------------------------
        def _fallback_color(cls: str) -> tuple[int, int, int]:
            c = (cls or "").strip().lower()
            palette = {
                "background": (0, 0, 0),
                "fat": (255, 0, 0),
                "muscle": (0, 255, 180),
                "meat": (0, 255, 180),
                "glandular": (0, 0, 255),
                "stroma": (255, 0, 255),
                "yellow ink": (0, 255, 255),
                "black ink": (80, 80, 80),
                "petri": (255, 255, 255),
                "shade": (80, 80, 80),
            }
            return palette.get(c, (128, 128, 128))

        def _color_for(cls: str) -> tuple[int, int, int]:
            names = list(getattr(self, "_ab_names", None) or [])
            colors = getattr(self, "_ab_colors_bgr", None)
            target = (cls or "").strip().lower()
            if colors is not None:
                colors = np.asarray(colors)
                for index, name in enumerate(names):
                    if name.strip().lower() == target and index < len(colors):
                        return tuple(int(value) for value in colors[index])
            return _fallback_color(cls)

        def _class_swatch_icon(cls: str) -> QIcon:
            b, g, r = _color_for(cls)
            pixmap = QPixmap(14, 14)
            pixmap.fill(Qt.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setBrush(QColor(r, g, b))
            painter.setPen(QPen(QColor("#9a9a9a"), 1))
            painter.drawRoundedRect(1, 1, 12, 12, 2, 2)
            painter.end()
            return QIcon(pixmap)

        # =========================
        # 1) Folder selection row
        # =========================
        row = QHBoxLayout()
        row.addWidget(QLabel("Corrected folder:"), 0)

        self.ab_dir_input = QLineEdit()
        self.ab_dir_input.setPlaceholderText(r"...\corrected\*_corrected")
        row.addWidget(self.ab_dir_input, 1)

        btn_browse = QPushButton("Browse…")
        row.addWidget(btn_browse, 0)
        if not embedded:
            root.addLayout(row, 0)

        self.ab_folder_label = QLabel("No folder selected")
        self.ab_folder_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        root.addWidget(self.ab_folder_label, 0)
        root.addWidget(self._make_cube_controls(), 0)

        # =========================
        # 2) Controls row
        # =========================
        controls_group = QGroupBox("Overlay display")
        controls_layout = QVBoxLayout(controls_group)
        controls_layout.setContentsMargins(10, 16, 10, 8)
        controls_layout.setSpacing(6)
        metrics_row = QHBoxLayout()
        metrics_row.setSpacing(18)
        for key, caption, value in (('meat', 'Max non-fat: P(fat)', DEFAULT_MEAT_THRESHOLD),
                                    ('fat', 'Min fat: P(fat)', DEFAULT_FAT_THRESHOLD)):
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 1000)
            slider.setValue(round(value*1000))
            slider.setEnabled(False)
            slider.setMinimumWidth(100)
            label = QLabel(f'{value:.1%}')
            label.setFixedWidth(55)
            setattr(self, f'ab_{key}_slider', slider)
            setattr(self, f'ab_{key}_label', label)

            slider.hide()
            label.hide()
        controls_layout.addLayout(metrics_row)
        metrics_row = QHBoxLayout()

        opacity_caption = QLabel("Opacity:")
        opacity_caption.setToolTip(
            "Controls the strength of the class colours and uncertainty display.\n"
            "At 0, the image remains unchanged. At 1, the overlay is fully visible."
        )
        self.ab_alpha_slider = QSlider(Qt.Horizontal)
        self.ab_alpha_slider.setRange(0, 100)
        self.ab_alpha_slider.setValue(45)
        self.ab_alpha_slider.setToolTip(opacity_caption.toolTip())
        self.ab_alpha_slider.setFixedWidth(170)
        self.ab_alpha_label = QLabel("0.50")
        self.ab_alpha_label.setFixedWidth(50)
        opacity_control = QHBoxLayout()
        opacity_control.setSpacing(7)
        opacity_control.addWidget(opacity_caption)
        opacity_control.addWidget(self.ab_alpha_slider)
        opacity_control.addWidget(self.ab_alpha_label)
        metrics_row.addLayout(opacity_control)
        metrics_row.addStretch(1)
        controls_layout.addLayout(metrics_row)

        options_row = QHBoxLayout()
        options_row.setSpacing(12)
        btn_refresh = QPushButton("Refresh")
        options_row.addWidget(btn_refresh)
        options_row.addSpacing(8)
        options_row.addWidget(QLabel("Display:"))
        self.ab_display_mode = QComboBox()
        self.ab_display_mode.addItem("Classes", "classes")
        self.ab_display_mode.setToolTip("Saved class masks for the selected model.")
        self.ab_display_mode.setMinimumWidth(210)
        options_row.addWidget(self.ab_display_mode)
        self.ab_show_tissue_mask = QCheckBox("Sample mask boundary")
        self.ab_show_tissue_mask.setChecked(True)
        self.ab_show_tissue_mask.setVisible(False)
        self.ab_show_tissue_mask.setToolTip("Cyan contour of the intensity mask after erosion; pixels outside have no tissue prediction.")
        options_row.addWidget(self.ab_show_tissue_mask)
        self.ab_save_fat_mask = QPushButton("Save fat mask…")
        self.ab_save_fat_mask.setEnabled(False)
        self.ab_save_fat_mask.hide()
        options_row.addStretch(1)
        controls_layout.addLayout(options_row)
        controls_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        root.addWidget(controls_group, 0)
        self.ab_ambiguity_legend = QLabel(
            FAT_CLASS_LEGEND)
        self.ab_ambiguity_legend.setVisible(False)
        self.ab_ambiguity_legend.setStyleSheet("color:#d9d9d9;")
        self.ab_ambiguity_legend.setWordWrap(True)
        self.ab_ambiguity_legend.setFixedHeight(2*self.ab_ambiguity_legend.fontMetrics().lineSpacing())
        legend_policy = self.ab_ambiguity_legend.sizePolicy()
        legend_policy.setRetainSizeWhenHidden(False)
        self.ab_ambiguity_legend.setSizePolicy(legend_policy)
        root.addWidget(self.ab_ambiguity_legend)
        self.ab_probability_scale = QLabel()
        width, margin, height = 440, 14, 16
        bar_width = width-2*margin
        rgb = np.tile(probability_rgb(np.linspace(0., 1., bar_width))[None], (height,1,1))
        scale = QPixmap(width, 44)
        scale.fill(Qt.transparent)
        painter = QPainter(scale)
        painter.drawImage(margin, 0, QImage(rgb.data, bar_width, height, rgb.strides[0], QImage.Format_RGB888).copy())
        painter.setPen(QColor('#dddddd'))
        painter.setFont(self.font())
        for value in np.linspace(0, 1, 6):
            x = margin+round(value*(bar_width-1))
            text = f'{value:.1f}'
            painter.drawLine(x, height, x, height+4)
            painter.drawText(x-painter.fontMetrics().horizontalAdvance(text)//2, 36, text)
        painter.end()
        self.ab_probability_scale.setPixmap(scale)
        # Both Platt views must have the same viewport: hiding the color scale
        # must not enlarge the image's fit-to-window scale while zoom says 1x.
        self.ab_probability_scale.setFixedHeight(scale.height())
        scale_policy = self.ab_probability_scale.sizePolicy()
        scale_policy.setRetainSizeWhenHidden(False)
        self.ab_probability_scale.setSizePolicy(scale_policy)
        self.ab_probability_scale.setToolTip('P(fat): 0 on the left, 1 on the right; continuous probabilities, no thresholding.')
        self.ab_probability_scale.setVisible(False)
        root.addWidget(self.ab_probability_scale)
        self.ab_decision_status = QLabel("Platt probabilities unavailable — rerun classification with a calibrated model.")
        self.ab_decision_status.setWordWrap(True)
        self.ab_decision_status.setTextFormat(Qt.PlainText)
        self.ab_decision_status.hide()

        # =========================
        # 3) Main area: classes | full-height image | live spectrum
        # =========================
        main = QHBoxLayout()
        main.setSpacing(10)

        # ---- LEFT
        grp = QGroupBox("Saved classes")
        grp_layout = QVBoxLayout(grp)
        grp_layout.setContentsMargins(6, 12, 6, 4)
        grp_layout.setSpacing(6)
        grp.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        grp.setMinimumWidth(135 if embedded else 150)
        grp.setMaximumWidth(175 if embedded else 200)

        self.ab_checks_container = QWidget()
        self.ab_checks_layout = QVBoxLayout(self.ab_checks_container)
        self.ab_checks_layout.setSizeConstraint(QLayout.SetMinimumSize)
        self.ab_checks_layout.setContentsMargins(0, 0, 0, 0)
        self.ab_checks_layout.setSpacing(6)
        self.ab_checks_layout.addStretch(1)

        self.ab_classes_scroll = QScrollArea()
        self.ab_classes_scroll.setWidgetResizable(True)
        self.ab_classes_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.ab_classes_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.ab_classes_scroll.setMinimumHeight(115)
        self.ab_classes_scroll.setWidget(self.ab_checks_container)
        grp_layout.addWidget(self.ab_classes_scroll, 1)

        quick = QVBoxLayout()
        btn_all = QPushButton("All")
        btn_none = QPushButton("None")
        btn_invert = QPushButton("Invert")
        quick.addWidget(btn_all)
        quick.addWidget(btn_none)
        quick.addWidget(btn_invert)
        grp_layout.addLayout(quick)

        main.addWidget(grp, 0)

        image_panel = QVBoxLayout()
        zoom_row = QHBoxLayout()
        self.ab_zoom_out = QPushButton('−')
        self.ab_zoom_in = QPushButton('+')
        self.ab_zoom_fit = QPushButton('Fit image')
        self.ab_zoom_info = QLabel('1.0×')
        for button in (self.ab_zoom_out, self.ab_zoom_in, self.ab_zoom_fit):
            zoom_row.addWidget(button)
        zoom_row.addWidget(self.ab_zoom_info)
        zoom_row.addStretch(1)
        image_panel.addLayout(zoom_row)
        self.ab_view_label = ZoomablePixmapLabel()
        self.ab_view_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ab_view_label.setMinimumSize(200, 200)
        self.ab_view_label.setMouseTracking(True)
        self.ab_view_label.setStyleSheet("QLabel { background:#111; border:1px solid #333; }")
        image_panel.addWidget(self.ab_view_label, 1)
        main.addLayout(image_panel, 4)
        self.ab_zoom_out.clicked.connect(lambda: self.ab_view_label.zoom_by(1/1.25))
        self.ab_zoom_in.clicked.connect(lambda: self.ab_view_label.zoom_by(1.25))
        self.ab_zoom_fit.clicked.connect(self.ab_view_label.reset_view)
        self.ab_view_label.viewChanged.connect(lambda zoom: self.ab_zoom_info.setText(f'{zoom:.1f}×'))

        # ---- RIGHT (CREER right AVANT de l'utiliser)
        right = QVBoxLayout()
        right.setSpacing(6)

        lbl = QLabel("<b>Spectrum · pixel and class reference</b>")
        right.addWidget(lbl, 0)
        help_label = QLabel("Move over the image to explore. Click to hold a point.")
        help_label.setWordWrap(True)
        right.addWidget(help_label)
        spectrum_controls = QHBoxLayout()
        self.ab_spectrum_mode = QComboBox()
        self.ab_spectrum_mode.addItem("Single pixel", 0)
        self.ab_spectrum_mode.addItem("3 × 3 mean", 1)
        self.ab_spectrum_mode.addItem("5 × 5 mean", 2)
        self.ab_spectrum_mode.setToolTip("Single pixel: the measured spectrum at the pointer.\nA window averages neighboring pixels to reduce noise.\nNear a boundary it can mix different materials.\nThe reference always uses the central pixel's class.")
        self.ab_spectrum_normalization = QComboBox()
        self.ab_spectrum_normalization.addItem('Raw spectrum', 'raw')
        self.ab_spectrum_normalization.addItem('Model-normalized spectrum', 'model')
        self.ab_spectrum_normalization.setToolTip('Apply the normalization saved with this classification, on the model spectral bands. Ratio features, scaler and PCA are not wavelength curves.')
        self.ab_resume_hover = QPushButton("Resume hover")
        self.ab_resume_hover.setEnabled(False)
        spectrum_controls.addWidget(self.ab_spectrum_mode)
        spectrum_controls.addWidget(self.ab_resume_hover)
        right.addLayout(spectrum_controls)
        right.addWidget(self.ab_spectrum_normalization)
        self.ab_show_reference = QCheckBox("Class mean ± standard deviation ⓘ")
        self.ab_show_reference.setChecked(True)
        self.ab_show_reference.setToolTip("Dashed line: mean spectrum of the same predicted class\nin this cube. Shading: mean ± one standard deviation\nat each wavelength (variation between pixels).\n\nUp to 10,000 pixels per class are sampled, with a fixed seed.\nDisplay threshold and hidden classes do not change it.\nThis is not a training reference or SVM confidence.\nOutside the band does not mean incorrectly classified.")
        right.addWidget(self.ab_show_reference)

        self.ab_hover_label = QLabel("x=?, y=?")
        self.ab_hover_label.setWordWrap(True)
        self.ab_hover_label.setTextFormat(Qt.PlainText)
        self.cb_ab_fixed_axes = QCheckBox("Fix axes to cube scale")
        self.cb_ab_fixed_axes.setToolTip(
            "Checked: use the same signal scale across the cube to compare pixels.\n"
            "The native signal axis is capped at 2; values above 2 are outside the plot.\n"
            "Unchecked: fit the vertical axis to the displayed spectrum and visible class reference.\n"
            "Only the display scale changes; spectral values are preserved."
        )
        self.cb_ab_fixed_axes.setChecked(False)
        right.addWidget(self.cb_ab_fixed_axes, 0)
        right.addWidget(self.ab_hover_label, 0)

        # --- RIGHT column: live spectrum plot (beau thème sombre) ---
        fig = Figure(figsize=(4, 3), dpi=110)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        # Theme sombre clean
        bg = "#2b2b2b"
        ax_bg = "#2b2b2b"
        fg = (0.92, 0.92, 0.92)
        gridc = (1.0, 1.0, 1.0, 0.12)

        fig.patch.set_facecolor(bg)
        ax.set_facecolor(ax_bg)

        # Style axes
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color((0.45, 0.45, 0.45))
        ax.spines["bottom"].set_color((0.45, 0.45, 0.45))
        ax.tick_params(colors=fg, labelsize=9)
        ax.xaxis.label.set_color(fg)
        ax.yaxis.label.set_color(fg)
        ax.title.set_color(fg)

        # Grille légère
        ax.grid(True, which="major", linewidth=0.7, alpha=0.35, color=gridc)
        ax.set_axisbelow(True)

        empty_spectrum_title = "Hover a pixel" if embedded else "Hover over the image to view a spectrum"
        ax.set_title(empty_spectrum_title, pad=10)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Signal (native cube values)")

        # Ligne principale (créée une fois)
        self._ab_spec_canvas = canvas
        self._ab_spec_ax = ax
        (self._ab_spec_line,) = ax.plot([], [], color="#f3f3f3", linewidth=2.0, label="Selected spectrum")
        (self._ab_ref_line,) = ax.plot([], [], linewidth=1.8, linestyle="--", label="Class mean")
        self._ab_ref_band = None
        (self._ab_spectrum_cursor,) = ax.plot([], [], 'o', color='#ffd166', markersize=6, zorder=10, label='_nolegend_')
        self._ab_spectrum_annotation = ax.annotate('', xy=(0,0), xytext=(12,12),
            textcoords='offset points', color='#f4f4f4', fontsize=9, zorder=11,
            bbox=dict(boxstyle='round,pad=.4',facecolor='#252525',edgecolor='#ffd166',alpha=.95))
        self._ab_spectrum_annotation.set_visible(False)

        def _hide_spectrum_cursor(_event=None):
            if self._ab_spectrum_annotation.get_visible():
                self._ab_spectrum_annotation.set_visible(False)
                self._ab_spectrum_cursor.set_data([],[])
                canvas.draw_idle()

        def _hover_spectrum(event):
            if event.inaxes is not self._ab_spec_ax or event.xdata is None:
                _hide_spectrum_cursor()
                return
            x = np.asarray(self._ab_spec_line.get_xdata(),dtype=float)
            y = np.asarray(self._ab_spec_line.get_ydata(),dtype=float)
            valid = np.flatnonzero(np.isfinite(x) & np.isfinite(y))
            if not len(valid):
                _hide_spectrum_cursor()
                return
            index = valid[np.argmin(np.abs(x[valid]-event.xdata))]
            wavelength, intensity = float(x[index]), float(y[index])
            self._ab_spectrum_cursor.set_data([wavelength],[intensity])
            annotation = self._ab_spectrum_annotation
            annotation.xy = (wavelength,intensity)
            normalized = self.ab_spectrum_normalization.currentData() == 'model'
            context = getattr(self,'_ab_spectrum_context',None)
            mode = context['mode'] if normalized and context else 'raw'
            if mode == 'snv_three_ratios':
                mode = 'SNV (3-ratio model)'
            unit = 'nm' if self._ab_wavelengths is not None else '(band)'
            annotation.set_text(f'{wavelength:g} {unit}\nIntensity: {intensity:.5g} · {mode}')
            xlo,xhi = ax.get_xlim();ylo,yhi = ax.get_ylim()
            left = wavelength > xlo+.65*(xhi-xlo)
            below = intensity > ylo+.7*(yhi-ylo)
            annotation.set_position((-12 if left else 12,-12 if below else 12))
            annotation.set_ha('right' if left else 'left')
            annotation.set_va('top' if below else 'bottom')
            annotation.set_visible(True)
            canvas.draw_idle()

        canvas.mpl_connect('motion_notify_event',_hover_spectrum)
        canvas.mpl_connect('axes_leave_event',_hide_spectrum_cursor)
        canvas.mpl_connect('figure_leave_event',_hide_spectrum_cursor)

        # Texte overlay (infos hover)
        self._ab_spec_info = ax.text(
            0.02, 0.98, "",
            transform=ax.transAxes,
            va="top", ha="left",
            color=fg, fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor=(0, 0, 0, 0.25),
                edgecolor=(1, 1, 1, 0.15),
            )
        )

        fig.tight_layout(pad=1.2)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        canvas.setMinimumHeight(220)
        right.addWidget(canvas, 1)
        self.ab_reference_info = QLabel("Class reference appears when a pixel has a unique assigned class.")
        self.ab_reference_info.setTextFormat(Qt.PlainText)
        self.ab_reference_info.setWordWrap(True)
        right.addWidget(self.ab_reference_info)

        # Keep the spectrum readable without letting it squeeze the overlay image.
        right_widget = QWidget()
        right_widget.setLayout(right)
        if embedded:
            right_widget.setMinimumWidth(400)
            right_widget.setMaximumWidth(500)
        main.addWidget(right_widget, 2)

        # AJOUTER le layout principal dans root (si pas déjà fait plus bas)
        root.addLayout(main, 1)



        # =========================================================
        # Helpers
        # =========================================================
        def _band_to_wl(n_bands: int) -> np.ndarray:
            """Return the wavelengths associated with the loaded cube bands.

            Important: do NOT assume band 0 is 420 nm. The user may have scanned only
            a subset, e.g. 460..720 nm. In that case the cube band order must be taken
            from metadata or from the corrected file names.
            """
            wl = getattr(self, "_ab_wavelengths", None)
            if wl is not None:
                wl = np.asarray(wl, dtype=np.float32).reshape(-1)
                if wl.size == int(n_bands):
                    return wl

            # Last-resort fallback: band indices, not fake wavelengths.
            # This avoids displaying wrong physical wavelengths.
            return np.arange(int(n_bands), dtype=np.float32)

        def _set_thr_label():
            self.ab_meat_label.setText(f"{self.ab_meat_slider.value()/1000.:.1%}")
            self.ab_fat_label.setText(f"{self.ab_fat_slider.value()/1000.:.1%}")

        def _selective_map(native=False):
            source = self._ab_probability_native if self._ab_probability_native is not None else self._ab_probability
            low, high = self.ab_meat_slider.value()/1000., self.ab_fat_slider.value()/1000.
            key = (id(source), source.shape, low, high)
            cache = getattr(self, '_ab_cleanup_cache', None)
            if cache is None or cache[0] != key:
                labels = spatial_fat_membership_labels(source, low, high)
                shown = labels if labels.shape == self._ab_probability.shape else cv2.resize(
                    labels, (self._ab_probability.shape[1], self._ab_probability.shape[0]), interpolation=cv2.INTER_NEAREST)
                cache = (key, labels, shown)
                self._ab_cleanup_cache = cache
            return cache[1 if native else 2]

        def _update_decision_status():
            if self._ab_probability is None:
                return
            labels = _selective_map(native=True)
            self.ab_decision_status.setText(
                f"Platt · fat ≥ {self.ab_fat_slider.value()/10.:g}%: {(labels == 1).sum():,} · "
                f"non-fat ≤ {self.ab_meat_slider.value()/10.:g}%: {(labels == 0).sum():,} · "
                f"unsure: {(labels == 2).sum():,} · excluded: {(labels < 0).sum():,}. "
                "Fat islands ≤5 native pixels → unsure. Preview only; saved maps change only when exported or classification is rerun.")

        def _set_alpha_label():
            self.ab_alpha_label.setText(f"{self.ab_alpha_slider.value()/100.0:.2f}")

        def _infer_display_max(arr: np.ndarray) -> float:
            """Infer a sensible full-scale value for display/stat axes."""
            a = np.asarray(arr)
            if a.size == 0:
                return 255.0
            if a.dtype == np.uint16:
                return 65535.0
            if a.dtype == np.uint8:
                return 255.0

            finite = a[np.isfinite(a)] if np.issubdtype(a.dtype, np.floating) else a
            if finite.size == 0:
                return 255.0

            vmax = float(np.nanmax(finite))
            if vmax <= 1.5:
                return 1.0
            if vmax <= 255.0:
                return 255.0
            return 65535.0

        def _to_display_u8_bgr(img: np.ndarray) -> np.ndarray:
            """Convert gray/BGR images from uint8, uint16, or float to display-safe uint8 BGR."""
            if img is None:
                return None

            arr = np.asarray(img)

            # Ensure BGR 3-channel for display.
            if arr.ndim == 2:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            elif arr.ndim == 3 and arr.shape[2] == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
            elif arr.ndim == 3 and arr.shape[2] >= 3:
                arr = arr[..., :3]
            else:
                raise ValueError(f"Unsupported display image shape: {arr.shape}")

            if arr.dtype == np.uint8:
                return arr.copy()

            arr_f = arr.astype(np.float32, copy=False)
            max_dn = _infer_display_max(arr)

            # For corrected reflectance-like images in 0..1, display as 0..255.
            # For uint16/raw-like images, display using the real 16-bit range.
            out = np.clip(arr_f / max_dn * 255.0, 0, 255).astype(np.uint8)
            return out

        def _np_bgr_to_qpix(img_bgr: np.ndarray) -> QPixmap:
            img_u8 = _to_display_u8_bgr(img_bgr)
            img_rgb = cv2.cvtColor(img_u8, cv2.COLOR_BGR2RGB)
            h, w, ch = img_rgb.shape
            qimg = QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
            return QPixmap.fromImage(qimg)

        def _abundance_to01(a: np.ndarray) -> np.ndarray:
            """Normalize abundance maps to 0..1 regardless of storage scale."""
            arr = np.asarray(a, dtype=np.float32)
            if arr.size == 0:
                return arr
            vmax = float(np.nanmax(arr)) if np.isfinite(arr).any() else 1.0
            if vmax > 4096.0:
                arr = arr / 65535.0
            elif vmax > 1.5:
                arr = arr / 255.0
            return np.clip(arr, 0.0, 1.0)

        def _infer_cube_ymax(cube: np.ndarray | None) -> float:
            """Return the fixed Y-axis upper bound from the values really used by hover spectra.

            IMPORTANT:
            - The Qt overlay image may be converted to uint8 for display.
            - The hover spectrum must stay on the native cube scale.
            - Corrected reflectance cubes are normally displayed on 0..1.

            The previous logic could classify a reflectance cube as 8-bit-like
            when a small number of corrected pixels were > 1.5. That made the
            "Fix axes" checkbox force Y=0..255 and visually crush 0..1 spectra.
            """
            if cube is None:
                return 1.0

            a = np.asarray(cube)
            if a.size == 0:
                return 1.0

            if np.issubdtype(a.dtype, np.floating):
                finite = a[np.isfinite(a)]
            else:
                finite = a.reshape(-1)

            if finite.size == 0:
                return 1.0

            finite = finite.astype(np.float64, copy=False)
            finite = finite[np.isfinite(finite)]
            if finite.size == 0:
                return 1.0

            vmax = float(np.nanmax(finite))
            p95  = float(np.nanpercentile(finite, 95.0))
            p99  = float(np.nanpercentile(finite, 99.0))
            p995 = float(np.nanpercentile(finite, 99.5))
            p999 = float(np.nanpercentile(finite, 99.9))
            frac_gt255 = float(np.mean(finite > 255.0))

            # Corrected reflectance cube in 0..1.
            # Use robust percentiles because reflectance correction can generate
            # rare invalid/outlier pixels > 1 without changing the intended display scale.
            if np.issubdtype(a.dtype, np.floating):
                if p95 <= 1.2 and p99 <= 2.0:
                    return 1.0
                if p995 <= 1.5 and vmax <= 5.0:
                    return 1.0

                # Corrected reflectance stored in percent. Keep one stable,
                # readable scale while leaving room to inspect modest values
                # above the nominal physical range.
                return 120.0

            # Integer / display-like cube that is truly 8-bit.
            if p999 <= 255.0 or (p995 <= 255.0 and frac_gt255 < 0.001):
                return 255.0

            # Genuine 16-bit-like cube.
            if p999 <= 65536.0:
                return 65536.0

            # Fallback for unusual numeric ranges.
            return float(np.ceil(p999))


        def _find_first(folder_p: Path, rel_candidates, glob_pat: str) -> str | None:
            for rel in rel_candidates:
                p = folder_p / rel
                if p.is_file():
                    return str(p)
            for p in folder_p.glob(glob_pat):
                if p.is_file():
                    return str(p)
            return None

        def _load_names_json(fp: str | None) -> list[str] | None:
            if not fp:
                return None
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    names = json.load(f)
                if isinstance(names, list) and all(isinstance(x, str) for x in names):
                    return names
            except Exception:
                pass
            return None

        def _load_cube_from_folder(folder):
            # Result provenance points to the original ENVI pair, never a copy.
            from shared.envi import load_spectral_cube, resolve_envi_header
            self._ab_cube_path = None
            self._ab_loaded_axis = None
            try:
                metadata = json.loads((Path(folder)/'classification_source.json').read_text(encoding='utf-8'))
                path = Path(metadata['cube_path'])
                if not path.is_absolute():
                    path = (PROJECT_ROOT/path) if metadata.get('path_base')=='project' else (Path(folder)/path)
                header = resolve_envi_header(path)
                cube, axis = load_spectral_cube(header)
                self._ab_cube_path = str(header)
                self._ab_loaded_axis = axis
                return cube
            except (OSError, ValueError, KeyError):
                return None

        def _load_wavelengths_from_folder(folder, n_bands):
            axis = getattr(self, '_ab_loaded_axis', None)
            return axis if axis is not None and len(axis)==n_bands else None

        def _selected_indices() -> list[int]:
            out = []
            for idx, name in enumerate(self._ab_names or []):
                cb = self.ab_class_checks.get(name)
                if cb is not None and cb.isChecked():
                    out.append(idx)
            return out

        def _rebuild_class_checkboxes(names: list[str]):
            self.ab_class_checks.clear()
            while self.ab_checks_layout.count():
                item = self.ab_checks_layout.takeAt(0)
                w = item.widget()
                if w is not None:
                    w.deleteLater()

            for nm in names:
                cb = QCheckBox(nm)
                cb.setIcon(_class_swatch_icon(nm))
                cb.setIconSize(QSize(14, 14))
                cb.setChecked(True)
                cb.toggled.connect(lambda _=False: _render())
                self.ab_class_checks[nm] = cb
                self.ab_checks_layout.addWidget(cb)

            self.ab_checks_layout.addStretch(1)

        def _update_mapping_info_for_mouse(img_shape_hw: tuple[int, int], pix: QPixmap):
            H, W = img_shape_hw
            if pix is None or pix.isNull():
                self._ab_last_img_shape = None
                self._ab_last_pix_size = None
                self._ab_last_offsets = None
                return

            lab_w = max(1, self.ab_view_label.width())
            lab_h = max(1, self.ab_view_label.height())

            pix_w = pix.width()
            pix_h = pix.height()
            if pix_w <= 0 or pix_h <= 0:
                self._ab_last_img_shape = None
                self._ab_last_pix_size = None
                self._ab_last_offsets = None
                return

            ox = (lab_w - pix_w) // 2
            oy = (lab_h - pix_h) // 2

            self._ab_last_img_shape = (H, W)
            self._ab_last_pix_size = (pix_w, pix_h)
            self._ab_last_offsets = (ox, oy)

        def _labelpos_to_imgxy(pos) -> tuple[int, int] | None:
            if isinstance(self.ab_view_label, ZoomablePixmapLabel):
                return self.ab_view_label.image_coordinates(pos)
            if self._ab_last_img_shape is None or self._ab_last_pix_size is None or self._ab_last_offsets is None:
                return None
            H, W = self._ab_last_img_shape
            pw, ph = self._ab_last_pix_size
            ox, oy = self._ab_last_offsets

            x0 = pos.x() - ox
            y0 = pos.y() - oy
            if x0 < 0 or y0 < 0 or x0 >= pw or y0 >= ph:
                return None

            x = int(x0 * (W / float(pw)))
            y = int(y0 * (H / float(ph)))
            x = max(0, min(W - 1, x))
            y = max(0, min(H - 1, y))
            return x, y

        def _clear_spectrum():
            _hide_spectrum_cursor()
            self._ab_spec_line.set_data([], [])
            self._ab_ref_line.set_data([], [])
            if self._ab_ref_band is not None:
                self._ab_ref_band.remove()
                self._ab_ref_band = None
            self._ab_spec_info.set_text("")
            legend = self._ab_spec_ax.get_legend()
            if legend is not None:
                legend.remove()

        def _plot_spectrum_at(x: int, y: int):
            _clear_spectrum()
            if self._ab_cube is None:
                self.ab_hover_label.setText("No spectral cube loaded")
                self.ab_reference_info.setText("Class reference unavailable without a spectral cube.")
                self._ab_spec_ax.set_title("No spectral cube loaded")
                self._ab_spec_canvas.draw_idle()
                return
            cube = self._ab_cube
            h, w, bands = cube.shape
            x, y = max(0, min(w - 1, x)), max(0, min(h - 1, y))
            r = self._ab_hover_radius
            patch = cube[max(0,y-r):min(h,y+r+1), max(0,x-r):min(w,x+r+1)]
            context = getattr(self, '_ab_spectrum_context', None)
            normalized = self.ab_spectrum_normalization.currentData() == 'model' and context is not None
            refs = self._ab_references
            wl = _band_to_wl(bands)
            if normalized:
                from shared.spectrum_normalization import normalize_spectra
                indices = context['indices']
                patch = normalize_spectra(patch.reshape(-1,bands)[:,indices],context['mode']).reshape(*patch.shape[:2],len(indices))
                bands = len(indices)
                wl = wl[indices]
                if self._ab_normalized_refs is None:
                    from gui.features.overlay_spectra import class_references
                    _, self._ab_normalized_refs = class_references(cube,self._ab_abund,
                        normalization_mode=context['mode'],band_indices=indices)
                refs = self._ab_normalized_refs
            finite = np.isfinite(patch)
            counts = finite.sum(axis=(0, 1))
            spec = np.divide(np.where(finite, patch, 0).sum(axis=(0, 1), dtype=np.float64),
                             counts, out=np.full(bands, np.nan), where=counts > 0)
            ax = self._ab_spec_ax
            mode = "Pixel" if r == 0 else f"{patch.shape[1]} × {patch.shape[0]} mean"
            self._ab_spec_line.set_data(wl, spec)
            self._ab_spec_line.set_label(mode)
            labels = self._ab_class_map
            k = int(labels[y, x]) if labels is not None else -1
            names = self._ab_names or []
            name = names[k] if 0 <= k < len(names) else f"component {k}" if k >= 0 else "Unclassified" if self._ab_abund is None else "Unassigned / ambiguous"
            state = "Held" if self._ab_pinned else "Live"
            mixed = (r > 0 and labels is not None and
                     np.any(labels[max(0,y-r):min(h,y+r+1), max(0,x-r):min(w,x+r+1)] != k))
            self.ab_hover_label.setText(f"{state} · x={x}, y={y} · {name} · {mode}" +
                                       (" · mixed-class window" if mixed else ""))
            if self._ab_decision is not None:
                decision = self._ab_decision
                gap = float(decision['gap'][y, x])
                if np.isfinite(gap):
                    favorite, runner = int(decision['favorite'][y, x]), int(decision['runner'][y, x])
                    detail = (f"\nRaw score favorite: {names[favorite]} · alternative: {names[runner]} · gap {gap:.3f}")
                    if favorite != k:
                        detail += " · raw ranking differs from saved class"
                    self.ab_hover_label.setText(self.ab_hover_label.text() + detail)
                else:
                    self.ab_hover_label.setText(self.ab_hover_label.text() + "\nDecision score unavailable for this pixel")
            if self._ab_probability is not None:
                p = self._ab_probability[y, x]
                code = _selective_map()[y,x]
                label = {-1: 'excluded', 0: 'non-fat', 1: 'fat', 2: 'unsure'}[int(code)]
                self.ab_hover_label.setText(self.ab_hover_label.text() + f"\nPlatt P(fat)={p:.3f} · displayed decision: {label}")
            mode_label = 'SNV (3-ratio model)' if normalized and context['mode'] == 'snv_three_ratios' else context['mode'] if normalized else ''
            ax.set_title(('Normalized · ' + mode_label) if normalized else 'Selected spectrum' if self._ab_abund is None else 'Selected spectrum and class reference', fontsize=11)
            ax.set_ylabel('Normalized spectral value' if normalized else 'Signal (native cube values)')
            ax.set_xlabel("Wavelength (nm)" if self._ab_wavelengths is not None else "Band index")
            bounds = [spec]
            ref = refs.get(k)
            if ref is not None:
                self.ab_reference_info.setText(
                    f"Reference: {name} in this cube\n"
                    f"{ref['count']:,} finite sampled spectra / {ref['total']:,} assigned pixels.\n"
                    "Shading shows variation, not classification confidence.")
                if self.ab_show_reference.isChecked():
                    bgr = _color_for(name)
                    # Keep background and dark classes readable on the dark plot.
                    color = tuple(v / 255 for v in bgr[::-1]) if max(bgr) >= 100 else (.65, .65, .65)
                    self._ab_ref_line.set_data(wl, ref['mean'])
                    self._ab_ref_line.set_color(color)
                    low, high = ref['mean'] - ref['std'], ref['mean'] + ref['std']
                    self._ab_ref_band = ax.fill_between(wl, low, high, color=color, alpha=.2,
                                                        label="Class ±1 std")
                    bounds.extend([low, high])
            else:
                self.ab_reference_info.setText("No classification applied." if self._ab_abund is None else "No usable class reference at this point.\nZero, tied or invalid class values are left unassigned.")
            if len(wl) > 1:
                ax.set_xlim(float(wl[0]), float(wl[-1]))
            else:
                ax.set_xlim(float(wl[0]) - .5, float(wl[0]) + .5)
            values = np.concatenate(bounds)
            values = values[np.isfinite(values)]
            if values.size:
                lo, hi = float(values.min()), float(values.max())
                if self.cb_ab_fixed_axes.isChecked():
                    if normalized:
                        all_bounds = np.concatenate([np.r_[ref['mean']-ref['std'],ref['mean']+ref['std']] for ref in refs.values()]) if refs else values
                        lo, hi = float(np.nanmin(all_bounds)), float(np.nanmax(all_bounds))
                    else:
                        lo, hi = self._ab_spectrum_limits
                        hi = min(hi, 2.0)
                # Cap before padding so extreme values cannot inflate the padding.
                if lo >= hi:
                    lo = min(0.0, hi - 1.0)
                pad = max((hi-lo)*.06, abs(hi)*.01, 1e-6)
                upper = hi + pad
                if self.cb_ab_fixed_axes.isChecked() and not normalized:
                    upper = min(upper, 2.0)
                ax.set_ylim(lo-pad, upper)
            handles = [self._ab_spec_line]
            if self._ab_ref_band is not None:
                handles.extend([self._ab_ref_line, self._ab_ref_band])
            legend = ax.legend(handles=handles, loc="best", fontsize=8,
                               facecolor="#2b2b2b", edgecolor="#555555")
            for text in legend.get_texts():
                text.set_color("#f0f0f0")
            self._ab_spec_canvas.draw_idle()

        def _refresh_spectrum():
            if self._ab_hover_xy is not None:
                _plot_spectrum_at(*self._ab_hover_xy)

        def _resume_hover():
            self._ab_pinned = False
            self.ab_resume_hover.setEnabled(False)
            _refresh_spectrum()

        def _change_spectrum_mode():
            self._ab_hover_radius = int(self.ab_spectrum_mode.currentData())
            _refresh_spectrum()
            _render()

        def _apply_decision_display(img_bgr: np.ndarray) -> np.ndarray:
            mode = self.ab_display_mode.currentData()
            if mode in ('fat_selective', 'fat_probability') and self._ab_probability is not None:
                return platt_overlay(self._ab_base_bgr, self._ab_probability, mode,
                    self.ab_meat_slider.value()/1000., self.ab_fat_slider.value()/1000., self.ab_alpha_slider.value()/100.,
                    decision_labels=_selective_map())
            return img_bgr

        def _draw_hover_box(img_bgr: np.ndarray) -> np.ndarray:
            img_bgr = _apply_decision_display(img_bgr)
            if self._ab_tissue_mask is not None and self.ab_show_tissue_mask.isChecked():
                from shared.tissue_segmentation import draw_dashed_boundary
                img_bgr = draw_dashed_boundary(img_bgr, self._ab_tissue_mask)
            img_bgr = self._draw_data_selection(img_bgr)
            if not getattr(self, "_ab_show_hover_box", True):
                return img_bgr
            if self._ab_hover_xy is None:
                return img_bgr

            x, y = self._ab_hover_xy
            r = int(getattr(self, "_ab_hover_radius", 2))
            H, W = img_bgr.shape[:2]

            x0 = max(0, x - r)
            x1 = min(W - 1, x + r)
            y0 = max(0, y - r)
            y1 = min(H - 1, y + r)

            out = img_bgr.copy()
            col = tuple(int(v) for v in getattr(self, "_ab_hover_color", (0, 255, 0)))
            cv2.rectangle(out, (x0, y0), (x1, y1), col, thickness=1)
            return out

        # throttle mouse redraw
        self._ab_last_hover_xy_drawn = None

        def _render():
            if self._ab_base_bgr is None:
                self.ab_view_label.clear()
                self.ab_view_label.setText("No image")
                return

            base = self._ab_base_bgr.copy()
            abund = self._ab_abund

            if self.ab_display_mode.currentData() in ('fat_selective', 'fat_probability') and self._ab_probability is not None:
                pix = _np_bgr_to_qpix(_draw_hover_box(base))
                self.ab_view_label.setText('')
                self.ab_view_label.setPixmap(pix)
                _update_mapping_info_for_mouse(base.shape[:2], self.ab_view_label.pixmap())
                return

            # ---- no abundances: show base only
            if abund is None or getattr(abund, "ndim", 0) != 3:
                out_u8 = _draw_hover_box(base)
                pix = _np_bgr_to_qpix(out_u8)
                self.ab_view_label.setText("")
                self.ab_view_label.setPixmap(pix)
                _update_mapping_info_for_mouse(base.shape[:2], self.ab_view_label.pixmap())
                return

            H, W = base.shape[:2]
            if abund.shape[0] != H or abund.shape[1] != W:
                C = abund.shape[2]
                abund_rs = np.zeros((H, W, C), np.float32)
                for k in range(C):
                    abund_rs[..., k] = cv2.resize(abund[..., k].astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
                abund = abund_rs

            thr = 0.5  # Class display is independent of the decision-score threshold.
            alpha = self.ab_alpha_slider.value() / 100.0
            base_f = base.astype(np.float32)

            idxs = _selected_indices()
            if not idxs:
                out_u8 = _draw_hover_box(base)
                pix = _np_bgr_to_qpix(out_u8)
                self.ab_view_label.setText("")
                self.ab_view_label.setPixmap(pix)
                _update_mapping_info_for_mouse((H, W), self.ab_view_label.pixmap())
                return

            out = base_f.copy()
            for k in idxs:
                a = _abundance_to01(abund[..., k])
                nz = a >= thr
                if not np.any(nz):
                    continue

                cls_name = (self._ab_names[k] if self._ab_names and k < len(self._ab_names) else f"component {k}")
                col = np.array(_color_for(cls_name), np.float32)
                overlay = np.zeros_like(out)
                overlay[..., 0] = col[0]
                overlay[..., 1] = col[1]
                overlay[..., 2] = col[2]
                out[nz] = (1.0 - alpha) * out[nz] + alpha * overlay[nz]

            out_u8 = np.clip(out, 0, 255).astype(np.uint8)
            out_u8 = _draw_hover_box(out_u8)


            pix = _np_bgr_to_qpix(out_u8)
            if pix.isNull():
                self.ab_view_label.clear()
                self.ab_view_label.setText("No image")
                return

            self.ab_view_label.setText("")
            self.ab_view_label.setPixmap(pix)

            # Update mapping for mouse hover
            _update_mapping_info_for_mouse((H, W), self.ab_view_label.pixmap())

        def _reload_folder(folder: str, show_warnings: bool = True):
            self._ab_tissue_mask = None
            self.ab_show_tissue_mask.setVisible(False)
            self._ab_decision = None
            self._ab_probability = self._ab_probability_native = None
            self._ab_spectrum_context = None
            self._ab_normalized_refs = None
            self._ab_cleanup_cache = None
            self.ab_meat_slider.setEnabled(False)
            self.ab_fat_slider.setEnabled(False)
            self.ab_save_fat_mask.setEnabled(False)
            self.ab_ambiguity_legend.setVisible(False)
            self.ab_probability_scale.setVisible(False)
            for index in range(1, self.ab_display_mode.count()):
                self.ab_display_mode.model().item(index).setEnabled(False)
            self.ab_decision_status.setText("Platt probabilities unavailable — rerun classification with a calibrated model.")
            self._ab_pinned = False
            self.ab_resume_hover.setEnabled(False)
            self._ab_hover_xy = None
            self._ab_class_map = None
            self._ab_references = {}
            _clear_spectrum()
            self.ab_hover_label.setText("Move over the image to inspect a pixel")
            self.ab_reference_info.setText("No class reference selected")
            self._ab_spec_ax.set_title(empty_spectrum_title)
            self._ab_spec_canvas.draw_idle()
            folder = (folder or "").strip()
            if not folder or not Path(folder).is_dir():
                self.ab_folder_label.setText("Invalid folder")
                self._ab_base_bgr = None
                self._ab_abund = None
                self._ab_names = None
                self._ab_colors_bgr = None
                self._ab_cube = None
                self._ab_wavelengths = None
                self._ab_cube_ymax = 1.0
                self._ab_cube_dtype = "n/a"
                self._ab_cube_vmax = 0.0
                _render()
                return

            folder_p = Path(folder)

            if (folder_p / 'classification').is_dir():
                folder_p = folder_p / 'classification'
                folder = str(folder_p)

            tissue_file = folder_p / 'tissue_mask.hdr'
            if tissue_file.is_file():
                try:
                    from shared.envi import read_envi
                    tissue, _, _ = read_envi(tissue_file, require_wavelengths=False)
                    self._ab_tissue_mask = tissue[:, :, 0].astype(bool)
                    tissue._mmap.close()
                    self.ab_show_tissue_mask.setVisible(True)
                except (OSError, ValueError):
                    pass

            base_fp = _find_first(folder_p, ["base_image_bgr.png", "cube/base_image_bgr.png"], "**/base_image_bgr.png")
            if not base_fp:
                self.ab_folder_label.setText(f"{folder_p.name} · base image missing")
                if show_warnings:
                    QMessageBox.warning(self, "Abundance overlay", "base_image_bgr.png not found.")
                self._ab_base_bgr = None
                _render()
                return

            base_raw = read_image(base_fp, cv2.IMREAD_UNCHANGED)
            if base_raw is None:
                QMessageBox.warning(self, "Abundance overlay", f"Failed to read base image:\n{base_fp}")
                self._ab_base_bgr = None
                _render()
                return

            try:
                base = _to_display_u8_bgr(base_raw)
            except Exception as e:
                QMessageBox.warning(self, "Abundance overlay", f"Failed to convert base image for display:\n{base_fp}\n\n{e}")
                self._ab_base_bgr = None
                _render()
                return

            abund_fp = _find_first(folder_p, ["abundances.hdr", "cube/abundances.hdr"], "**/abundances.hdr")
            if not abund_fp:
                self.ab_folder_label.setText(f"{folder_p.name} · classification output missing")
                if show_warnings:
                    QMessageBox.warning(self, "Abundance overlay", "abundances.hdr not found.")
                self._ab_base_bgr = base
                self._ab_abund = None
                self._ab_names = None
                self._ab_colors_bgr = None
                self._ab_cube = None
                self._ab_wavelengths = None
                self._ab_cube_ymax = 1.0
                self._ab_cube_dtype = "n/a"
                self._ab_cube_vmax = 0.0
                _render()
                return

            try:
                from shared.envi import read_envi
                raster, _, _ = read_envi(abund_fp, require_wavelengths=False)
                # Class maps are small and copied so a worker can replace its own outputs on Windows.
                abund = np.array(raster)
                raster._mmap.close()
                abund = np.asarray(abund)
            except Exception as e:
                QMessageBox.warning(self, "Abundance overlay", f"Failed to load abundances.hdr:\n{abund_fp}\n\n{e}")
                self._ab_base_bgr = base
                self._ab_abund = None
                self._ab_names = None
                self._ab_colors_bgr = None
                self._ab_cube = None
                self._ab_wavelengths = None
                self._ab_cube_ymax = 1.0
                self._ab_cube_dtype = "n/a"
                self._ab_cube_vmax = 0.0
                _render()
                return

            if abund.ndim != 3:
                QMessageBox.warning(self, "Abundance overlay", f"abundances.hdr must be (H,W,C). Got shape={abund.shape}")
                self._ab_base_bgr = base
                self._ab_abund = None
                self._ab_names = None
                self._ab_cube = None
                self._ab_wavelengths = None
                self._ab_cube_ymax = 1.0
                self._ab_cube_dtype = "n/a"
                self._ab_cube_vmax = 0.0
                _render()
                return

            Hb, Wb = base.shape[:2]
            score_source_shape = abund.shape[:2]
            if abund.shape[0] != Hb or abund.shape[1] != Wb:
                C = abund.shape[2]
                abund_rs = np.zeros((Hb, Wb, C), np.float32)
                for k in range(C):
                    abund_rs[..., k] = cv2.resize(abund[..., k].astype(np.float32), (Wb, Hb), interpolation=cv2.INTER_NEAREST)
                abund = abund_rs

            names_fp = _find_first(folder_p, ["endmember_names.json", "cube/endmember_names.json"], "**/endmember_names.json")
            names = _load_names_json(names_fp)
            if not names:
                names = [f"component {k}" for k in range(abund.shape[2])]
            else:
                names = names[:abund.shape[2]]

            # Use the exact palette saved by the classifier so this overlay and
            # mixture_side_by_side.png always identify classes with the same colors.
            colors_by_name = {}
            spectra_fp = _find_first(
                folder_p,
                ["spectra_data.npz", "cube/spectra_data.npz"],
                "**/spectra_data.npz",
            )
            if spectra_fp:
                try:
                    with np.load(spectra_fp, allow_pickle=True) as spectra:
                        raw_colors = spectra.get("class_colors_bgr")
                        raw_names = spectra.get("class_names", spectra.get("classes"))
                        if raw_colors is not None and raw_names is not None:
                            raw_colors = np.asarray(raw_colors, dtype=np.uint8)
                            raw_names = [str(value) for value in np.asarray(raw_names).tolist()]
                            if raw_colors.ndim == 2 and raw_colors.shape[1] == 3:
                                colors_by_name = {
                                    name.strip().lower(): raw_colors[index]
                                    for index, name in enumerate(raw_names)
                                    if index < len(raw_colors)
                                }
                except (OSError, ValueError, KeyError):
                    colors_by_name = {}

            self._ab_colors_bgr = np.asarray(
                [
                    colors_by_name.get(
                        name.strip().lower(),
                        np.asarray(_fallback_color(name), dtype=np.uint8),
                    )
                    for name in names
                ],
                dtype=np.uint8,
            )

            cube = _load_cube_from_folder(folder)
            cube_wavelengths = _load_wavelengths_from_folder(folder, cube.shape[2]) if cube is not None else None
            cube_ymax = _infer_cube_ymax(cube)
            cube_dtype = str(np.asarray(cube).dtype) if cube is not None else "n/a"
            if cube is not None and np.asarray(cube).size > 0:
                cube_arr_for_info = np.asarray(cube)
                if np.issubdtype(cube_arr_for_info.dtype, np.floating):
                    finite_info = cube_arr_for_info[np.isfinite(cube_arr_for_info)]
                    cube_vmax = float(np.nanmax(finite_info)) if finite_info.size else 0.0
                else:
                    cube_vmax = float(np.max(cube_arr_for_info))
            else:
                cube_vmax = 0.0

            if cube is not None:
                from gui.features.overlay_spectra import spectrum_normalization_context
                self._ab_spectrum_context = spectrum_normalization_context(folder_p,cube_wavelengths)
                self.ab_spectrum_normalization.model().item(1).setEnabled(self._ab_spectrum_context is not None)
                if self._ab_spectrum_context is not None:
                    mode_label = 'SNV (3-ratio model)' if self._ab_spectrum_context['mode'] == 'snv_three_ratios' else self._ab_spectrum_context['mode']
                    self.ab_spectrum_normalization.setItemText(1, 'Model-normalized spectrum · ' + mode_label)
                    self.ab_spectrum_normalization.setToolTip('Normalization saved with this classification, on model bands. Ratio, scaler and PCA are separate features, not wavelength curves.')
                if self._ab_spectrum_context is None:
                    self.ab_spectrum_normalization.setCurrentIndex(0)
                    self.ab_spectrum_normalization.setToolTip('Normalization metadata unavailable. Reclassify to save preprocessing metadata with the results.')
                if cube.shape[0] != Hb or cube.shape[1] != Wb:
                    Hc, Wc, B = cube.shape
                    # Resizing for hover alignment may require float32, but the y-axis
                    # was already inferred from the original native cube above.
                    cube_rs = np.zeros((Hb, Wb, B), np.float32)
                    for b in range(B):
                        cube_rs[..., b] = cv2.resize(
                            cube[..., b].astype(np.float32, copy=False),
                            (Wb, Hb),
                            interpolation=cv2.INTER_NEAREST
                        )
                    cube = cube_rs

            self._ab_cube = cube
            self._ab_wavelengths = cube_wavelengths
            self._ab_cube_ymax = cube_ymax
            self._ab_cube_dtype = cube_dtype
            self._ab_cube_vmax = cube_vmax

            self._ab_base_bgr = base
            self._ab_abund = abund.astype(np.float32, copy=False)
            self._ab_names = names
            score_path = Path(abund_fp).with_name("decision_scores.npz")
            if score_path.is_file():
                try:
                    self._ab_decision = load_decision_scores(
                        score_path, abund_fp, names, score_source_shape, (Hb, Wb))
                except (OSError, ValueError, KeyError) as exc:
                    self._ab_decision = None
            if cube is not None:
                self._ab_class_map, self._ab_references = class_references(cube, self._ab_abund)
                lower, upper = 0., max(cube_ymax, cube_vmax)
                for reference in self._ab_references.values():
                    lower = min(lower, float(np.min(reference['mean'] - reference['std'])))
                    upper = max(upper, float(np.max(reference['mean'] + reference['std'])))
                finite_min = float(np.nanmin(cube)) if np.isfinite(cube).any() else 0.
                self._ab_spectrum_limits = (min(lower, finite_min), upper)

            self.ab_folder_label.setText(f"{folder_p.parent.name} / classification" if folder_p.name == 'classification' else folder_p.name)
            self.ab_dir_input.setText(folder)
            self._ab_last_folder = folder

            _rebuild_class_checkboxes(self._ab_names)
            if cube is not None and cube_wavelengths is not None:
                self._set_cube_source(cube, cube_wavelengths, getattr(self, '_ab_cube_path', ''))
            else:
                self._data_cube = None
                self._data_pixels = []
                self._data_rois = []
                self._data_selection_mask = None
                self._data_draft = []
                self._data_drag_end = None
                self._update_data_selection_info()
                self.data_info.setText('Saved image loaded; spectral cube unavailable.')
            _render()

        def _browse_dir():
            start_dir = ""
            if isinstance(getattr(self, "corrected_root_dir", ""), str):
                p = Path(self.corrected_root_dir)
                if p.is_dir():
                    start_dir = str(p)
            if not start_dir and Path("corrected").is_dir():
                start_dir = str(Path("corrected").resolve())
            if not start_dir:
                start_dir = str(Path.cwd())

            folder = QFileDialog.getExistingDirectory(self, "Select folder inside corrected/", start_dir)
            if not folder:
                return

            pf = Path(folder)
            if pf.parent.name.lower() == "corrected":
                self.corrected_root_dir = str(pf.parent)

            _reload_folder(folder)

        def _on_thr_changed(which, _v):
            if self.ab_meat_slider.value() >= self.ab_fat_slider.value():
                slider = self.ab_meat_slider if which == 'meat' else self.ab_fat_slider
                value = self.ab_fat_slider.value()-1 if which == 'meat' else self.ab_meat_slider.value()+1
                slider.blockSignals(True)
                slider.setValue(value)
                slider.blockSignals(False)
            _set_thr_label()
            _update_decision_status()
            _refresh_spectrum()
            _render()

        def _save_fat_mask():
            if self._ab_probability_native is None:
                return
            destination, _ = QFileDialog.getSaveFileName(self, 'Save fat mask at displayed thresholds',
                str(Path(self.ab_dir_input.text())/'mask_fat_selected.png'), 'PNG (*.png)')
            if destination:
                labels = _selective_map(native=True)
                if not write_image(destination, (labels == 1).astype(np.uint8)*255):
                    QMessageBox.warning(self, 'Save fat mask', 'Could not save the selected mask.')

        def _on_alpha_changed(_v: int):
            _set_alpha_label()
            _render()

        def _on_display_mode():
            mode = self.ab_display_mode.currentData()
            platt = mode in ('fat_selective', 'fat_probability')
            self.ab_ambiguity_legend.setVisible(platt and self._ab_probability is not None)
            self.ab_probability_scale.setVisible(mode == 'fat_probability' and self._ab_probability is not None)
            self.ab_ambiguity_legend.setText(
                'Platt heatmap: continuous P(fat), 0 to 1 (Turbo). Colours represent probabilities, not tissue classes.'
                if mode == 'fat_probability' else
                FAT_CLASS_LEGEND + ' Outside the mask remains excluded.')
            self.ab_checks_container.setEnabled(not platt)
            for button in (btn_all, btn_none, btn_invert):
                button.setEnabled(not platt)
            _update_decision_status()
            _render()

        def _set_all(state: bool):
            for cb in self.ab_class_checks.values():
                cb.blockSignals(True)
                cb.setChecked(state)
                cb.blockSignals(False)
            _render()

        def _invert():
            for cb in self.ab_class_checks.values():
                cb.blockSignals(True)
                cb.setChecked(not cb.isChecked())
                cb.blockSignals(False)
            _render()

        # =========================================================
        # Mouse hover filter (live spectrum) + throttle
        # =========================================================
        class _HoverFilter(QObject):
            def eventFilter(_, obj, event):
                if obj is self.ab_view_label:
                    xy = _labelpos_to_imgxy(event.pos()) if event.type() in (
                        QEvent.MouseButtonPress, QEvent.MouseButtonRelease,
                        QEvent.MouseButtonDblClick, QEvent.MouseMove, QEvent.Wheel) else None
                    if event.type() == QEvent.MouseButtonPress:
                        obj.setFocus()
                    if self._handle_data_selection_event(event, xy):
                        return True
                    if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                        xy = _labelpos_to_imgxy(event.pos())
                        if xy is not None:
                            self._ab_pinned = True
                            self.ab_resume_hover.setEnabled(True)
                            self._ab_hover_xy = xy
                            self._record_data_pixel(*xy)
                            _plot_spectrum_at(*xy)
                            _render()
                    elif event.type() == QEvent.MouseMove and not self._ab_pinned:
                        if event.buttons() & (Qt.RightButton | Qt.MiddleButton):
                            return False
                        xy = _labelpos_to_imgxy(event.pos())
                        if xy is not None:
                            if self._ab_hover_xy != xy:
                                self._ab_hover_xy = xy
                                _plot_spectrum_at(xy[0], xy[1])
                                _render()
                    elif event.type() == QEvent.Leave:
                        # Keep the last point visible while reading controls.
                        pass
                return False

        self._ab_hover_filter = _HoverFilter(self.ab_view_label)
        self.ab_view_label.setFocusPolicy(Qt.StrongFocus)
        self.ab_view_label.installEventFilter(self._ab_hover_filter)

        # =========================================================
        # Wire signals
        # =========================================================
        btn_browse.clicked.connect(_browse_dir)
        self.ab_resume_hover.clicked.connect(_resume_hover)
        self.ab_spectrum_mode.currentIndexChanged.connect(_change_spectrum_mode)
        self.ab_spectrum_normalization.currentIndexChanged.connect(_refresh_spectrum)
        self.ab_show_reference.toggled.connect(_refresh_spectrum)
        self.ab_show_tissue_mask.toggled.connect(lambda _=False: _render())
        self.ab_display_mode.currentIndexChanged.connect(lambda _=0: _on_display_mode())
        self.cb_ab_fixed_axes.toggled.connect(_refresh_spectrum)
        btn_refresh.clicked.connect(lambda: _reload_folder(self.ab_dir_input.text().strip()))
        self.ab_dir_input.returnPressed.connect(lambda: _reload_folder(self.ab_dir_input.text().strip()))

        self.ab_meat_slider.valueChanged.connect(lambda value: _on_thr_changed('meat', value))
        self.ab_fat_slider.valueChanged.connect(lambda value: _on_thr_changed('fat', value))
        self.ab_save_fat_mask.clicked.connect(_save_fat_mask)
        self.ab_alpha_slider.valueChanged.connect(_on_alpha_changed)

        btn_all.clicked.connect(lambda: _set_all(True))
        btn_none.clicked.connect(lambda: _set_all(False))
        btn_invert.clicked.connect(_invert)

        def _show_unclassified_cube(cube, wavelengths, path):
            self._ab_cube = cube
            self._ab_cube_path = path
            self._ab_wavelengths = wavelengths
            self._ab_abund = self._ab_names = self._ab_class_map = None
            self._ab_colors_bgr = self._ab_tissue_mask = None
            self._ab_decision = self._ab_probability = self._ab_probability_native = None
            self._ab_references = {}
            self._ab_spectrum_context = self._ab_normalized_refs = None
            self._ab_pinned = False
            self._ab_hover_xy = None
            self.ab_resume_hover.setEnabled(False)
            self.ab_spectrum_normalization.setCurrentIndex(0)
            self.ab_spectrum_normalization.model().item(1).setEnabled(False)
            self.ab_show_tissue_mask.setVisible(False)
            self.ab_ambiguity_legend.hide()
            self.ab_probability_scale.hide()
            self._ab_cube_dtype = str(cube.dtype)
            lower, upper = np.inf, -np.inf
            for row in range(0, cube.shape[0], 32):
                chunk = np.asarray(cube[row:row+32])
                finite = chunk[np.isfinite(chunk)]
                if finite.size:
                    lower = min(lower, float(finite.min()))
                    upper = max(upper, float(finite.max()))
            if not np.isfinite(lower): lower, upper = 0., 1.
            self._ab_spectrum_limits = (lower, upper)
            self._ab_cube_ymax = upper
            self._ab_cube_vmax = upper
            _rebuild_class_checkboxes([])
            _clear_spectrum()
            self._ab_spec_ax.set_title('Hover a pixel')
            self._ab_spec_canvas.draw_idle()
            self.ab_hover_label.setText('Move over the image to inspect a pixel')
            self.ab_reference_info.setText('No classification applied')
            self.ab_folder_label.setText(Path(path).name + ' · unclassified cube')
            self.ab_view_label.reset_view()

        self._show_unclassified_cube = _show_unclassified_cube
        self._render_spectral_image = _render
        self._plot_pixel_spectrum = _plot_spectrum_at
        # Expose the loader so post-processing can update its embedded explorer.
        self._reload_abundance_overlay_folder = _reload_folder

        # init
        _set_thr_label()
        _set_alpha_label()

        if isinstance(self._ab_last_folder, str) and self._ab_last_folder and Path(self._ab_last_folder).is_dir():
            self.ab_dir_input.setText(self._ab_last_folder)
            _reload_folder(self._ab_last_folder)
        else:
            _render()

        return tab
