# Rôle : Construire l’onglet de post-traitement et lancer les workers du pipeline.

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

import cv2
from shared.image_io import read_image, write_image
import joblib
import numpy as np
import pyqtgraph as pg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import QEvent, QObject, QProcess, Qt, QTimer
from PyQt5.QtGui import QColor, QFont, QImage, QIntValidator, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QColorDialog, QComboBox, QFileDialog, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLayout, QLineEdit, QMessageBox,
    QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy, QSlider, QTabWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QSpinBox, QDoubleSpinBox, QDialog,
)

from shared.process_io import utf8_decoder
from shared.paths import PROJECT_ROOT
from shared.svm_transforms import NORMALIZATION_OPTIONS

from gui.widgets import ResizablePixmapLabel, SpectraInteractive, color_for
from gui.features.postprocessing_models import PostprocessingModelsMixin


class PostProcessingMixin(PostprocessingModelsMixin):
    def _make_postproc_tab(self) -> QWidget:
        tab = QWidget()
        from PyQt5.QtWidgets import QHBoxLayout
        main = QHBoxLayout(tab)

        # --- INITIALISATION DES VARIABLES POSTPROCESSING ---
        self.pp_masks_paths = {}      # <-- ajout essentiel
        self.pp_mask_checks = {}      # checkboxes des classes
        self.pp_selected = set()      # classes actuellement cochées

        # -------- Left: selectors + HSL + log --------
        left = QVBoxLayout()
        self.btn_load_pp = QPushButton("Open saved results...")
        self.btn_run_pp = QPushButton("Classify ENVI cube")

        outer_sel = QVBoxLayout()
        outer_sel.setSpacing(10)
        source_box = QGroupBox("1. Input data")
        source_layout = QVBoxLayout(source_box)
        self.combo_pp_source = QComboBox()
        self.combo_pp_source.addItem("Spectral cube — explore or classify", "cube")

        source_layout.addWidget(self.combo_pp_source)

        self.pp_cube_group = QGroupBox("Spectral cube")
        corrected = QVBoxLayout(self.pp_cube_group)
        cube_header = QHBoxLayout()
        cube_header.addWidget(QLabel("Cube file"), 1)
        self.btn_refresh_pp_cubes = QPushButton("Refresh")
        self.btn_refresh_pp_cubes.setToolTip("Refresh corrected cubes; preserves the selected file, including external files.")
        cube_header.addWidget(self.btn_refresh_pp_cubes)
        corrected.addLayout(cube_header)
        self.combo_pp_cube = QComboBox()
        self.combo_pp_cube.setMinimumContentsLength(12)
        self.combo_pp_cube.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        corrected.addWidget(self.combo_pp_cube)
        self.pp_cube_path_label = QLabel("No cube selected")
        self.pp_cube_path_label.setWordWrap(True)
        self.pp_cube_path_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        corrected.addWidget(self.pp_cube_path_label)
        self.btn_browse_pp_cube = QPushButton("Browse cube...")
        corrected.addWidget(self.btn_browse_pp_cube)
        self.btn_view_cube = QPushButton("Open cube without classification")
        self.btn_view_cube.clicked.connect(self._view_selected_cube)
        corrected.addWidget(self.btn_view_cube)
        note = QLabel("Native resolution. No white/dark or shading correction is applied.")
        note.setWordWrap(True)
        corrected.addWidget(note)
        source_layout.addWidget(self.pp_cube_group)
        outer_sel.addWidget(source_box)

        model_box = QGroupBox("2. SVM classification")
        sel = QVBoxLayout(model_box)
        self.combo_pp_model = QComboBox()
        self.combo_pp_model.setToolTip(
            "Base SVM model family. The wavelength-specific cached model is chosen automatically."
        )
        self.btn_refresh_pp_models = QPushButton("Refresh")
        self.btn_refresh_pp_models.setToolTip("Refresh only the SVM model families in this list.")
        model_row = QHBoxLayout()
        model_row.addWidget(self.combo_pp_model, 1)
        model_row.addWidget(self.btn_refresh_pp_models)
        sel.addWidget(QLabel("Project / ROI dataset"))
        sel.addLayout(model_row)

        self.combo_pp_classification = QComboBox()
        self.combo_pp_classification.addItem("Linear", "linear")
        self.combo_pp_classification.addItem("RBF", "rbf")
        self.combo_pp_classification.setToolTip(
            "Hybrid uses the 512x512 pass for shading and a conservative biological mask. "
            "It expands that mask, runs Linear again on its native-resolution pixels, then "
            "sends only uncertain tissue pixels to RBF."
        )
        initial_mode = str(
            getattr(self, "selected_pp_classification_mode", "linear")
        ).lower()
        initial_index = self.combo_pp_classification.findData(initial_mode)
        self.combo_pp_classification.setCurrentIndex(max(0, initial_index))

        def _update_pp_classification_mode(_index: int) -> None:
            mode = str(self.combo_pp_classification.currentData() or "linear")
            self.selected_pp_classification_mode = mode
            self.selected_pp_svm_type = "rbf" if mode in {"rbf", "hybrid"} else "linear"

        self.combo_pp_classification.currentIndexChanged.connect(_update_pp_classification_mode)
        _update_pp_classification_mode(self.combo_pp_classification.currentIndex())
        sel.addWidget(QLabel("Classification method"))
        sel.addWidget(self.combo_pp_classification)

        self.combo_pp_normalization = QComboBox()
        for label, mode in NORMALIZATION_OPTIONS:
            self.combo_pp_normalization.addItem(label, mode)
        self.combo_pp_normalization.setToolTip(
            "Applied to training and image spectra. SNV without ratio uses only normalized bands. "
            "SNV + 3 ratios appends raw R620/R580, R560/R540 and R700/R620. Other non-none modes append R620/R580. "
            "Changing this setting only selects preprocessing. A compatible saved model is reused at classification; "
            "training starts only if none matches. ROI datasets and libraries are not modified."
        )
        sel.addWidget(QLabel("Normalization"))
        sel.addWidget(self.combo_pp_normalization)
        self.combo_pp_normalization.activated.connect(self._on_pp_normalization_changed)
        self.combo_pp_model.activated.connect(self._on_pp_profile_changed)

        for combo in (self.combo_pp_cube, self.combo_pp_model, self.combo_pp_classification,
                      self.combo_pp_normalization, self.combo_pp_source):
            combo.setMinimumContentsLength(12)
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            combo.setMinimumWidth(0)
        outer_sel.addWidget(model_box)
        tissue_box = QGroupBox("3. Segmentation Tool")
        tissue_layout = QVBoxLayout(tissue_box)
        self.cb_pp_tissue_only = QCheckBox("Enable sample segmentation")
        self.cb_pp_tissue_only.setToolTip("Intensity segmentation of the sample. Training classes remain those of the selected ROI dataset.")
        tissue_layout.addWidget(self.cb_pp_tissue_only)
        self.pp_tissue_controls = QWidget()
        tissue_form = QGridLayout(self.pp_tissue_controls)
        tissue_form.setContentsMargins(0, 0, 0, 0)
        self.pp_tissue_band = QDoubleSpinBox()
        self.pp_tissue_band.setRange(400, 1000); self.pp_tissue_band.setDecimals(6); self.pp_tissue_band.setValue(720)
        self.pp_tissue_band.setSuffix(" nm")
        self.combo_pp_segmentation = QComboBox()
        self.combo_pp_segmentation.addItem("Otsu (automatic)", "otsu")
        self.combo_pp_segmentation.addItem("Manual threshold", "threshold")
        self.cb_pp_log_otsu = QCheckBox("Otsu on log intensity")
        self.cb_pp_log_otsu.setChecked(True)
        self.pp_tissue_threshold = QDoubleSpinBox()
        self.pp_tissue_threshold.setRange(-1e6, 1e9); self.pp_tissue_threshold.setDecimals(4); self.pp_tissue_threshold.setValue(10)
        self.pp_tissue_threshold.setToolTip("Threshold in the scientific cube's native intensity units.")
        self.pp_tissue_erosion = QSpinBox()
        self.pp_tissue_erosion.setRange(0, 100); self.pp_tissue_erosion.setValue(5); self.pp_tissue_erosion.setSuffix(" px")
        self.combo_pp_segmentation.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.combo_pp_segmentation.setMinimumContentsLength(6)
        for row, (label, widget) in enumerate((("Band", self.pp_tissue_band), ("Method", self.combo_pp_segmentation),
                ("Threshold", self.pp_tissue_threshold), ("Erosion", self.pp_tissue_erosion))):
            tissue_form.addWidget(QLabel(label), row, 0)
            tissue_form.addWidget(widget, row, 1)
        tissue_form.addWidget(self.cb_pp_log_otsu, 5, 0, 1, 2)
        self.btn_preview_tissue = QPushButton("Preview sample mask...")
        self.btn_preview_tissue.setToolTip("Compute the mask on the selected corrected cube, without training or classifying. Dashed cyan: retained sample boundary.")
        self.btn_preview_tissue.clicked.connect(self._preview_tissue_mask)
        tissue_form.addWidget(self.btn_preview_tissue, 6, 0, 1, 2)
        tissue_layout.addWidget(self.pp_tissue_controls)
        self.pp_tissue_controls.setVisible(False)
        self.cb_pp_tissue_only.toggled.connect(self.pp_tissue_controls.setVisible)
        self.combo_pp_segmentation.currentIndexChanged.connect(self._update_pp_segmentation_controls)
        self._update_pp_segmentation_controls()
        outer_sel.addWidget(tissue_box)
        actions = QGroupBox("4. Run / inspect results")
        actions_layout = QVBoxLayout(actions)
        self.btn_run_pp.setText("Classify ENVI cube")
        actions_layout.addWidget(self.btn_run_pp)
        self.btn_load_pp.setText("Open saved results...")
        self.btn_load_pp.setToolTip("Display saved maps; does not run a classifier.")
        actions_layout.addWidget(self.btn_load_pp)
        outer_sel.addWidget(actions)
        left.addLayout(outer_sel)

        self.hsl_label = QLabel("Classification preview")
        self.hsl_label.setAlignment(Qt.AlignCenter)
        self.hsl_label.setMinimumSize(200, 110)
        self.hsl_label.setMaximumHeight(200)
        self.hsl_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.hsl_label.hide()  # Legacy result hooks; the shared viewer is the sole image display.

        self.pp_svm_training_status = QLabel("SVM training: idle")
        left.addWidget(self.pp_svm_training_status)
        self.pp_process_status = QLabel("Processing: idle")
        self.pp_process_status.setTextFormat(Qt.PlainText)
        self.pp_process_status.setWordWrap(True)
        left.addWidget(self.pp_process_status)
        self.log_output = QPlainTextEdit(); self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText("Post-processing Log...")
        self.log_output.setFixedHeight(75)
        left.addWidget(self.log_output, stretch=0)
        left.addStretch(1)

        # -------- Right: interactive overlay + hovered spectrum --------
        overlay_panel = self._make_abundance_overlay_tab(embedded=True)

        # -------- Assemble main layout --------
        left_widget = QWidget()
        left_widget.setLayout(left)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left_widget)
        left_scroll.setMinimumWidth(440)
        left_scroll.setMaximumWidth(510)
        main.addWidget(left_scroll, 0)
        main.addWidget(overlay_panel, 2)

        # -------- Connections --------
        self.btn_run_pp.clicked.connect(self._run_postprocessing_external)
        self.btn_refresh_pp_models.clicked.connect(self._refresh_pp_models)
        self.btn_load_pp.clicked.connect(self._load_hsl_from_corrected)
        self.btn_refresh_pp_cubes.clicked.connect(self._refresh_pp_cube_choices)
        self.btn_browse_pp_cube.clicked.connect(self._browse_pp_cube)
        self.combo_pp_cube.currentIndexChanged.connect(self._update_pp_cube_path)
        self.combo_pp_source.currentIndexChanged.connect(self._update_pp_source_mode)
        self._refresh_pp_cube_choices()
        self._update_pp_source_mode()

        # === Normalization status box (unmix + library) ===
        from PyQt5.QtWidgets import QFrame
        self.pp_norm_box = QFrame()
        self.pp_norm_box.setFrameShape(QFrame.StyledPanel)
        self.pp_norm_box.setStyleSheet("""
            QFrame#pp_norm_box {
                border: 2px solid #777; border-radius: 8px;
                background: #1e1e1e; color: #eaeaea;
            }
            QLabel { color: #eaeaea; }
        """)
        self.pp_norm_box.setObjectName("pp_norm_box")
        box_v = QVBoxLayout(self.pp_norm_box)
        box_v.setContentsMargins(10, 10, 10, 10)
        box_v.setSpacing(4)

        title_lbl = QLabel("<b>Normalization status</b>")
        self.pp_norm_unmix_lbl = QLabel("Unmixing: —")
        self.pp_norm_lib_lbl   = QLabel("Library: —")
        self.pp_norm_features_lbl = QLabel("SVM features: spectral bands")
        for label in (self.pp_norm_unmix_lbl, self.pp_norm_lib_lbl, self.pp_norm_features_lbl):
            label.setWordWrap(True)

        box_v.addWidget(title_lbl)
        box_v.addWidget(self.pp_norm_unmix_lbl)
        box_v.addWidget(self.pp_norm_lib_lbl)
        box_v.addWidget(self.pp_norm_features_lbl)

        # Valeurs initiales
        init_unmix = getattr(self, "selected_norm_mode", "none")
        self.pp_norm_unmix_lbl.setText(f"Classification: {init_unmix}")
        self.pp_norm_lib_lbl.setText("Library: —")
        # contour neutre pour commencer
        self.pp_norm_box.setStyleSheet(self.pp_norm_box.styleSheet().replace("#777", "#777"))

        # Ajoute la box au panneau de sélection (colonne de gauche)
        sel.addWidget(self.pp_norm_box)
        self._init_pp_model_status(sel)

        return tab

    def _update_pp_source_mode(self, _index=0):
        self.btn_run_pp.setText("Classify ENVI cube")
        self.btn_preview_tissue.setEnabled(True)
        self.combo_pp_classification.setToolTip(
            "Linear or RBF classification at native ENVI resolution. No radiometric or shading correction."
        )

    def _update_pp_segmentation_controls(self, _index=0):
        manual = self.combo_pp_segmentation.currentData() == "threshold"
        self.pp_tissue_threshold.setEnabled(manual)
        self.cb_pp_log_otsu.setEnabled(not manual)

    def _preview_tissue_mask(self):
        from types import SimpleNamespace
        from shared.envi import load_spectral_cube
        from shared.tissue_segmentation import segment_cube, draw_dashed_boundary
        path = Path(str(self.combo_pp_cube.currentData() or ""))
        if not path.is_file():
            QMessageBox.warning(self, "Tissue mask", "Select a corrected cube first.")
            return
        cube = None
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            cube, wavelengths = load_spectral_cube(path)
            args = SimpleNamespace(segmentation_band_nm=self.pp_tissue_band.value(),
                segmentation_method=self.combo_pp_segmentation.currentData(),
                segmentation_threshold=self.pp_tissue_threshold.value(),
                segmentation_log_intensity=self.cb_pp_log_otsu.isChecked(),
                tissue_mask_erosion_px=self.pp_tissue_erosion.value())
            _, mask, threshold, fallback = segment_cube(cube, wavelengths, args)
            if not mask.any():
                raise ValueError("Empty mask after erosion. Adjust the threshold or reduce erosion.")
            base = None
            from shared.corrected_layout import nearby_image_folders
            for folder in nearby_image_folders(path):
                candidate = read_image(str(folder/'base_image_bgr.png')) if (folder/'base_image_bgr.png').is_file() else None
                if candidate is not None and candidate.shape[:2] == mask.shape:
                    base = candidate
                    break
            if base is None:
                band = int(np.flatnonzero(np.isclose(wavelengths, args.segmentation_band_nm))[0])
                intensity = np.asarray(cube[:, :, band], dtype=np.float32)
                low, high = np.nanpercentile(intensity, [1, 99])
                gray = np.nan_to_num(np.clip((intensity-low)/max(float(high-low), 1e-6), 0, 1))*255
                base = cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            preview = draw_dashed_boundary(base, mask)
        except Exception as exc:
            QMessageBox.warning(self, "Tissue mask preview", str(exc))
            return
        finally:
            if isinstance(cube, np.memmap):
                cube._mmap.close()
            QApplication.restoreOverrideCursor()
        previous = getattr(self, '_pp_mask_preview', None)
        if previous is not None:
            previous.close()
        dialog = QDialog(self)
        dialog.setWindowTitle("Tissue mask preview - no classification")
        layout = QVBoxLayout(dialog)
        details = QLabel(f"{path.name}\n{args.segmentation_band_nm:g} nm | {args.segmentation_method} | threshold={threshold:.5g} | erosion={args.tissue_mask_erosion_px}px\n"
                         f"Retained: {int(mask.sum()):,}/{mask.size:,} pixels. Dashed cyan: boundary after erosion.")
        details.setWordWrap(True)
        layout.addWidget(details)
        if fallback:
            layout.addWidget(QLabel("All components touch the frame: inspect the selected region carefully."))
        image = ResizablePixmapLabel()
        image.setAlignment(Qt.AlignCenter)
        image.setMinimumSize(320, 240)
        image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        height, width = preview.shape[:2]
        rgb = np.ascontiguousarray(preview[:, :, ::-1])
        qimage = QImage(rgb.data, width, height, width*3, QImage.Format_RGB888).copy()
        image.setPixmap(QPixmap.fromImage(qimage))
        layout.addWidget(image, 1)
        close = QPushButton("Close preview")
        close.clicked.connect(dialog.close)
        layout.addWidget(close)
        dialog.resize(900, 850)
        self._pp_mask_preview = dialog
        dialog.show()

    def _update_pp_cube_path(self, _index=0):
        path = str(self.combo_pp_cube.currentData() or "")
        self.pp_cube_path_label.setText(Path(path).name if path else "No cube selected")
        self.pp_cube_path_label.setToolTip(path)
        self.combo_pp_cube.setToolTip(path)

    def _refresh_pp_cube_choices(self, _checked=False, selected_path=None):
        selected = str(selected_path or self.combo_pp_cube.currentData() or "")
        from shared.envi_catalog import registered_sources
        roots = (PROJECT_ROOT/'synthetic_data', PROJECT_ROOT/'data')
        sources = {str(path.resolve()) for root in roots if root.is_dir() for path in root.rglob('*.hdr')}
        sources.update(registered_sources())
        sources = {os.path.normcase(value): value for value in sources}
        paths = sorted(sources.values(), key=lambda value: Path(value).name.casefold())
        if selected and Path(selected).is_file() and selected not in paths:
            paths.insert(0, selected)
        self.combo_pp_cube.blockSignals(True)
        self.combo_pp_cube.clear()
        for path in paths:
            self.combo_pp_cube.addItem(f"{Path(path).parent.name} / {Path(path).name}", path)
        self.combo_pp_cube.setCurrentIndex(max(0, self.combo_pp_cube.findData(selected)))
        self.combo_pp_cube.blockSignals(False)
        self._update_pp_cube_path()

    def _browse_pp_cube(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose an already corrected cube",
                    str(PROJECT_ROOT / "synthetic_data"), "ENVI cube (*.hdr *.raw *.img *.dat *.bin)")
        if path:
            try:
                path = self._register_cube_source(path)
                self._load_data_cube(path)
            except Exception as exc:
                QMessageBox.warning(self, "Import", str(exc))

    def _on_pp_profile_changed(self, _index=0):
        """Keep both tabs on the same dataset without modifying its files."""
        mode = self.combo_pp_normalization.currentData() or "none"
        profile = self._get_selected_postprocessing_profile_paths()["profile"]
        self.roi_dataset_profile = profile
        if hasattr(self, "combo_roi_dataset"):
            blocked = self.combo_roi_dataset.blockSignals(True)
            self.combo_roi_dataset.setEditText(profile)
            self.combo_roi_dataset.blockSignals(blocked)
        self._refresh_roi_library_view()
        self._set_pp_norm_status(mode, getattr(self, 'current_library_norm', None))
        self._queue_pp_model_check()

    def _refresh_pp_models(self):
        self._refresh_svm_model_choices()
        if self.combo_pp_model.currentData():
            self._on_pp_profile_changed()
        self._queue_pp_model_check()

    def _on_pp_normalization_changed(self, _index=0):
        mode = self.combo_pp_normalization.currentData() or "none"
        try:
            paths = self._get_selected_postprocessing_profile_paths()
        except (ValueError, FileNotFoundError) as exc:
            self._set_pp_norm_status(getattr(self, "selected_norm_mode", "none"), None)
            QMessageBox.warning(self, "Normalization", str(exc))
            return
        lib_mode = None
        library = paths.get('library')
        if library and library.is_file():
            try:
                lib_mode = (json.loads(library.read_text(encoding='utf-8')).get('_meta') or {}).get('normalization')
            except (OSError, ValueError, TypeError):
                pass
        self._set_pp_norm_status(mode, lib_mode)
        self.pp_norm_features_lbl.setToolTip('Classifier features come from the raw ROI dataset, independently of the library preview normalization.')
        self.pp_svm_training_status.setText('SVM: settings updated.')
        self._queue_pp_model_check()

    def _refresh_postprocessing_overlay(self, mix_dir: str) -> None:
        """Load the post-processing result into the embedded overlay explorer."""
        self.pp_mix_dir = mix_dir
        self.last_mix_dir = mix_dir
        reload_overlay = getattr(self, "_reload_abundance_overlay_folder", None)
        if callable(reload_overlay):
            reload_overlay(mix_dir, False)

    def _set_pp_norm_status(self, unmix_mode: str | None, lib_mode: str | None):
        """Met à jour les libellés et le contour couleur de la box post-proc."""
        if unmix_mode not in {mode for _, mode in NORMALIZATION_OPTIONS}:
            unmix_mode = "none"
        # The ROI selector edits library previews. Postprocessing selects a
        # model preprocessing variant and must not silently change that editor.
        for name in ("combo_pp_normalization",):
            combo = getattr(self, name, None)
            if combo is not None:
                blocked = combo.blockSignals(True)
                combo.setCurrentIndex(max(0, combo.findData(unmix_mode or "none")))
                combo.blockSignals(blocked)
        if not hasattr(self, "pp_norm_box"):
            return
        if unmix_mode is None:
            unmix_mode = "none"
        # lib_mode peut être None si la lib n'existe pas encore
        lib_txt = "SNV + 3 ratios" if lib_mode == "snv_three_ratios" else lib_mode if lib_mode else "—"

        mode_label = "SNV + 3 ratios" if unmix_mode == "snv_three_ratios" else unmix_mode
        self.pp_norm_unmix_lbl.setText(f"Classification: {mode_label}")
        self.pp_norm_lib_lbl.setText(f"Library: {lib_txt}")
        self.pp_norm_features_lbl.setText(
            "SVM features: −log10(R/100), no ratios; reflectance floor: 0.0001%" if unmix_mode == "log" else
            "SVM features: SNV bands + raw R620/R580, R560/R540, R700/R620" if unmix_mode == "snv_three_ratios" else
            "SVM features: spectral bands" if unmix_mode in {"none", "snv_no_ratio"}
            else "SVM features: normalized bands + raw R620/R580"
        )

        same = (lib_mode is not None) and (str(unmix_mode).lower() == str(lib_mode).lower())
        border = "#2e7d32" if same else "#777777"
        self.pp_norm_lib_lbl.setToolTip('Library preview only. SVM preprocessing is applied directly to raw dataset spectra; a library mismatch does not require retraining.')
        # on ne touche qu'à la couleur de bordure (le reste inchangé)
        base_css = """
            QFrame#pp_norm_box {
                border: 2px solid %s; border-radius: 8px;
                background: #1e1e1e; color: #eaeaea;
            }
            QLabel { color: #eaeaea; }
        """ % border
        self.pp_norm_box.setStyleSheet(base_css)

    def _pp_load_class_masks(self, cube_dir: str):
        """Construit les checkboxes et relie les PNG individuels générés."""
        lay = self.pp_mask_checks_container.layout()
        while lay.count():
            w = lay.takeAt(0).widget()
            if w:
                w.deleteLater()

        self.pp_mask_checks = {}
        for f in os.listdir(cube_dir):
            if f.startswith("mask_") and f.endswith(".png"):
                cls = f[5:-4]  # nom de classe
                cb = QCheckBox(cls)
                cb.setChecked(True)
                cb.stateChanged.connect(lambda s, c=cls: self._pp_update_mask_display(cube_dir))
                lay.addWidget(cb)
                self.pp_mask_checks[cls] = cb
        lay.addStretch(1)
        # affichage initial
        self._pp_update_mask_display(cube_dir)

    def _pp_update_mask_display(self, cube_dir: str):
        """Compose l'image à partir des PNG cochés et l'affiche."""
        imgs = []
        for cls, cb in self.pp_mask_checks.items():
            if cb.isChecked():
                path = os.path.join(cube_dir, f"mask_{cls}.png")
                if os.path.exists(path):
                    img = read_image(path, cv2.IMREAD_COLOR)
                    imgs.append(img)
        if not imgs:
            self.mask_label.clear()
            return
        overlay = np.zeros_like(imgs[0], np.float32)
        for im in imgs:
            overlay += im.astype(np.float32) / 255.0
        overlay = np.clip(overlay / max(1, len(imgs)), 0, 1)
        overlay = (overlay * 255).astype(np.uint8)
        rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self._safe_set_pixmap("mask_label", QPixmap.fromImage(qimg))

    def _pp_update_displays(self, mix_dir: str):
        """
        Compose une image overlay des masques cochés sur l'image moyenne N&B du cube
        (base_image_mean.png). Fallbacks : *_mean.png dans cube/, puis overlay_mixture->N&B,
        puis fond noir si rien trouvé.
        """

        self.last_mix_dir = mix_dir

        # 1) Fond = image moyenne en N&B (priorité absolue)
        base = None
        # a) corrected/<sweep>_corrected/base_image_mean.png
        p_mean_corr = os.path.join(mix_dir, "base_image_mean.png")
        if os.path.isfile(p_mean_corr):
            base = read_image(p_mean_corr, cv2.IMREAD_GRAYSCALE)

        # b) corrected/<sweep>_corrected/cube/*_mean.png
        if base is None:
            cube_dir = os.path.join(mix_dir, "cube")
            if os.path.isdir(cube_dir):
                for f in os.listdir(cube_dir):
                    if f.lower().endswith("_mean.png"):
                        p = os.path.join(cube_dir, f)
                        g = read_image(p, cv2.IMREAD_GRAYSCALE)
                        if g is not None:
                            base = g
                            break

        # c) dernier recours : overlay_mixture converti en N&B
        if base is None:
            p_overlay = os.path.join(mix_dir, "overlay_mixture.png")
            if os.path.isfile(p_overlay):
                tmp = read_image(p_overlay, cv2.IMREAD_COLOR)
                if tmp is not None:
                    base = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)

        # d) très dernier recours : fond noir basé sur la taille d'un masque
        if base is None:
            if getattr(self, "pp_masks_paths", None):
                for p in self.pp_masks_paths.values():
                    g = read_image(p, cv2.IMREAD_GRAYSCALE)
                    if g is not None:
                        base = np.zeros_like(g, np.uint8)
                        break

        if base is None:
            if hasattr(self, "mask_label"): self.mask_label.clear()
            return

        # 2) Converti en BGR juste pour pouvoir colorer par-dessus
        base_bgr = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

        # 3) Base : fond noir au lieu de l'image moyenne
        H, W = base_bgr.shape[:2]
        canvas = np.zeros((H, W, 3), np.uint8)  # <- fond noir pur

        # 4) Superpose uniquement les classes cochées
        selected = sorted(getattr(self, "pp_selected", []))
        pp_alpha_dict = getattr(self, "pp_class_alpha", {})
        default_alpha = float(getattr(self, "pp_default_alpha", 1.0))

        for cls in selected:
            # chemin du masque
            mask_path = None
            if hasattr(self, "pp_masks_paths") and isinstance(self.pp_masks_paths, dict):
                mask_path = self.pp_masks_paths.get(cls)
            if not mask_path:
                mask_path = os.path.join(mix_dir, f"mask_{cls}.png")
            if not os.path.isfile(mask_path):
                continue

            # lecture du masque
            mask = read_image(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            # redimension
            if mask.shape[:2] != (H, W):
                mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

            # color_for() already returns the classifier's BGR color.
            col_bgr = np.array(color_for(cls), dtype=np.uint8)

            # couche colorée
            layer = np.zeros_like(canvas, dtype=np.uint8)
            layer[mask > 127] = col_bgr

            # alpha par classe
            alpha_cls = float(pp_alpha_dict.get(cls, default_alpha))
            alpha_cls = np.clip(alpha_cls, 0.0, 1.0)

            # fusion
            canvas = cv2.addWeighted(canvas, 1.0, layer, alpha_cls, 0.0)

        # 5) Affiche dans le QLabel
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, 3*w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        if hasattr(self, "mask_label") and self.mask_label:
            pix = pix.scaled(self.mask_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.mask_label.setPixmap(pix)

    def _pp_get_checked_classes(self) -> list[str]:
        """Retourne la liste des classes cochées dans la colonne de droite."""
        return [c for c, cb in getattr(self, "pp_mask_checks", {}).items() if cb.isChecked()]

    def _update_interactive_spectra(self, mix_dir: str):
        """
        Charge corrected/<sweep>_corrected/spectra_data.npz et place un SpectraInteractive
        filtré sur les classes cochées, avec affichage de l'écart-type (±1σ) en zone ombrée.
        """

        self.last_mix_dir = mix_dir  # pour les toggles
        npz_path = os.path.join(mix_dir, "spectra_data.npz")

        if hasattr(self, "spectra_status"):
            self.spectra_status.setText(f"Loading: {npz_path}")

        # 1) Fichier manquant
        if not os.path.isfile(npz_path):
            print("[SPECTRA][ERR] spectra_data.npz NOT FOUND")
            if hasattr(self, "spectra_status"):
                self.spectra_status.setText("No spectra_data.npz found")
            if hasattr(self, "spectra_layout"):
                while self.spectra_layout.count():
                    w = self.spectra_layout.takeAt(0).widget()
                    if w:
                        w.setParent(None)
            return

        # 2) Lecture npz
        d = np.load(npz_path, allow_pickle=True)
        classes_arr = d.get("class_names", d.get("classes"))
        if classes_arr is None:
            print("[SPECTRA][ERR] no 'class_names'/'classes' in npz")
            if hasattr(self, "spectra_status"):
                self.spectra_status.setText("Invalid npz: no class list")
            return
        classes = [str(x) for x in classes_arr.tolist()]

        # 3) Abscisses
        if "wavelengths" in d.files:
            bands = d["wavelengths"].astype(float)
        elif "bands" in d.files:
            bands = d["bands"].astype(float)
        elif "bands_idx" in d.files:
            bands = d["bands_idx"].astype(float)
        else:
            bands = np.arange(d["lib_median"].shape[1], dtype=np.float64)

        # 4) Données
        lib_median = np.asarray(d["lib_median"], dtype=float)
        lib_std    = d.get("lib_stdev", d.get("lib_std", None))
        img_mean   = d.get("img_mean", None)
        if img_mean is not None:
            img_mean = np.asarray(img_mean, dtype=float)

        # Nettoyage std
        if lib_std is not None:
            lib_std = np.asarray(lib_std, dtype=float)
            lib_std = np.nan_to_num(lib_std, nan=0.0, posinf=0.0, neginf=0.0)
            lib_std = np.clip(lib_std, 0.0, None)

        colors_bgr = d.get("class_colors_bgr", None)
        if colors_bgr is not None:
            colors_bgr = np.asarray(colors_bgr, dtype=np.uint8)

        # 5) Aligner longueurs (troncature tolérante)
        Bm = lib_median.shape[1]
        Bs = lib_std.shape[1] if lib_std is not None else Bm
        Bi = img_mean.shape[1] if img_mean is not None else Bm
        Bb = bands.shape[0]
        m = min(Bb, Bm, (Bs if lib_std is not None else Bm), (Bi if img_mean is not None else Bm))

        bands = bands[:m]
        lib_median = lib_median[:, :m]
        if lib_std is not None: lib_std = lib_std[:, :m]
        if img_mean is not None: img_mean = img_mean[:, :m]

        # 6) Filtrage UI
        selected = sorted(getattr(self, "pp_selected", []))
        idxs = [i for i, c in enumerate(classes) if (not selected or c in selected)]
        classes_sel = [classes[i] for i in idxs]

        # 7) Remplacer le widget de tracé
        if hasattr(self, "spectra_layout"):
            while self.spectra_layout.count():
                w = self.spectra_layout.takeAt(0).widget()
                if w:
                    w.setParent(None)

            w_plot = SpectraInteractive()
            # injecte les données prêtes (comme notebook)
            w_plot._data = {
                "classes": classes,
                "bands":   bands,
                "lib_med": lib_median,
                "lib_std": lib_std,
                "img_mean": img_mean,
                "colors_bgr": colors_bgr
            }
            setattr(w_plot, "show_std", True)
            setattr(w_plot, "show_lib", True)
            setattr(w_plot, "show_img", img_mean is not None)
            w_plot.set_selected_classes(classes_sel)

            try:
                w_plot.update_plot()
            except Exception as e:
                print(f"[SPECTRA][ERR] update_plot failed: {e!r}")
                if hasattr(self, "spectra_status"):
                    self.spectra_status.setText(f"Failed to load spectra: {e}")
            else:
                self.spectra_layout.addWidget(w_plot)
                spectra_log_key = (os.path.abspath(npz_path), tuple(classes_sel), int(m))
                if getattr(self, "_last_spectra_log_key", None) != spectra_log_key:
                    print(f"[SPECTRA] Loaded: {len(classes_sel)} classes, {m} bands")
                    self._last_spectra_log_key = spectra_log_key
                if hasattr(self, "spectra_status"):
                    if lib_std is None or float(np.nanmax(lib_std)) == 0.0:
                        self.spectra_status.setText(f"{len(classes_sel)} class(es) displayed (no/zero std)")
                    else:
                        self.spectra_status.setText(f"{len(classes_sel)} class(es) displayed (±1σ should be visible)")

    def _pp_load_class_elements(self, mix_dir: str):
        """Construit la liste des checkboxes de classes d'après les mask_*.png trouvés dans mix_dir."""

        # On attend un QGroupBox ou un conteneur existant côté UI
        # Ici, on suppose que tu as self.pp_mask_checks_container déjà créé dans l'onglet.
        container = getattr(self, "pp_mask_checks_container", None)
        if container is None:
            # Crée un conteneur minimal si absent
            self.pp_mask_checks_container = QGroupBox("Displayed classes")
            self.pp_mask_checks_container.setLayout(QVBoxLayout())
            # Essaie de l'ajouter à la colonne de droite si possible
            # (à adapter si nécessaire)
            return

        # purge layout
        lay = self.pp_mask_checks_container.layout()
        while lay.count():
            w = lay.takeAt(0).widget()
            if w:
                w.deleteLater()

        # (re)créé le dict de checkboxes
        self.pp_mask_checks = {}
        names = []
        for f in sorted(os.listdir(mix_dir)):
            fl = f.lower()
            if fl.startswith("mask_") and fl.endswith(".png"):
                cname = f[len("mask_"):-4]
                names.append(cname)

        if not names:
            # rien à lister
            return

        # par défaut : coché
        for name in names:
            cb = QCheckBox(name)
            cb.setChecked(True)
            lay.addWidget(cb)
            self.pp_mask_checks[name] = cb
            cb.toggled.connect(lambda _=False, cls=name: self._on_class_toggle(cls))

        # set initial selection
        self.pp_selected = set(names)

    def _on_class_toggle(self, cls_name: str):
        """Cochage/décochage d'une classe -> met à jour l'overlay et le graphique."""
        if getattr(self, "pp_selected", None) is None:
            self.pp_selected = set()

        cb = self.pp_mask_checks.get(cls_name)
        if cb and cb.isChecked():
            self.pp_selected.add(cls_name)
        else:
            self.pp_selected.discard(cls_name)

        # mix_dir = corrected/<sweep>_corrected  (là où sont mask_*.png et spectra_data.npz)
        mix_dir = getattr(self, "last_mix_dir", None)
        if mix_dir is None:
            # essaie d'utiliser le dernier dossier chargé par _load_hsl_from_corrected
            # (on l’enregistre dans _update_interactive_spectra)
            return
        self._pp_update_displays(mix_dir)
        self._update_interactive_spectra(mix_dir)

    def _start_postprocessing_external(self):

        self._should_abort = False
        self.log_output.clear()
        msg = "===== [POSTPROCESSING] Launching pipeline ====="
        print(msg)
        self.log_output.appendPlainText(msg)
        self.log_output.verticalScrollBar().setValue(self.log_output.verticalScrollBar().maximum())
        self.pp_process_status.setText("Preparing post-processing…")
        QApplication.processEvents()

        # Chemins
        PYTHON64 = self._get_python64_path() 
        SCRIPT64 = str(PROJECT_ROOT / "processing" / "postprocessing_runner.py")

        if not os.path.exists(PYTHON64):
            msg = f"⚠️ Interpreter not found: {PYTHON64}"
            print(msg)
            self.log_output.appendPlainText(msg)
            self.pp_process_status.setText(msg)
            return
        if not os.path.exists(SCRIPT64):
            msg = f"⚠️ Script not found : {SCRIPT64}"
            print(msg)
            self.log_output.appendPlainText(msg)
            self.pp_process_status.setText(msg)
            return

        cube_path = str(self.combo_pp_cube.currentData() or "")
        if not cube_path or not Path(cube_path).is_file():
            QMessageBox.warning(self, "Classification", "Select an ENVI cube first.")
            return
        args = [PYTHON64, SCRIPT64, "--cube_path", cube_path]
        target_resolution = "native"

        # --- Selected SVM family and its associated dataset/library ---
        try:
            profile_paths = self._get_selected_postprocessing_profile_paths()
        except Exception as exc:
            msg = f"Cannot start post-processing: {exc}"
            self.log_output.appendPlainText(msg)
            self.pp_process_status.setText(msg)
            QMessageBox.warning(self, "Post-processing", msg)
            return

        missing = [
            str(profile_paths[key])
            for key in ("dataset", "library")
            if not profile_paths[key].is_file()
        ]
        if missing:
            msg = "Missing selected ROI/SVM files:\n" + "\n".join(missing)
            self.log_output.appendPlainText(msg)
            self.pp_process_status.setText("Selected model is incomplete")
            QMessageBox.warning(self, "Post-processing", msg)
            return

        # The SVM prepares raw ROI spectra; the library preview may use another mode.
        lib_mode = None
        try:
            lib_data = json.loads(profile_paths["library"].read_text(encoding="utf-8"))
            lib_mode = (lib_data.get("_meta", {}) or {}).get("normalization")
        except Exception:
            lib_mode = None
        sel_mode = str(self.combo_pp_normalization.currentData() or "none")
        classification_mode = str(
            getattr(self, "selected_pp_classification_mode", "linear")
        ).lower()
        if classification_mode not in {"linear", "rbf", "hybrid"}:
            classification_mode = "linear"
        svm_type = "rbf" if classification_mode in {"rbf", "hybrid"} else "linear"
        selected_bands = None
        bands_profile = profile_paths["model"].parent / f"{profile_paths['profile']}_svm_bands.json"
        if bands_profile.is_file():
            try:
                selected_bands = [float(value) for value in json.loads(
                    bands_profile.read_text(encoding="utf-8")
                ).get("selected_wavelengths_nm", [])]
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                QMessageBox.warning(self, "SVM bands", f"Invalid SVM band profile:\n{bands_profile}")
                return

        # The trained spectral schema is fixed by the dataset, never by the cube.
        selected_bands = (getattr(self, "_pp_model_result", None) or {}).get("bands") or selected_bands

        args += [
            "--norm", sel_mode,
            "--target_resolution", target_resolution,
            "--svm_type", svm_type,
            "--classification_mode", classification_mode,
            "--dataset_path", str(profile_paths["dataset"]),
            "--library_path", str(profile_paths["library"]),
            "--model_path", str(profile_paths["model"]),
        ]
        if self.cb_pp_tissue_only.isChecked():
            args += ["--tissue_only", "--segmentation_band_nm", str(self.pp_tissue_band.value()),
                     "--segmentation_method", str(self.combo_pp_segmentation.currentData()),
                     "--segmentation_threshold", str(self.pp_tissue_threshold.value()),
                     "--tissue_mask_erosion_px", str(self.pp_tissue_erosion.value()),
                     "--segmentation_log_intensity" if self.cb_pp_log_otsu.isChecked() else "--no-segmentation_log_intensity",
                     ]
        if selected_bands:
            args += ["--selected_wavelengths_nm", *[f"{value:g}" for value in selected_bands]]

        self.log_output.appendPlainText(f"SVM model: {profile_paths['model']}")
        self.log_output.appendPlainText(f"ROI dataset: {profile_paths['dataset']}")
        self.log_output.appendPlainText(f"ROI library: {profile_paths['library']}")
        self.log_output.appendPlainText(f"Normalization: {sel_mode}")
        self.log_output.appendPlainText(f"Output resolution: {target_resolution}")
        self.log_output.appendPlainText(
            "SVM bands: " + (", ".join(f"{value:g}" for value in selected_bands) if selected_bands else "all dataset bands")
        )
        classification_label = {
            "linear": "Linear classic on full image",
            "rbf": "RBF classic on full image",
            "hybrid": "Conservative tissue mask + full-resolution Linear + RBF on uncertain tissue",
        }[classification_mode]
        self.log_output.appendPlainText(f"Classification mode: {classification_label}")
        self._set_pp_norm_status(sel_mode, lib_mode)

        self.pp_process_status.setText("Starting post-processing…")
        self.log_output.appendPlainText("Starting post-processing…")
        self.log_output.verticalScrollBar().setValue(self.log_output.verticalScrollBar().maximum())
        QApplication.processEvents()

        if isinstance(getattr(self, "_proc_pp", None), QProcess) and self._proc_pp.state() != QProcess.NotRunning:
            QMessageBox.information(self, "Post-processing", "Post-processing is already running.")
            return

        process = QProcess(self)
        from gui.runtime import configure_qt_worker
        configure_qt_worker(process)
        process.setWorkingDirectory(str(PROJECT_ROOT))
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.setProgram(str(args[0]))
        process.setArguments([str(value) for value in args[1:]])
        process.readyReadStandardOutput.connect(self._read_postprocessing_output)
        process.finished.connect(self._postprocessing_finished)
        process.errorOccurred.connect(self._postprocessing_process_error)
        self._pp_output_buffer = ""
        self._pp_decoder = utf8_decoder()
        self._pp_started_at = time.monotonic()
        self._pp_last_output_at = self._pp_started_at
        self._proc_pp = process
        self.log_output.appendPlainText("Waiting for first step")
        process.start()
        self._start_postprocessing_monitor()

    @staticmethod
    def _format_elapsed_seconds(seconds: float) -> str:
        total = max(0, int(seconds))
        minutes, secs = divmod(total, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:d} h {minutes:02d} min {secs:02d} s"
        if minutes:
            return f"{minutes:d} min {secs:02d} s"
        return f"{secs:d} s"

    def _start_postprocessing_monitor(self) -> None:
        old_timer = getattr(self, "_pp_monitor_timer", None)
        if old_timer is not None:
            old_timer.stop()
            old_timer.deleteLater()
        timer = QTimer(self)
        timer.setInterval(30_000)
        timer.timeout.connect(self._postprocessing_monitor_tick)
        self._pp_monitor_timer = timer
        timer.start()

    def _stop_postprocessing_monitor(self) -> None:
        timer = getattr(self, "_pp_monitor_timer", None)
        if timer is not None:
            timer.stop()
            timer.deleteLater()
        self._pp_monitor_timer = None

    def _postprocessing_monitor_tick(self) -> None:
        """Report liveness every 30 s without mistaking silence for a crash."""
        process = getattr(self, "_proc_pp", None)
        if not isinstance(process, QProcess):
            self._stop_postprocessing_monitor()
            return

        now = time.monotonic()
        elapsed = now - getattr(self, "_pp_started_at", now)
        silent = now - getattr(self, "_pp_last_output_at", now)
        if process.state() == QProcess.Running:
            prefix = "[MONITOR] Process alive"
            if silent >= 60.0:
                prefix = "[MONITOR][WARNING] Process alive but possibly stalled"
            message = (
                f"{prefix} | elapsed {self._format_elapsed_seconds(elapsed)} | "
                f"no worker output for {self._format_elapsed_seconds(silent)}"
            )
        elif process.state() == QProcess.Starting:
            message = f"[MONITOR] Process starting | elapsed {self._format_elapsed_seconds(elapsed)}"
        else:
            message = "[MONITOR] Process has stopped; waiting for its exit result"
        self.pp_process_status.setText(message)
        self.log_output.appendPlainText(message)
        self.log_output.verticalScrollBar().setValue(self.log_output.verticalScrollBar().maximum())

    def _read_postprocessing_output(self) -> None:
        """Read child output only when Qt reports data, never with blocking readline()."""
        process = getattr(self, "_proc_pp", None)
        if not isinstance(process, QProcess):
            return
        chunk = self._pp_decoder.decode(bytes(process.readAllStandardOutput()))
        self._pp_output_buffer = getattr(self, "_pp_output_buffer", "") + chunk
        lines = self._pp_output_buffer.splitlines(keepends=True)
        self._pp_output_buffer = ""
        if lines and not lines[-1].endswith(("\n", "\r")):
            self._pp_output_buffer = lines.pop()
        for line in lines:
            self._handle_postprocessing_line(line.strip())

    def _handle_postprocessing_line(self, text: str) -> None:
        if not text:
            return
        self._pp_last_output_at = time.monotonic()
        print(text)
        self.pp_process_status.setText(text)
        self.log_output.appendPlainText(text)
        self.log_output.verticalScrollBar().setValue(self.log_output.verticalScrollBar().maximum())

        if text.startswith("M_PNG_PATH"):
            self._refresh_postprocessing_overlay(str(Path(text.split(" ", 1)[1].strip()).parent))

    def _postprocessing_finished(self, exit_code: int, _exit_status) -> None:
        self._stop_postprocessing_monitor()
        self._read_postprocessing_output()
        trailing = getattr(self, "_pp_output_buffer", "").strip()
        if trailing:
            self._handle_postprocessing_line(trailing)
        self._pp_output_buffer = ""
        aborted = bool(getattr(self, "_closing_gui", False) or getattr(self, "_should_abort", False))
        if not aborted:
            final_msg = "✅ Post-processing done!" if int(exit_code) == 0 else "❌ Error during post-processing"
            print(final_msg)
            self.pp_process_status.setText(final_msg)
            self.log_output.appendPlainText(final_msg)
        process = getattr(self, "_proc_pp", None)
        self._proc_pp = None
        if isinstance(process, QProcess):
            process.deleteLater()
        if not aborted:
            self._queue_pp_model_check()

    def _postprocessing_process_error(self, error) -> None:
        self._stop_postprocessing_monitor()
        if getattr(self, "_closing_gui", False):
            return
        process = getattr(self, "_proc_pp", None)
        detail = process.errorString() if isinstance(process, QProcess) else str(error)
        msg = f"Post-processing process error: {detail}"
        self.pp_process_status.setText(msg)
        self.log_output.appendPlainText(msg)

    def _load_hsl_from_corrected(self):
        directory = QFileDialog.getExistingDirectory(self, "Open saved classification results", str(PROJECT_ROOT/'results'))
        if directory:
            self._refresh_postprocessing_overlay(directory)
