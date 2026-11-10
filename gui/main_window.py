# Rôle : Assembler les onglets, l’état applicatif, les minuteries et la fermeture propre du matériel.

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
from PyQt5.QtCore import QEvent, QObject, QProcess, Qt, QTimer
from PyQt5.QtGui import QColor, QFont, QImage, QIntValidator, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QColorDialog, QComboBox, QFileDialog, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLayout, QLineEdit, QMessageBox,
    QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy, QSlider, QTabWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from shared.paths import PROJECT_ROOT

from gui.features.data_tab import DataMixin
from gui.features.overlay_tab import OverlayMixin
from gui.features.postprocessing_tab import PostProcessingMixin
from gui.features.roi_tab import RoiMixin
from gui.features.svm_models_tab import SvmModelsMixin
from gui.runtime import load_runtime_settings


class MainGUI(
    DataMixin,
    PostProcessingMixin,
    RoiMixin,
    SvmModelsMixin,
    OverlayMixin,
    QWidget,
):
    def __init__(self):
        super().__init__()
        self.runtime = load_runtime_settings()
        self.setWindowTitle("HERAMAN — HERA spectral analysis")
        self.resize(1200, 900)
        self._should_abort = False
        self._proc_pp = None  
        self._proc_roi = None
        self.roi_append_mode = True
        self.roi_class_checks = {}
        self.roi_lib_visible = {} 

        # Dossier global et unique pour ROI (indépendant des cubes)
        self.roi_out_dir = str(PROJECT_ROOT / "roi_outputs")
        os.makedirs(self.roi_out_dir, exist_ok=True)

        self.svm_models_dir = str(PROJECT_ROOT / "svm_models")
        os.makedirs(self.svm_models_dir, exist_ok=True)
        self.roi_dataset_profile = "roi_dataset"

        # One-time compatibility copy for the historical fixed library name.
        legacy_library = Path(self.roi_out_dir) / "library_medians.json"
        canonical_library = Path(self.roi_out_dir) / "roi_dataset_library.json"
        if legacy_library.is_file() and not canonical_library.exists():
            try:
                shutil.copy2(legacy_library, canonical_library)
            except OSError:
                pass

        # Widgets du panneau Library (assignés dans _make_roi_tab)
        self.roi_lib_canvas = None
        self.roi_lib_ax = None
        self.roi_lib_table = None
        self.roi_lib_tabs = None

        # PP: masks UI state
        self.pp_mask_checks = {}          # {class: QCheckBox}
        self.pp_mask_checks_container = None
        self.pp_mask_visible = {}         # {class: bool}
        self._pp_last_cube_dir = None     # où sont écrits les PNG
        # Binary-mask display: keep the classifier colors fully opaque.
        self.pp_class_alpha = {
            "background": 1.0,
            "fat": 1.0,
            "muscle": 1.0,
            "glandular": 1.0,
            "stroma": 1.0,
            "yellow ink": 1.0,
            "black ink": 1.0,
            "petri": 1.0,
            "shade": 1.0,
        }
        self.pp_default_alpha = 1.0


        self._build_ui()

        self.corrected_root_dir = getattr(self, "corrected_root_dir", str(PROJECT_ROOT / "results"))

    def _safe_set_pixmap(self, label_attr: str, pix: QPixmap):
        """Pose un pixmap dans un QLabel si le QLabel existe."""
        lbl = getattr(self, label_attr, None)
        if lbl is not None:
            lbl.setPixmap(pix)

    def _build_ui(self):
        print(">>> Initialization of UI <<<")

        main_layout = QVBoxLayout(self)
        main_layout.setSizeConstraint(QLayout.SetNoConstraint)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        self.tabs = QTabWidget(self)
        self.tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Tabs assembled from factories (everything grouped)
        tab_pp   = self._make_postproc_tab()
        tab_roi   = self._make_roi_tab()
        tab_svm = self._make_svm_models_tab()

        # Populate the new multi-dataset/model selectors after both tabs exist.
        self._refresh_roi_dataset_choices(select_profile=self.roi_dataset_profile)
        self._refresh_svm_model_choices(select_profile=self.roi_dataset_profile)

        self.tabs.addTab(tab_pp,   "Visualisation")
        self.tabs.addTab(tab_roi,   "ROI selection")
        svm_tab_index = self.tabs.addTab(tab_svm, "SVM Models")
        self.tabs.currentChanged.connect(
            lambda index: self._refresh_svm_dashboard() if index == svm_tab_index else None
        )

        main_layout.addWidget(self.tabs)



        print(">>> UI ready <<<")

    def _abort_all_processes(self):
        self._should_abort = True

        pp_monitor = getattr(self, "_pp_monitor_timer", None)
        if pp_monitor is not None:
            pp_monitor.stop()

        # Stop postprocessing externe
        if hasattr(self, "_proc_pp") and self._proc_pp is not None:
            process = self._proc_pp
            try:
                process.terminate()
                if isinstance(process, QProcess) and not process.waitForFinished(1500):
                    process.kill()
                    process.waitForFinished(1000)
            except Exception:
                pass
            self._proc_pp = None

        # Stop ROI externe
        if hasattr(self, "_proc_roi") and self._proc_roi is not None:
            process = self._proc_roi
            try:
                process.terminate()
                if isinstance(process, QProcess) and not process.waitForFinished(1500):
                    process.kill()
                    process.waitForFinished(1000)
                elif hasattr(process, "wait"):
                    try:
                        process.wait(timeout=1.5)
                    except Exception:
                        process.kill()
            except Exception:
                pass
            self._proc_roi = None

        # Stop dashboard retraining process.
        svm_process = getattr(self, "_svm_dashboard_process", None)
        if isinstance(svm_process, QProcess):
            try:
                svm_process.terminate()
                if not svm_process.waitForFinished(1500):
                    svm_process.kill()
                    svm_process.waitForFinished(1000)
            except Exception:
                pass
            self._svm_dashboard_process = None

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Recharge le dernier PNG si affiché
        label = getattr(self, "hsl_label", None)
        pixmap = label.pixmap() if label is not None else None
        if pixmap:
            scaled = pixmap.scaled(
                label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            label.setPixmap(scaled)

    def closeEvent(self, event):
        """Stop GUI jobs and release the directly connected hardware."""
        self._closing_gui = True
        try:
            self._abort_all_processes()
        except Exception:
            pass
        super().closeEvent(event)
