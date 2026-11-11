# Rôle : Construire l’onglet ROI et piloter la sélection des régions et des profils spectraux.

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
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLayout, QLineEdit, QMessageBox,
    QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy, QSlider, QTabWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from shared.paths import PROJECT_ROOT
from shared.process_io import worker_environment
from shared.svm_transforms import NORMALIZATION_OPTIONS


class RoiMixin:
    def _sanitize_roi_profile_name(self, value: str | None) -> str:
        """Return a safe file stem used by the dataset, library and SVM family."""
        raw = str(value or "").strip()
        if raw.lower().endswith(".npz"):
            raw = raw[:-4]
        name = Path(raw).name
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
        return name or "roi_dataset"

    def _get_current_roi_profile(self) -> str:
        combo = getattr(self, "combo_roi_dataset", None)
        raw = combo.currentText() if combo is not None else getattr(self, "roi_dataset_profile", "roi_dataset")
        profile = self._sanitize_roi_profile_name(raw)
        self.roi_dataset_profile = profile
        return profile

    def _get_roi_profile_paths(self, profile: str | None = None) -> dict[str, Path]:
        profile = self._sanitize_roi_profile_name(profile or self._get_current_roi_profile())
        return {
            "profile": profile,
            "dataset": Path(self.roi_out_dir) / f"{profile}.npz",
            "library": Path(self.roi_out_dir) / f"{profile}_library.json",
            "model": Path(self.svm_models_dir) / f"{profile}_svm.joblib",
        }

    def _refresh_roi_dataset_choices(self, select_profile: str | None = None):
        combo = getattr(self, "combo_roi_dataset", None)
        if combo is None:
            return

        requested = self._sanitize_roi_profile_name(
            select_profile or combo.currentText() or getattr(self, "roi_dataset_profile", "roi_dataset")
        )
        root = Path(self.roi_out_dir)
        root.mkdir(parents=True, exist_ok=True)

        files = [
            p for p in root.glob("*.npz")
            if "_backup_" not in p.stem.lower()
        ]
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        profiles = [self._sanitize_roi_profile_name(p.stem) for p in files]
        if requested not in profiles:
            profiles.insert(0, requested)
        if not profiles:
            profiles = ["roi_dataset"]

        combo.blockSignals(True)
        combo.clear()
        for profile in dict.fromkeys(profiles):
            combo.addItem(profile, profile)
        combo.setEditText(requested)
        combo.blockSignals(False)
        self.roi_dataset_profile = requested
        self._sync_roi_dataset_classes()

    def _sync_roi_dataset_classes(self):
        """Show labels from the selected NPZ, independently of the library preview."""
        layout = getattr(self, '_roi_annotation_classes_layout', None)
        if layout is None:
            return
        paths = self._get_roi_profile_paths()
        defaults = ['material_1', 'material_2', 'background']
        try:
            if paths['dataset'].is_file():
                with np.load(paths['dataset'], allow_pickle=True) as dataset:
                    classes = sorted({str(value).strip() for value in dataset['y'] if str(value).strip()})
            else:
                classes = defaults
        except Exception as exc:
            QMessageBox.warning(self, 'ROI dataset classes', f"Cannot read classes from {paths['dataset'].name}:\n{exc}")
            return
        identity = (paths['profile'], tuple(classes))
        if getattr(self, '_roi_annotation_classes_identity', None) == identity:
            return
        for checkbox in self.roi_class_checks.values():
            layout.removeWidget(checkbox)
            checkbox.deleteLater()
        self.roi_class_checks = {}
        for index, name in enumerate(classes):
            checkbox = QCheckBox(name)
            checkbox.setChecked(True)
            self.roi_class_checks[name] = checkbox
            layout.insertWidget(index, checkbox)
        self._roi_annotation_classes_identity = identity

    def _refresh_svm_model_choices(self, select_profile: str | None = None):
        combo = getattr(self, "combo_pp_model", None)
        if combo is None:
            return

        requested = self._sanitize_roi_profile_name(
            select_profile or getattr(self, "roi_dataset_profile", "roi_dataset")
        )
        root = Path(self.svm_models_dir)
        root.mkdir(parents=True, exist_ok=True)
        dataset_root = Path(self.roi_out_dir)
        profiles = sorted(
            {
                self._sanitize_roi_profile_name(path.stem)
                for path in dataset_root.glob("*.npz")
                if "_backup_" not in path.stem.lower()
            }
        )

        combo.blockSignals(True)
        combo.clear()
        selected_index = -1
        for profile in profiles:
            model_path = root / f"{profile}_svm.joblib"
            variants = [
                kernel for kernel in ("linear", "rbf")
                if (root / f"{profile}_svm_{kernel}.joblib").is_file()
                or any(root.glob(f"{profile}_svm_{kernel}_b*_norm_*.joblib"))
            ]
            legacy_variants = [
                kernel for kernel in ("linear", "rbf")
                if any(root.glob(f"{profile}_svm_{kernel}_*.joblib"))
            ]
            if variants:
                state = ", ".join(variants).upper()
            elif legacy_variants or model_path.is_file():
                legacy_text = ", ".join(legacy_variants).upper() or "GENERIC"
                state = f"legacy {legacy_text} — canonical model created on next training/use"
            else:
                state = "no trained model — created on first use"
            combo.addItem(profile, str(model_path))
            combo.setItemData(combo.count() - 1, state + ". Exact compatibility is checked below for the current selections.", Qt.ToolTipRole)
            if profile == requested:
                selected_index = combo.count() - 1

        if not profiles:
            combo.addItem("No trained SVM model", "")
            combo.setEnabled(False)
        else:
            combo.setEnabled(True)
            combo.setCurrentIndex(selected_index if selected_index >= 0 else 0)
        combo.blockSignals(False)
        if hasattr(self, '_queue_pp_model_check'):
            self._queue_pp_model_check()

    def _get_selected_postprocessing_profile_paths(self) -> dict[str, Path]:
        combo = getattr(self, "combo_pp_model", None)
        model_value = combo.currentData() if combo is not None else ""
        if not model_value:
            raise FileNotFoundError("No trained SVM model is selected.")

        model_path = Path(str(model_value))
        stem = model_path.stem
        if not stem.endswith("_svm"):
            raise ValueError(f"Invalid base SVM model name: {model_path.name}")
        profile = self._sanitize_roi_profile_name(stem[:-4])
        paths = self._get_roi_profile_paths(profile)
        paths["model"] = model_path
        return paths

    def _read_roi_library_classes(self) -> list[str]:
        """Return class names from the library associated with the selected ROI dataset."""
        lib_path = self._get_roi_profile_paths()["library"]
        if not lib_path.exists():
            self.current_library_norm = None
            self.selected_norm_mode = "none"
            self._set_pp_norm_status("none", None)
            return []

        try:
            data = json.loads(lib_path.read_text(encoding="utf-8"))
        except Exception:
            return []

        if not isinstance(data, dict):
            return []

        return sorted([str(k) for k in data.keys() if str(k) != "_meta"])

    def _refresh_roi_library_view(self, skip_rebuild_checks: bool = False):
        """Refresh the ROI library view from JSON file.
        If skip_rebuild_checks is True, the checkboxes will not be rebuilt (useful when only redrawing).
        Returns :  None."""


        lib_path = self._get_roi_profile_paths()["library"]

        # Si les tabs classe existent, on les efface (sauf "All")
        if self.roi_lib_tabs is not None:
            while self.roi_lib_tabs.count() > 1:
                self.roi_lib_tabs.removeTab(1)

        if not lib_path.exists():
            # vide l'onglet All
            if self.roi_lib_ax is not None:
                ax = self.roi_lib_ax
                ax.clear()
                ax.set_title("Median spectra per class")
                ax.set_xlabel("Band index")
                ax.set_ylabel("Value")
                self.roi_lib_canvas.draw()

                self.roi_lib_canvas.draw()
            if self.roi_lib_table is not None:
                self.roi_lib_table.setRowCount(0)
            # vide les checkboxes
            if hasattr(self, "roi_lib_checks_container") and not skip_rebuild_checks:
                lay = self.roi_lib_checks_container.layout()
                if lay:
                    while lay.count():
                        w = lay.takeAt(0).widget()
                        if w: w.deleteLater()
            return

        # Charge la librairie
        try:
            data = json.loads(lib_path.read_text(encoding="utf-8"))
        except Exception as e:
            QMessageBox.warning(self, "Library", f"Cannot read library JSON:\n{e}")
            return

        # Lire le mode global…
        meta = data.get("_meta", {}) if isinstance(data, dict) else {}
        norm = meta.get("normalization")
        classes = sorted([k for k in data.keys() if k != "_meta"])
        if not norm and classes:
            norm = (data.get(classes[0], {}) or {}).get("normalization", "none")
        if not norm:
            norm = "none"

        # Pixel counts alone can be misleading: expose how many independent
        # selections and cubes contribute to each class when provenance exists.
        provenance_by_class = {}
        wavelengths = meta.get('wavelengths')
        dataset_path = self._get_roi_profile_paths()["dataset"]
        if dataset_path.exists():
            try:
                with np.load(dataset_path, allow_pickle=True) as dataset:
                    if 'wavelengths' in dataset.files:
                        wavelengths = dataset['wavelengths']
                    elif 'meta' in dataset.files:
                        legacy_meta = dataset['meta'].item()
                        if isinstance(legacy_meta,dict):
                            wavelengths = legacy_meta.get('wavelengths',wavelengths)
                    labels = np.asarray(dataset["y"]).astype(str)
                    roi_ids = np.asarray(dataset["roi_id"]) if "roi_id" in dataset.files else None
                    sample_cube_index = (
                        np.asarray(dataset["sample_cube_index"])
                        if "sample_cube_index" in dataset.files else None
                    )
                    for cls in classes:
                        selected = labels == str(cls)
                        provenance_by_class[str(cls)] = {
                            "rois": int(np.unique(roi_ids[selected]).size) if roi_ids is not None else None,
                            "cubes": (
                                int(np.unique(sample_cube_index[selected]).size)
                                if sample_cube_index is not None else None
                            ),
                        }
            except Exception:
                provenance_by_class = {}

        try:
            wavelengths = np.asarray(wavelengths,dtype=float).reshape(-1)
            counts = {len(data[cls].get('median',[])) for cls in classes if data[cls].get('median')}
            if (not counts or counts != {len(wavelengths)} or not np.isfinite(wavelengths).all()
                    or len(np.unique(wavelengths)) != len(wavelengths)):
                wavelengths = None
        except (TypeError,ValueError):
            wavelengths = None
        self.roi_lib_wavelengths = wavelengths

        # Keep the normalization associated with this dataset/profile.
        self.current_library_norm = norm
        self.selected_norm_mode = norm if norm in {mode for _, mode in NORMALIZATION_OPTIONS} else "none"
        if hasattr(self, "combo_norm_mode"):
            idx = self.combo_norm_mode.findData(self.selected_norm_mode)
            if idx >= 0 and self.combo_norm_mode.currentIndex() != idx:
                self.combo_norm_mode.blockSignals(True)
                self.combo_norm_mode.setCurrentIndex(idx)
                self.combo_norm_mode.blockSignals(False)
        self._set_pp_norm_status(self.selected_norm_mode, self.current_library_norm)

        # Pendant la construction de l'onglet ROI, cette fonction peut être appelée
        # avant que les widgets de librairie existent. Dans ce cas, on garde seulement
        # current_library_norm à jour et on sort proprement. Le refresh complet sera
        # refait à la fin de _make_roi_tab().
        if (
            getattr(self, "roi_lib_table", None) is None
            or getattr(self, "roi_lib_ax", None) is None
            or getattr(self, "roi_lib_canvas", None) is None
            or getattr(self, "roi_lib_tabs", None) is None
        ):
            return

        # (Re)crée le tableau global
        self.roi_lib_table.setRowCount(len(classes))
        self.roi_lib_table.setColumnCount(9)
        self.roi_lib_table.setHorizontalHeaderLabels([
            "Class", "Samples", "Kept", "Retention", "ROIs", "Cubes", "Bands",
            "Median range", "Mean σ",
        ])
        self.roi_lib_table.setAlternatingRowColors(True)
        self.roi_lib_table.setStyleSheet("""
        QTableWidget { 
            background:#1f1f1f; 
            alternate-background-color:#2b2b2b;
            color:#eaeaea; 
            gridline-color:#3a3a3a; 
        }
        QHeaderView::section { 
            background:#2b2b2b; 
            color:#eaeaea; 
            border:0px; 
            padding:6px; 
        }
        QTableWidget::item:selected { 
            background-color:#444; 
        }
        """)
        for i, cls in enumerate(classes):
            row = data.get(cls, {})
            n_total = int(row.get("n_total", 0))
            n_kept = int(row.get("n_kept", 0))
            retention = (100.0 * n_kept / n_total) if n_total else 0.0
            median = np.asarray(row.get("median", []), dtype=float)
            std = np.asarray(row.get("std", []), dtype=float)
            finite_median = median[np.isfinite(median)]
            finite_std = std[np.isfinite(std)]
            provenance = provenance_by_class.get(str(cls), {})
            median_range = (
                f"{finite_median.min():.3g} – {finite_median.max():.3g}"
                if finite_median.size else "—"
            )
            mean_std = f"{finite_std.mean():.3g}" if finite_std.size else "—"
            values = [
                str(cls), f"{n_total:,}", f"{n_kept:,}", f"{retention:.1f}%",
                str(provenance.get("rois") if provenance.get("rois") is not None else "—"),
                str(provenance.get("cubes") if provenance.get("cubes") is not None else "—"),
                str(int(median.size)), median_range, mean_std,
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column > 0:
                    item.setTextAlignment(Qt.AlignCenter)
                self.roi_lib_table.setItem(i, column, item)

        header = self.roi_lib_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        self.roi_lib_table.setToolTip(
            "Samples: all selected ROI pixels. Kept: pixels retained after central trim. "
            "ROIs and Cubes: independent selections and source cubes contributing to the class. "
            "Median range: minimum and maximum of the class median spectrum. "
            "Mean σ: average within-class spectral standard deviation."
        )

        # Rebuild checkboxes if needed
        if hasattr(self, "roi_lib_checks_container") and not skip_rebuild_checks:
            lay = self.roi_lib_checks_container.layout()

            while lay.count():
                w = lay.takeAt(0).widget()
                if w: w.deleteLater()

            for cls in classes:
                cb = QCheckBox(cls)
                cb.setChecked(self.roi_lib_visible.get(cls, True))
                cb.stateChanged.connect(lambda state, c=cls: (
                    self._set_roi_lib_visible(c, state == Qt.Checked),
                    self._refresh_roi_library_view(skip_rebuild_checks=True)
                ))
                lay.addWidget(cb)

        # --------- "ALL Tab" ----------
        ax = self.roi_lib_ax
        ax.clear()
        ax.set_title(f"Median spectra per class ({norm})")
        ax.set_xlabel("Wavelength (nm)" if wavelengths is not None else "Band index — wavelength metadata unavailable")
        ax.set_ylabel(f"{norm} value")

        bg = (0.16, 0.16, 0.16); fg = (0.85, 0.85, 0.85)
        ax.set_facecolor((0.18,0.18,0.18))

        ax.figure.subplots_adjust(right=0.80) 
        for spine in ax.spines.values(): spine.set_color((0.5,0.5,0.5))
        ax.tick_params(colors=fg)
        ax.title.set_color(fg)
        ax.xaxis.label.set_color(fg)
        ax.yaxis.label.set_color(fg)

        for cls in classes:
            # n'affiche pas si décoché
            if self.roi_lib_visible.get(cls, True) is False:
                continue
            row = data[cls]
            med = np.array(row.get("median", []), dtype=float)
            std = np.array(row.get("std", []), dtype=float) if "std" in row else None
            if med.size == 0: continue
            x = wavelengths if wavelengths is not None else np.arange(med.size)
            ax.plot(x, med, label=cls, linewidth=2)
            if std is not None and std.size == med.size and med.size > 1:
                ax.fill_between(x, med-std, med+std, alpha=0.18)

        leg = ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9, frameon=False)
        if leg:
            for t in leg.get_texts(): t.set_color("white")
            if leg.get_title(): leg.get_title().set_color("white")

        from gui.inspection_widgets import attach_spectrum_cursor
        attach_spectrum_cursor(self.roi_lib_canvas,ax,wavelength_axis=wavelengths is not None)
        self.roi_lib_canvas.draw()

        # --------- TABS ----------
        for cls in classes:
            row = data[cls]
            med = np.array(row.get("median", []), dtype=float)
            std = np.array(row.get("std", []), dtype=float) if "std" in row else None

            w = QWidget()
            vl = QVBoxLayout(w)
            fig = Figure(figsize=(5,3), dpi=100)
            canvas = FigureCanvas(fig)
            axc = fig.add_subplot(111)
            fig.patch.set_facecolor(bg)
            axc.set_facecolor((0.18,0.18,0.18))
            for s in axc.spines.values(): s.set_color((0.5,0.5,0.5))
            axc.tick_params(colors=fg)
            axc.title.set_color(fg)
            axc.xaxis.label.set_color(fg)
            axc.yaxis.label.set_color(fg)

            axc.set_title(f"{cls} — median ±1σ")
            axc.set_xlabel("Wavelength (nm)" if wavelengths is not None else "Band index — wavelength metadata unavailable")
            axc.set_ylabel(f"{norm} value")


            if med.size > 0:
                x = wavelengths if wavelengths is not None else np.arange(med.size)
                axc.plot(x, med, linewidth=2, label=cls)
                if std is not None and std.size == med.size and med.size > 1:
                    axc.fill_between(x, med-std, med+std, alpha=0.18)

            attach_spectrum_cursor(canvas,axc,wavelength_axis=wavelengths is not None)
            vl.addWidget(canvas)

            t = QTableWidget(1, 9)
            t.setHorizontalHeaderLabels([
                "Class", "Samples", "Kept", "Retention", "ROIs", "Cubes", "Bands",
                "Median range", "Mean σ",
            ])
            t.verticalHeader().setVisible(False)
            t.setStyleSheet("""
            QTableWidget { background:#1f1f1f; color:#eaeaea; gridline-color:#3a3a3a; }
            QHeaderView::section { background:#2b2b2b; color:#eaeaea; border:0px; padding:6px; }
            """)
            n_total = int(row.get("n_total", 0))
            n_kept = int(row.get("n_kept", 0))
            retention = (100.0 * n_kept / n_total) if n_total else 0.0
            finite_med = med[np.isfinite(med)]
            finite_std = std[np.isfinite(std)] if std is not None else np.array([])
            provenance = provenance_by_class.get(str(cls), {})
            values = [
                cls, f"{n_total:,}", f"{n_kept:,}", f"{retention:.1f}%",
                str(provenance.get("rois") if provenance.get("rois") is not None else "—"),
                str(provenance.get("cubes") if provenance.get("cubes") is not None else "—"),
                str(int(med.size)),
                f"{finite_med.min():.3g} – {finite_med.max():.3g}" if finite_med.size else "—",
                f"{finite_std.mean():.3g}" if finite_std.size else "—",
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column > 0:
                    item.setTextAlignment(Qt.AlignCenter)
                t.setItem(0, column, item)
            t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
            t.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
            t.setFixedHeight(60)
            vl.addWidget(t)

            self.roi_lib_tabs.addTab(w, cls)

    def _set_roi_lib_visible(self, cls: str, visible: bool):
        """Update visibility state for class tab in "All" tab.
        Args: 
            cls: class name
            visible: True to show, False to hide
        Returns: 
            None
        """

        self.roi_lib_visible[cls] = visible

    def _append_roi_log(self, text: str):
        """Ajoute du texte au log ROI (garde le scroll en bas)."""
        if not text:
            return
        if getattr(self, "roi_log", None) is None:
            return
        self.roi_log.appendPlainText(text.rstrip())

    def _log_roi(self, msg: str):
        """Ajoute une ligne avec un petit préfixe temporel."""

        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self._append_roi_log(f"[{ts}] {msg}")

    def _get_python64_path(self) -> str:
        """Interpréteur Python 64 bits utilisé pour les scripts externes."""
        return str(self.runtime.python64)

    def _get_roi_script_path(self) -> str:
        """Return the ROI worker launched in the GUI Python environment."""
        base = PROJECT_ROOT / "processing"

        runner_path = base / "roi_selection_runner.py"
        return str(runner_path)

    def _roi_script_has_cli_entrypoint(self, script_path: str) -> bool:
        """Vérifie le lanceur 64 bits; l’implémentation active reste dans processing/pipeline."""
        return Path(script_path).is_file()

    def _run_roi_select_external(
        self,
        *,
        cube_path: str,
        out_dir: str,
        class_names: list[str],
        rgb_image_path: str | None = None,
        dataset_name: str | None = None,
        library_name: str | None = None,
        force_new_dataset: bool = False,
        update_only: bool = False,
        show_error: bool = True,
    ) -> bool:
        """
        Lance le runner de sélection ROI dans Python 64 bits.
        Important: le GUI lance le worker ROI avec l'interpréteur 64 bits configuré.
        """
        PYTHON64 = self._get_python64_path()
        SCRIPT64 = self._get_roi_script_path()

        if not os.path.exists(PYTHON64):
            msg = f"⚠️ Interpreter not found: {PYTHON64}"
            self._log_roi(msg)
            if show_error:
                QMessageBox.warning(self, "ROI", msg)
            return False

        if not os.path.exists(SCRIPT64):
            msg = f"⚠️ ROI script not found: {SCRIPT64}"
            self._log_roi(msg)
            if show_error:
                QMessageBox.warning(self, "ROI", msg)
            return False

        if not self._roi_script_has_cli_entrypoint(SCRIPT64):
            msg = (
                "⚠️ The ROI runner is missing its command-line entry point.\n\n"
                f"Script found: {SCRIPT64}\n\n"
                "Replace the ROI pipeline script or add its command-line entry point."
            )
            self._log_roi(msg)
            if show_error:
                QMessageBox.critical(self, "ROI", msg)
            return False

        self._log_roi(f"Python 64-bit: {PYTHON64}")
        self._log_roi(f"ROI script: {SCRIPT64}")

        profile_paths = self._get_roi_profile_paths()
        dataset_name = dataset_name or profile_paths["dataset"].name
        library_name = library_name or profile_paths["library"].name

        args = [
            PYTHON64,
            "-u",
            SCRIPT64,
            "--cube_path", str(cube_path or ""),
            "--out_dir", str(out_dir),
            "--classes_json", json.dumps(list(class_names), ensure_ascii=False),
            "--keep_frac", "0.95",
            "--sat_thresh", "auto",
            "--dataset_name", str(dataset_name),
            "--library_name", str(library_name),
            "--normalization_mode", str(getattr(self, "selected_norm_mode", "none")),
            "--use_smoothed",
            "--smooth_window", "5",
            "--smooth_poly", "2",
        ]

        if rgb_image_path:
            args += ["--rgb_image_path", str(rgb_image_path)]
        if update_only:
            args += ["--update_only"]
        if force_new_dataset:
            args += ["--force_new_dataset"]

        self._log_roi("Launching external ROI process with Python 64-bit…")

        try:
            self._proc_roi = Popen(args, stdout=PIPE, stderr=STDOUT, text=True, encoding="utf-8", errors="replace", env=worker_environment(), cwd=str(PROJECT_ROOT))
            proc_tail = []

            while self._proc_roi is not None:
                line = self._proc_roi.stdout.readline()
                if not line:
                    if self._proc_roi.poll() is not None:
                        break
                    QApplication.processEvents()
                    continue

                text = line.rstrip()
                if text:
                    print(text)
                    self._append_roi_log(text)
                    proc_tail.append(text)
                    proc_tail = proc_tail[-30:]
                    QApplication.processEvents()

            return_code = self._proc_roi.returncode if self._proc_roi is not None else -1
            self._proc_roi = None

            if return_code != 0:
                details = "\n".join(proc_tail[-12:])
                msg = f"External ROI process failed with return code {return_code}."
                if details:
                    msg += "\n\nDernières lignes du processus ROI:\n" + details
                self._log_roi(msg)
                if show_error:
                    QMessageBox.critical(self, "ROI", msg)
                return False

            return True

        except Exception as e:
            self._proc_roi = None
            msg = f"Error launching external ROI process: {e}"
            self._log_roi(msg)
            if show_error:
                QMessageBox.critical(self, "ROI", msg)
            return False

    def _get_svm_script_path(self) -> str:
        base = PROJECT_ROOT / "processing"
        return str(base / "svm_classifier_runner.py")

    def _run_svm_training_external(
        self,
        *,
        dataset_path: str | Path,
        model_path: str | Path,
        normalization_mode: str,
        show_error: bool = True,
    ) -> bool:
        """Retrain the base SVM associated with the selected ROI dataset."""
        python64 = self._get_python64_path()
        script64 = self._get_svm_script_path()
        dataset_path = Path(dataset_path)
        model_path = Path(model_path)

        if not dataset_path.is_file():
            msg = f"ROI dataset not found: {dataset_path}"
            self._log_roi(msg)
            if show_error:
                QMessageBox.warning(self, "SVM training", msg)
            return False
        if not os.path.isfile(script64):
            msg = f"SVM script not found: {script64}"
            self._log_roi(msg)
            if show_error:
                QMessageBox.warning(self, "SVM training", msg)
            return False

        args = [
            python64,
            "-u",
            script64,
            "--train-only",
            "--dataset_path", str(dataset_path),
            "--model_path", str(model_path),
            "--normalization_mode", str(normalization_mode),
            "--svm_type", "linear",
            "--force_retrain",
        ]
        if self.roi_enable_platt.isChecked(): args.append("--platt")
        profile = model_path.stem[:-4] if model_path.stem.endswith("_svm") else model_path.stem
        bands_path = model_path.parent / f"{profile}_svm_bands.json"
        if bands_path.is_file():
            try:
                selected_bands = [float(value) for value in json.loads(
                    bands_path.read_text(encoding="utf-8")
                ).get("selected_wavelengths_nm", [])]
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                msg = f"Invalid SVM band profile {bands_path.name}: {exc}"
                self._log_roi(msg)
                if show_error:
                    QMessageBox.warning(self, "SVM bands", msg)
                return False
            if selected_bands:
                args += ["--selected_wavelengths_nm", *[f"{value:g}" for value in selected_bands]]

        self._log_roi(f"Retraining associated SVM: {model_path.name}")
        try:
            proc = Popen(args, stdout=PIPE, stderr=STDOUT, text=True, encoding="utf-8", errors="replace", env=worker_environment(), cwd=str(PROJECT_ROOT))
            tail = []
            while True:
                line = proc.stdout.readline()
                if not line:
                    if proc.poll() is not None:
                        break
                    QApplication.processEvents()
                    continue
                line = line.rstrip()
                if line:
                    print(line)
                    self._append_roi_log(line)
                    tail.append(line)
                    tail = tail[-25:]
                    QApplication.processEvents()

            if proc.returncode != 0:
                msg = f"SVM training failed with return code {proc.returncode}."
                if tail:
                    msg += "\n\n" + "\n".join(tail[-10:])
                self._log_roi(msg)
                if show_error:
                    QMessageBox.critical(self, "SVM training", msg)
                return False

            self._log_roi(f"SVM model updated: {model_path}")
            self._refresh_svm_model_choices(select_profile=model_path.stem[:-4])
            return True
        except Exception as exc:
            msg = f"Could not launch SVM training: {exc}"
            self._log_roi(msg)
            if show_error:
                QMessageBox.critical(self, "SVM training", msg)
            return False

    def _browse_corrected_file(self):

            start_dir = self._envi_browse_directory()
            os.makedirs(start_dir, exist_ok=True)

            path, _ = QFileDialog.getOpenFileName(
                self,
                "Choisir un cube ENVI",
                start_dir,
                "ENVI cubes (*.hdr *.raw *.img *.dat *.bin)"
            )
            if path:
                try:
                    path = self._register_cube_source(path)
                except Exception as exc:
                    QMessageBox.warning(self, "Import", str(exc))
                    return
                self.corrected_file_input.setText(path)
                self.corrected_file_label.setText(os.path.basename(path))
                self.corrected_cube_path = path
                self._refresh_pp_cube_choices(selected_path=path)

    def _make_roi_tab(self) -> QWidget:
        tab = QWidget()
        root = QVBoxLayout(tab)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        title = QLabel("<b>Select reference image for ROI selection</b>")
        root.addWidget(title)

        # --- Ligne de sélection ---
        row = QHBoxLayout()
        row.addWidget(QLabel("File :"))

        # Chemin complet (lecture seule)
        self.corrected_file_input = QLineEdit()
        self.corrected_file_input.setReadOnly(True)
        row.addWidget(self.corrected_file_input, 1)

        # Bouton parcourir
        btn = QPushButton("Browse…")
        btn.clicked.connect(self._browse_corrected_file)
        row.addWidget(btn)
        root.addLayout(row)

        # Nom de fichier (base name) en gros
        self.corrected_file_label = QLabel("No file selected")
        root.addWidget(self.corrected_file_label)

        # --- Dataset/profile selection ---
        dataset_row = QHBoxLayout()
        dataset_row.addWidget(QLabel("<b>ROI dataset:</b>"))
        self.combo_roi_dataset = QComboBox()
        self.combo_roi_dataset.setEditable(True)
        self.combo_roi_dataset.setInsertPolicy(QComboBox.NoInsert)
        self.combo_roi_dataset.setToolTip(
            "Select an existing ROI dataset or type a new profile name. "
            "The library and SVM model use the same profile name."
        )
        self.btn_refresh_roi_datasets = QPushButton("Refresh datasets")
        dataset_row.addWidget(self.combo_roi_dataset, 1)
        dataset_row.addWidget(self.btn_refresh_roi_datasets)
        root.addLayout(dataset_row)

        def _activate_roi_profile():
            profile = self._sanitize_roi_profile_name(self.combo_roi_dataset.currentText())
            self.combo_roi_dataset.setEditText(profile)
            self.roi_dataset_profile = profile
            self._refresh_roi_library_view()
            self._sync_roi_dataset_classes()
            self._refresh_svm_model_choices(select_profile=profile)

        self.combo_roi_dataset.activated.connect(lambda _idx: _activate_roi_profile())
        if self.combo_roi_dataset.lineEdit() is not None:
            self.combo_roi_dataset.lineEdit().editingFinished.connect(_activate_roi_profile)
        self.btn_refresh_roi_datasets.clicked.connect(
            lambda: self._refresh_roi_dataset_choices(self.combo_roi_dataset.currentText())
        )

        self.roi_enable_platt = QCheckBox("Enable Platt scaling for new / retrained models")
        self.roi_enable_platt.setToolTip("Fits sigmoid calibration on cube-held-out scores. Requires multiple annotated cubes; training fails explicitly if calibration cannot be fitted.")
        root.addWidget(self.roi_enable_platt)
        # --- Mode d'écriture du dataset (Append / Force new) ---
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("<b>Dataset mode:</b>"))

        self.btn_toggle_dataset_mode = QPushButton("Append")
        self.btn_toggle_dataset_mode.setCheckable(True)
        self.btn_toggle_dataset_mode.setToolTip(
            "Click to toggle between appending to the existing ROI dataset or forcing a new dataset file."
        )

        # ---- Normalization mode ----
        # état par défaut si pas encore présent
        self.selected_norm_mode = getattr(self, "selected_norm_mode", "none")

        norm_row = QHBoxLayout()
        norm_row.addWidget(QLabel("Normalization:"))

        self.combo_norm_mode = QComboBox()

        # IMPORTANT: userData doit matcher EXACTEMENT normalize_rows(...):
        for label, mode in NORMALIZATION_OPTIONS:
            self.combo_norm_mode.addItem(label, userData=mode)


        # sélectionne l’état courant
        for i in range(self.combo_norm_mode.count()):
            if self.combo_norm_mode.itemData(i) == self.selected_norm_mode:
                self.combo_norm_mode.setCurrentIndex(i)
                break

        btn_refresh_lib = QPushButton("Refresh")
        btn_refresh_lib.setToolTip("Re-read and redraw the library view")

        norm_row.addWidget(self.combo_norm_mode)
        norm_row.addStretch(1)
        norm_row.addWidget(btn_refresh_lib)
        root.addLayout(norm_row)

        def _clean_mode(s: str) -> str:
            return str(s).strip().lower().replace("`", "")

        def _apply_norm_mode(mode: str, *, show_error: bool = True):
            # 1) mémorise le mode (propre)
            mode = _clean_mode(mode)
            self.selected_norm_mode = mode

            # 2) renormalise immédiatement la librairie existante (sans ajouter de nouvelles ROIs)
            out_dir = Path(self.roi_out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            cube_path = getattr(self, "corrected_cube_path", "").strip() or self.corrected_file_input.text().strip()
            if not cube_path or not Path(cube_path).exists():
                cube_path = ""  # ok: update_only doit agir sur la librairie déjà sur disque

            class_names = [name for name, cb in self.roi_class_checks.items() if cb.isChecked()]
            profile_paths = self._get_roi_profile_paths()
            if not profile_paths["dataset"].is_file():
                # Selecting preprocessing before the first ROI is a valid action.
                # There is no library/model to regenerate until annotations exist.
                self._log_roi(f"Normalization selected: {mode}. Add ROIs to create this dataset before training.")
                self._set_pp_norm_status(mode, None)
                self.pp_svm_training_status.setText("SVM training: waiting for ROI dataset")
                return


            ok = self._run_roi_select_external(
                cube_path=cube_path,
                out_dir=str(out_dir),
                class_names=class_names,
                dataset_name=profile_paths["dataset"].name,
                library_name=profile_paths["library"].name,
                update_only=True,
                show_error=show_error,
            )

            if ok:
                if hasattr(self, "pp_norm_label"):
                    self.pp_norm_label.setText(f"Normalization: {mode}")
                self._refresh_roi_library_view()
                self._set_pp_norm_status(self.selected_norm_mode, getattr(self, "current_library_norm", None))
                self._log_roi(f"Library re-normalized with mode: {self.selected_norm_mode}")

                # The model preprocessing changed, so retrain the associated SVM.
                if hasattr(self, "log_output"):
                    self.log_output.appendPlainText(
                        f"SVM training in progress: {profile_paths['model'].name} (normalization: {mode})"
                    )
                    self.log_output.verticalScrollBar().setValue(
                        self.log_output.verticalScrollBar().maximum()
                    )
                if hasattr(self, "pp_svm_training_status"):
                    self.pp_svm_training_status.setText("SVM training: in progress…")
                QApplication.processEvents()
                trained = self._run_svm_training_external(
                    dataset_path=profile_paths["dataset"],
                    model_path=profile_paths["model"],
                    normalization_mode=mode,
                    show_error=show_error,
                )
                if hasattr(self, "log_output"):
                    self.log_output.appendPlainText(
                        f"SVM training {'completed' if trained else 'failed'}: {profile_paths['model'].name}"
                    )
                    self.log_output.verticalScrollBar().setValue(
                        self.log_output.verticalScrollBar().maximum()
                    )
                if hasattr(self, "pp_svm_training_status"):
                    self.pp_svm_training_status.setText(
                        "SVM training: completed" if trained else "SVM training: failed"
                    )
                if not trained and hasattr(self, "pp_norm_features_lbl"):
                    self.pp_norm_features_lbl.setText("SVM training failed — retraining required before classification")
                    self.pp_norm_box.setStyleSheet("QFrame#pp_norm_box { border: 2px solid #c62828; }")
            else:
                self.selected_norm_mode = getattr(self, "current_library_norm", None) or "none"
                self._set_pp_norm_status(self.selected_norm_mode, getattr(self, "current_library_norm", None))
                self._log_roi(f"Library normalization failed (requested mode={mode})")

        def _apply_shared_normalization(mode, *, show_error=True):
            if getattr(self, "_normalization_busy", False):
                return
            self._normalization_busy = True
            controls = [getattr(self, name, None) for name in (
                "combo_norm_mode", "combo_pp_normalization", "combo_roi_dataset",
                "combo_pp_model", "btn_run_pp", "btn_refresh_pp_models", "btn_refresh_roi_datasets")]
            states = [(widget, widget.isEnabled()) for widget in controls if widget is not None]
            try:
                for widget, _enabled in states:
                    widget.setEnabled(False)
                _apply_norm_mode(mode, show_error=show_error)
            finally:
                self._normalization_busy = False
                for widget, enabled in states:
                    widget.setEnabled(enabled)

        self._apply_roi_normalization = _apply_shared_normalization

        def _on_norm_changed(_idx: int):
            mode = self.combo_norm_mode.currentData() or self.combo_norm_mode.currentText()
            self._apply_roi_normalization(mode, show_error=True)

        # --- 1) Mettre le mode par défaut à "none" À L'INITIALISATION du tab ---
        # (à placer après addItem(...) sur combo_norm_mode)
        idx_none = self.combo_norm_mode.findData("none")
        if idx_none >= 0:
            self.combo_norm_mode.blockSignals(True)
            self.combo_norm_mode.setCurrentIndex(idx_none)
            self.combo_norm_mode.blockSignals(False)
            self.selected_norm_mode = "none"
            if hasattr(self, "pp_norm_label"):
                self.pp_norm_label.setText("Normalization: none")

        # --- 2) Connect ---
        self.combo_norm_mode.currentIndexChanged.connect(_on_norm_changed)
        btn_refresh_lib.clicked.connect(lambda: self._refresh_roi_library_view())

        # Do not modify or retrain a dataset merely by opening the GUI.
        # The selected profile is updated only after an explicit normalization change
        # or a completed ROI selection.





        def _apply_mode_style(is_append: bool):
            if is_append:
                self.btn_toggle_dataset_mode.setText("Append")
                self.btn_toggle_dataset_mode.setChecked(False)
                self.btn_toggle_dataset_mode.setStyleSheet(
                    "QPushButton { background-color:#2e7d32; color:white; border-radius:8px; padding:6px 12px; }"
                    "QPushButton:hover { background-color:#388e3c; }"
                )
            else:
                self.btn_toggle_dataset_mode.setText("Force new dataset")
                self.btn_toggle_dataset_mode.setChecked(True)
                self.btn_toggle_dataset_mode.setStyleSheet(
                    "QPushButton { background-color:#c62828; color:white; border-radius:8px; padding:6px 12px; }"
                    "QPushButton:hover { background-color:#d32f2f; }"
                )

        def _toggle_dataset_mode_clicked():
            self.roi_append_mode = not self.roi_append_mode
            _apply_mode_style(self.roi_append_mode)

        self.btn_toggle_dataset_mode.clicked.connect(_toggle_dataset_mode_clicked)
        _apply_mode_style(self.roi_append_mode)

        mode_row.addWidget(self.btn_toggle_dataset_mode)
        mode_row.addStretch(1)
        root.addLayout(mode_row)

        # ---- Classe(s) à annoter : GroupBox avec cases à cocher ----
        grp_classes = QGroupBox("Select ROI for:")
        classes_row = QHBoxLayout()
        self._roi_annotation_classes_layout = classes_row

        def _add_class_checkbox(name: str, checked: bool = True):
            key = name.strip()
            if not key:
                return
            exists = [k for k in self.roi_class_checks.keys() if k.lower() == key.lower()]
            if exists:
                self.roi_class_checks[exists[0]].setChecked(True)
                return
            cb = QCheckBox(key)
            cb.setChecked(checked)
            self.roi_class_checks[key] = cb
            classes_row.addWidget(cb)

        for default_cls in ["material_1", "material_2", "background"]:
            _add_class_checkbox(default_cls, checked=True)

        btn_other = QPushButton("Other…")
        btn_other.setToolTip("Add another class label")
        def _on_add_other():
            from PyQt5.QtWidgets import QInputDialog
            text, ok = QInputDialog.getText(self, "New class", "Enter class name:")
            if ok and text.strip():
                _add_class_checkbox(text.strip(), checked=True)

        btn_other.clicked.connect(_on_add_other)
        classes_row.addWidget(btn_other)
        classes_row.addStretch(1)
        grp_classes.setLayout(classes_row)
        root.addWidget(grp_classes)

        # --- Lancement sélection ROI ---
        run_row = QHBoxLayout()
        btn_run_roi = QPushButton("Run ROI selection")
        btn_run_roi.setToolTip("Generate an RGB preview from the ENVI cube, open ROI selection and update the dataset/library")

        def _on_run_roi(_checked: bool = False):
            # 1) Classes cochées
            class_names = [name for name, cb in self.roi_class_checks.items() if cb.isChecked()]
            if not class_names:
                QMessageBox.information(self, "ROI", "Please check at least one class in 'Select ROI for:'.")
                return

            # 2) Chemin cube EXACTEMENT celui du Browse…
            cube_path = getattr(self, "corrected_cube_path", "").strip()
            if not cube_path:
                cube_path = self.corrected_file_input.text().strip()

            cube_path_p = Path(cube_path)
            if not cube_path or not cube_path_p.exists():
                QMessageBox.warning(self, "ROI", "Please select an ENVI cube (.hdr and its binary) with the Browse button.")
                return

            # The ROI worker generates its RGB directly from the selected ENVI.
            rgb_image_path = None

            # 3) Dossier de sortie GLOBAL (indépendant des cubes)
            out_dir = Path(self.roi_out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            # 4) Selected dataset/profile and associated files
            profile = self._sanitize_roi_profile_name(self.combo_roi_dataset.currentText())
            self.combo_roi_dataset.setEditText(profile)
            self.roi_dataset_profile = profile
            profile_paths = self._get_roi_profile_paths(profile)
            dataset_path = profile_paths["dataset"]
            library_path = profile_paths["library"]
            model_path = profile_paths["model"]

            # Back up the currently selected profile before modifying it.
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            for saved_path in (dataset_path, library_path):
                if saved_path.exists():
                    backup = saved_path.with_name(saved_path.stem + f"_backup_{ts}" + saved_path.suffix)
                    shutil.copy2(saved_path, backup)

            # 5) Append / Force new applies only to the selected profile.
            append_mode = getattr(self, "roi_append_mode", True)

            # 6) ROI selection, dataset/library update and automatic SVM retraining.
            self.roi_log.clear()
            self._log_roi("——— ROI selection started ———")
            self._log_roi(f"Profile: {profile}")
            self._log_roi(f"Mode: {'Append' if append_mode else 'Force new'}")
            self._log_roi(f"Cube: {str(cube_path_p)}")
            self._log_roi(f"Dataset: {dataset_path}")
            self._log_roi(f"Library: {library_path}")
            self._log_roi(f"Model: {model_path}")
            self._log_roi(f"Classes: {', '.join(class_names)}")
            self._log_roi(f"ROI display image: {rgb_image_path if rgb_image_path else 'NOT FOUND (fallback to grayscale)'}")

            ok = self._run_roi_select_external(
                cube_path=str(cube_path_p),
                out_dir=str(out_dir),
                class_names=class_names,
                rgb_image_path=rgb_image_path,
                dataset_name=dataset_path.name,
                library_name=library_path.name,
                force_new_dataset=not append_mode,
                update_only=False,
                show_error=True,
            )

            if ok:
                svm_ok = self._run_svm_training_external(
                    dataset_path=dataset_path,
                    model_path=model_path,
                    normalization_mode=self.selected_norm_mode,
                    show_error=True,
                )
                self._refresh_roi_dataset_choices(select_profile=profile)
                self._refresh_svm_model_choices(select_profile=profile)
                self._refresh_roi_library_view()
                self._log_roi("——— ROI selection finished ———")
                lib_classes = self._read_roi_library_classes()

                model_status = str(model_path) if svm_ok else "training failed"
                QMessageBox.information(
                    self, "ROI",
                    "ROI selection finished.\n\n"
                    f"Dataset: {dataset_path}\n"
                    f"Library: {library_path}\n"
                    f"SVM: {model_status}\n"
                    f"Classes: {', '.join(lib_classes) or 'none'}"
                )
            else:
                self._log_roi("——— ROI selection failed ———")




        run_row.addStretch(1)
        run_row.addWidget(btn_run_roi)
        run_row.addStretch(1)
        btn_run_roi.clicked.connect(_on_run_roi)
        root.addLayout(run_row)

        # ---------- Library (SNV, trim 80%) ----------
        grp_lib = QGroupBox("Library (SNV, trim 80%)")
        lib_col = QVBoxLayout(grp_lib)
        grp_lib.setStyleSheet("""
                                QGroupBox {
                                border: 1px solid #3a3a3a;
                                border-radius: 8px;
                                margin-top: 24px;         /* <-- espace réservé au titre */
                                }
                                QGroupBox::title {
                                subcontrol-origin: margin;
                                left: 10px;
                                padding: 0 6px;
                                color: #dddddd;
                                font-weight: 600;
                                }
                        """)


        # Onglets : "All" + 1 onglet par classe
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.setStyleSheet("""
                                QTabWidget::pane { border: 0px; }
                                QTabBar::tab {
                                background: #2b2b2b;
                                color: #ddd;
                                padding: 6px 10px;
                                margin-top: 4px;         /* <-- espace sous la ligne du titre */
                                border: 1px solid #3a3a3a;
                                }
                                QTabBar::tab:selected {
                                background: #353535;
                                }
                            """)

        self.roi_lib_tabs = tabs
        btn_enlarge_roi = QPushButton('Enlarge graph…')
        btn_enlarge_roi.setToolTip('Open the graph of the selected class in a resizable window with zoom and pan controls.')
        def _enlarge_roi_graph():
            from gui.inspection_widgets import enlarge_plot
            page = self.roi_lib_tabs.currentWidget()
            canvas = page.findChild(FigureCanvas) if page is not None else None
            if canvas is not None:
                enlarge_plot(canvas, self)
        btn_enlarge_roi.clicked.connect(_enlarge_roi_graph)
        lib_col.addWidget(btn_enlarge_roi)
        lib_col.addWidget(tabs)

        # --- Onglet "All" (overview) : graphe à gauche, checkboxes à droite
        w_all = QWidget()
        all_layout = QVBoxLayout(w_all)
        all_layout.setContentsMargins(0, 0, 0, 0)
        all_layout.setSpacing(8)
        hl_all = QHBoxLayout()
        hl_all.setContentsMargins(0, 0, 0, 0)
        hl_all.setSpacing(12)
        all_layout.addLayout(hl_all, 1)

        # ---- Colonne gauche : Figure (prend tout l'espace)
        left = QVBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(6)

        fig_all = Figure(figsize=(5, 3), dpi=100)
        canvas_all = FigureCanvas(fig_all)
        ax_all = fig_all.add_subplot(111)
        ax_all.set_title("Median spectra per class (SNV)", pad=20)
        ax_all.set_xlabel("Band index")
        ax_all.set_ylabel("SNV value")

        # couleurs de fond/axes pour thème sombre
        bg = (0.16, 0.16, 0.16); fg = (0.85, 0.85, 0.85)
        fig_all.patch.set_facecolor(bg)
        ax_all.set_facecolor((0.18, 0.18, 0.18))
        for spine in ax_all.spines.values(): spine.set_color((0.5, 0.5, 0.5))
        ax_all.tick_params(colors=fg)
        ax_all.title.set_color(fg)
        ax_all.xaxis.label.set_color(fg)
        ax_all.yaxis.label.set_color(fg)

        # le canvas doit s'étirer
        from PyQt5.QtWidgets import QSizePolicy
        canvas_all.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        canvas_all.setMinimumHeight(240)
        canvas_all.setMaximumHeight(360)
        left.addWidget(canvas_all, 1)

        # (optionnel) petite barre d'actions sous le graphe
        row_refresh = QHBoxLayout()
        btn_refresh = QPushButton("Refresh")
        row_refresh.addStretch(1); row_refresh.addWidget(btn_refresh)
        left.addLayout(row_refresh)

        tbl = QTableWidget(0, 9)
        tbl.setHorizontalHeaderLabels([
            "Class", "Samples", "Kept", "Retention", "ROIs", "Cubes", "Bands",
            "Median range", "Mean σ",
        ])
        tbl.verticalHeader().setVisible(False)
        tbl.setMinimumWidth(620)
        tbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        tbl.setStyleSheet("""
        QTableWidget {
            background:#1f1f1f; alternate-background-color:#292929;
            color:#eaeaea; gridline-color:#3a3a3a;
        }
        QHeaderView::section { background:#2b2b2b; color:#eaeaea; border:0px; padding:6px; }
        """)
        tbl.setAlternatingRowColors(True)

        # mémorise la référence
        self.roi_lib_table = tbl

        # ---- Log ROI (sous la table) ----
        self.roi_log = QPlainTextEdit()
        self.roi_log.setReadOnly(True)
        self.roi_log.setMaximumBlockCount(2000)  # rotation auto si très long
        mono = QFont("Consolas")
        mono.setPointSize(10)
        self.roi_log.setFont(mono)
        self.roi_log.setStyleSheet("""
        QPlainTextEdit {
        background: #161616;
        color: #eaeaea;
        border: 1px solid #3a3a3a;
        border-radius: 6px;
        }
        """)
        self.roi_log.setFixedHeight(140)  # ajuste la hauteur si tu veux
        all_layout.addWidget(self.roi_log)


        # ---- Colonne droite : checkboxes
        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(6)
        right.addWidget(tbl, 1)

        self.roi_lib_checks = {}  # {classe: QCheckBox}
        self.roi_lib_checks_container = QGroupBox("Displayed classes")
        self.roi_lib_checks_container.setStyleSheet("""
        QGroupBox {
        border: 1px solid #3a3a3a; border-radius: 8px; margin-top: 0px; color:#ddd;
        }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; }
        """)
        checks_layout = QVBoxLayout(self.roi_lib_checks_container)
        checks_layout.setContentsMargins(8, 22, 8, 8)
        checks_layout.setSpacing(4)

        # si tu veux un scroll quand il y a beaucoup de classes :
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        wrap = QWidget()
        wrap_layout = QVBoxLayout(wrap)
        wrap_layout.setContentsMargins(0,0,0,0)
        wrap_layout.addWidget(self.roi_lib_checks_container)
        wrap_layout.addStretch(1)
        scroll.setWidget(wrap)
        scroll.setMaximumHeight(150)
        left.addWidget(scroll, 0)

        # ---- Assemblage HBox
        hl_all.addLayout(left, 2)
        hl_all.addLayout(right, 3)

        # mémos
        self.roi_lib_canvas = canvas_all
        self.roi_lib_ax = ax_all

        btn_refresh.clicked.connect(self._refresh_roi_library_view)

        tabs.addTab(w_all, "All")


        # mémos
        self.roi_lib_canvas = canvas_all
        self.roi_lib_ax = ax_all

        btn_refresh.clicked.connect(self._refresh_roi_library_view)

        root.addWidget(grp_lib)

        # Maintenant que les widgets ROI Library existent, on peut rafraîchir
        # l'affichage sans risquer roi_lib_table == None.
        self._refresh_roi_library_view()

        self._sync_roi_dataset_classes()

        root.addStretch(1)
        return tab
