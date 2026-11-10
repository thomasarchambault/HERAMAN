# Rôle : Construire l’onglet de gestion des jeux de données, entraînements et modèles SVM.

from __future__ import annotations

import json
import re
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.exceptions import InconsistentVersionWarning
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import QProcess, Qt, QTimer
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QInputDialog,
    QMessageBox, QPlainTextEdit, QPushButton, QSizePolicy, QSplitter,
    QLineEdit, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from shared.process_io import utf8_decoder
from shared.paths import PROJECT_ROOT, resolve_stored_path, resolve_project_path
from gui.features.svm_health import SvmHealthWidget
from gui.features.svm_calibration import SvmCalibrationWidget

SVM_DASHBOARD_COLUMNS = [
    "Specimen profile", "Classes used", "Kernel", "Parameters",
    "Spectral bands", "Normalization", "Dataset", "CV macro-F1",
]


class SvmModelsMixin:
    """Read-only model dashboard plus targeted asynchronous retraining."""

    def _make_svm_models_tab(self) -> QWidget:
        tab = QWidget()
        root = QVBoxLayout(tab)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        toolbar = QHBoxLayout()
        title = QLabel("<b>SVM model dashboard</b>")
        explanation = QLabel(
            "One visible row per dataset. Double-click its name to expand or collapse its saved model variants. "
            "Double-click a column header to sort all variants; repeat to reverse the order."
        )
        explanation.setStyleSheet("color:#b8b8b8;")
        explanation.setWordWrap(True)
        self.btn_refresh_svm_dashboard = QPushButton("Refresh")
        toolbar.addWidget(title)
        toolbar.addWidget(explanation, 1)
        toolbar.addStretch(1)
        toolbar.addWidget(self.btn_refresh_svm_dashboard)
        root.addLayout(toolbar)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, 1)

        model_panel = QGroupBox("Available SVM models")
        model_panel.setMinimumWidth(390)
        model_layout = QVBoxLayout(model_panel)
        self._make_svm_dashboard_filters(model_layout)
        self.svm_dashboard_table = QTableWidget(0, len(SVM_DASHBOARD_COLUMNS))
        self.svm_dashboard_table.setHorizontalHeaderLabels(SVM_DASHBOARD_COLUMNS)
        self.svm_dashboard_table.verticalHeader().setVisible(False)
        self.svm_dashboard_table.setAlternatingRowColors(True)
        self.svm_dashboard_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.svm_dashboard_table.setSelectionMode(QTableWidget.SingleSelection)
        self.svm_dashboard_table.setEditTriggers(QTableWidget.NoEditTriggers)
        header = self.svm_dashboard_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionsClickable(True)
        header.sectionDoubleClicked.connect(self._sort_svm_dashboard)
        header.setToolTip("Double-click to sort all matching models. Double-click again to reverse. Missing scores are always last.")
        self.svm_dashboard_table.setStyleSheet("""
            QTableWidget { background:#1f1f1f; alternate-background-color:#292929;
                           color:#eaeaea; gridline-color:#3a3a3a; }
            QHeaderView::section { background:#2b2b2b; color:#eaeaea;
                                   border:0; padding:6px; }
            QTableWidget::item:selected { background:#43566e; }
        """)
        model_layout.addWidget(self.svm_dashboard_table)
        self.btn_toggle_svm_variants = QPushButton("Expand / collapse selected dataset")
        model_layout.addWidget(self.btn_toggle_svm_variants)
        self.btn_toggle_svm_variants.clicked.connect(self._toggle_svm_dataset_variants)
        self.svm_dashboard_table.itemDoubleClicked.connect(
            lambda item: self._toggle_svm_dataset_variants() if item.column() == 0 else None)
        splitter.addWidget(model_panel)

        details = QWidget()
        details_layout = QVBoxLayout(details)
        details_layout.setContentsMargins(0, 0, 0, 0)
        detail_tabs = QTabWidget()
        detail_tabs.setDocumentMode(True)
        details_layout.addWidget(detail_tabs)

        summary_group = QGroupBox("Selected model")
        summary_grid = QGridLayout(summary_group)
        self.svm_dashboard_labels = {}
        fields = [
            ("file", "File"), ("dataset", "Dataset"), ("kernel", "Kernel"),
            ("params", "Parameters"), ("normalization", "Training normalization"),
            ("samples", "Training pixels"),
            ("classes", "Classes"), ("cubes", "Model training cubes"),
            ("bands", "Spectral bands"), ("pca", "PCA"),
            ("cv", "Grouped validation"), ("updated", "Modified"),
            ("calibration", "Probability calibration"),
            ("compatibility", "Compatibility"),
            ("dataset_status", "Dataset status"),
        ]
        for row, (key, caption) in enumerate(fields):
            caption_label = QLabel(f"<b>{caption}:</b>")
            value_label = QLabel("—")
            value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            value_label.setWordWrap(True)
            summary_grid.addWidget(caption_label, row, 0, Qt.AlignTop)
            summary_grid.addWidget(value_label, row, 1)
            self.svm_dashboard_labels[key] = value_label
        self.svm_class_distribution_table = QTableWidget(0, 5)
        self.svm_class_distribution_table.setHorizontalHeaderLabels(
            ["Class", "Pixels", "Share", "ROIs", "Cubes"]
        )
        self.svm_class_distribution_table.verticalHeader().setVisible(False)
        self.svm_class_distribution_table.setAlternatingRowColors(True)
        self.svm_class_distribution_table.setMinimumHeight(150)
        self.svm_class_distribution_table.setStyleSheet("""
            QTableWidget { background:#1f1f1f; alternate-background-color:#292929;
                           color:#eaeaea; gridline-color:#3a3a3a; }
            QHeaderView::section { background:#2b2b2b; color:#eaeaea;
                                   border:0; padding:6px; }
        """)
        class_header = self.svm_class_distribution_table.horizontalHeader()
        class_header.setSectionResizeMode(QHeaderView.ResizeToContents)
        class_header.setSectionResizeMode(0, QHeaderView.Stretch)
        self.svm_class_distribution_caption = QLabel("<b>Current dataset distribution:</b>")
        self.svm_class_distribution_caption.setToolTip(
            "This table describes the ROI dataset currently on disk, which may be newer than the selected model."
        )
        summary_grid.addWidget(self.svm_class_distribution_caption, len(fields), 0, Qt.AlignTop)
        summary_grid.addWidget(self.svm_class_distribution_table, len(fields), 1)
        summary_grid.setColumnStretch(1, 1)
        detail_tabs.addTab(summary_group, "Overview")

        validation_tabs = QTabWidget()
        matrix_group = QGroupBox("Cube-grouped cross-validation confusion matrix")
        matrix_layout = QVBoxLayout(matrix_group)
        self.svm_confusion_figure = Figure(figsize=(5, 3), dpi=100)
        self.svm_confusion_canvas = FigureCanvas(self.svm_confusion_figure)
        self.svm_confusion_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.svm_confusion_canvas.setMinimumHeight(260)
        matrix_layout.addWidget(self.svm_confusion_canvas)
        self.svm_validation_caption = QLabel("Select a model to view its evaluation protocol.")
        self.svm_validation_caption.setWordWrap(True)
        matrix_layout.insertWidget(0, self.svm_validation_caption)
        validation_tabs.addTab(matrix_group, "Classification metrics")
        self.svm_calibration = SvmCalibrationWidget()
        validation_tabs.addTab(self.svm_calibration, "Platt calibration")
        detail_tabs.addTab(validation_tabs, "Validation")

        self.svm_visualization = SvmHealthWidget()
        detail_tabs.addTab(self.svm_visualization, "Visualization")

        retrain_group = QGroupBox("Retrain selected model family")
        retrain_layout = QGridLayout(retrain_group)
        self.combo_svm_retrain_kernel = QComboBox()
        self.combo_svm_retrain_kernel.addItem("Linear", "linear")
        self.combo_svm_retrain_kernel.addItem("RBF", "rbf")
        self.btn_configure_svm_bands = QPushButton("Select spectral bands…")
        self.svm_retrain_bands = QLabel("Bands: all dataset wavelengths")
        self.svm_retrain_bands.setWordWrap(True)
        self.svm_retrain_band_weights = QLineEdit()
        self.svm_retrain_band_weights.setPlaceholderText("Optional: 620=1.5, 680=0.7")
        self.svm_retrain_band_weights.setToolTip(
            "Weight selected wavelengths as nm=factor. Unlisted wavelengths keep weight 1. "
            "Weights must be greater than zero and are applied after robust scaling, before PCA."
        )
        self.svm_retrain_cubes_table = QTableWidget(0, 3)
        self.svm_retrain_cubes_table.setHorizontalHeaderLabels(["Use", "Cube", "Labeled pixels"])
        self.svm_retrain_cubes_table.verticalHeader().setVisible(False)
        self.svm_retrain_cubes_table.setAlternatingRowColors(True)
        self.svm_retrain_cubes_table.setMinimumHeight(115)
        self.svm_retrain_cubes_table.setStyleSheet("""
            QTableWidget { background:#1f1f1f; alternate-background-color:#292929;
                           color:#eaeaea; gridline-color:#3a3a3a; }
            QHeaderView::section { background:#2b2b2b; color:#eaeaea;
                                   border:0; padding:6px; }
            QTableWidget::item:selected { background:#43566e; color:#ffffff; }
        """)
        self.svm_retrain_cubes_table.setToolTip(
            "Unchecked cubes are excluded from both grouped validation and final model fitting."
        )
        cubes_header = self.svm_retrain_cubes_table.horizontalHeader()
        cubes_header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        cubes_header.setSectionResizeMode(1, QHeaderView.Stretch)
        cubes_header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.svm_retrain_cubes_caption = QLabel("<b>Cubes to include in next retraining:</b>")
        self.svm_retrain_cubes_caption.setToolTip(
            "A cube unchecked here is not used at all for the next training run. "
            "Platt uses cube-held-out scores. Nested validation needs at least three cubes, "
            "with every class present in each training fold. RBF calibration adds SVM fits."
        )
        self.btn_retrain_svm_dashboard = QPushButton("Retrain")
        self.btn_cancel_svm_training = QPushButton("Cancel training")
        self.btn_cancel_svm_training.setEnabled(False)
        self.btn_delete_svm_model = QPushButton("Delete model…")
        self.svm_training_indicator = QLabel("● Idle")
        self.svm_training_indicator.setStyleSheet("color:#9e9e9e; font-weight:600;")
        self.svm_retrain_status = QLabel("Select a model belonging to a valid ROI dataset.")
        self.svm_retrain_status.setWordWrap(True)
        self.svm_training_steps = QLabel()
        self.svm_training_steps.setTextFormat(Qt.RichText)
        self.svm_training_steps.setWordWrap(True)
        self._svm_training_steps_state = {}
        self._update_svm_training_steps()
        self.svm_retrain_log = QPlainTextEdit()
        self.svm_retrain_log.setReadOnly(True)
        self.svm_retrain_log.setMinimumHeight(260)
        self.svm_enable_platt = QCheckBox("Enable Platt scaling (on retraining)")
        self.svm_enable_platt.setToolTip("Calibrate every class on cube-held-out scores. Requires sufficient annotated cubes; saved models retain their own calibration setting.")
        retrain_layout.addWidget(self.svm_enable_platt, 8, 0, 1, 6)
        retrain_layout.addWidget(QLabel("Kernel:"), 0, 0)
        retrain_layout.addWidget(self.combo_svm_retrain_kernel, 0, 1)
        retrain_layout.addWidget(self.btn_configure_svm_bands, 0, 2)
        retrain_layout.addWidget(self.btn_retrain_svm_dashboard, 0, 3)
        retrain_layout.addWidget(self.btn_cancel_svm_training, 0, 4)
        retrain_layout.addWidget(self.btn_delete_svm_model, 0, 5)
        retrain_layout.addWidget(self.svm_training_indicator, 1, 0)
        retrain_layout.addWidget(self.svm_retrain_bands, 1, 1, 1, 2)
        retrain_layout.addWidget(QLabel("Band weights:"), 1, 3)
        retrain_layout.addWidget(self.svm_retrain_band_weights, 1, 4, 1, 2)
        retrain_layout.addWidget(self.svm_retrain_status, 2, 0, 1, 6)
        retrain_layout.addWidget(self.svm_retrain_cubes_caption, 3, 0, 1, 6)
        retrain_layout.addWidget(self.svm_retrain_cubes_table, 4, 0, 1, 6)
        retrain_layout.addWidget(self.svm_training_steps, 5, 0, 1, 6)
        retrain_layout.addWidget(self.svm_retrain_log, 6, 0, 1, 6)
        self._svm_training_ui_timer = QTimer(self)
        self._svm_training_ui_timer.setInterval(1000)
        self._svm_training_ui_timer.timeout.connect(self._update_svm_training_elapsed)
        detail_tabs.addTab(retrain_group, "Retraining")
        splitter.addWidget(details)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 4)
        splitter.setSizes([430, 750])

        self.btn_refresh_svm_dashboard.clicked.connect(self._refresh_svm_dashboard)
        self.svm_dashboard_table.itemSelectionChanged.connect(self._show_selected_svm_model)
        self.btn_retrain_svm_dashboard.clicked.connect(self._retrain_selected_svm_model)
        self.btn_configure_svm_bands.clicked.connect(self._configure_selected_svm_bands)
        self.btn_cancel_svm_training.clicked.connect(self._cancel_svm_training)
        self.btn_delete_svm_model.clicked.connect(self._delete_selected_svm_model)
        self._refresh_svm_dashboard()
        return tab

    @staticmethod
    def _svm_profile_from_filename(path: Path) -> str:
        stem = re.sub(r"_backup_.*$", "", path.stem)
        return stem.split("_svm", 1)[0] if "_svm" in stem else stem

    def _make_svm_dashboard_filters(self, layout):
        row = QGridLayout()
        self.svm_filter_profile = QLineEdit()
        self.svm_filter_profile.setPlaceholderText("Profile contains… e.g. beef")
        self.svm_filter_profile.setClearButtonEnabled(True)
        self.svm_filter_scope = QComboBox()
        for text,value in [('Any training scope',''),('Legacy class subset','biological'),('All classes','complete')]:
            self.svm_filter_scope.addItem(text,value)
        self.svm_filter_kernel = QComboBox()
        for text,value in [('Any kernel',''),('Linear','linear'),('RBF','rbf')]:
            self.svm_filter_kernel.addItem(text,value)
        self.svm_filter_normalization = QComboBox()
        self.svm_filter_normalization.addItem('Any normalization','')
        self.svm_filter_normalization.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.svm_filter_normalization.setMinimumContentsLength(12)
        reset = QPushButton('Reset filters and sort')
        row.addWidget(self.svm_filter_profile,0,0,1,2)
        row.addWidget(reset,0,2)
        for column,widget in enumerate((self.svm_filter_scope,self.svm_filter_kernel,self.svm_filter_normalization)):
            row.addWidget(widget,1,column)
            row.setColumnStretch(column,1)
        layout.addLayout(row)
        self.svm_filter_count = QLabel()
        self.svm_filter_count.setStyleSheet('color:#b8b8b8;')
        layout.addWidget(self.svm_filter_count)
        self.svm_filter_profile.textChanged.connect(lambda *_: self._render_svm_dashboard())
        for combo in (self.svm_filter_scope,self.svm_filter_kernel,self.svm_filter_normalization):
            combo.currentIndexChanged.connect(lambda *_: self._render_svm_dashboard())
        reset.clicked.connect(self._reset_svm_dashboard_filters)

    def _reset_svm_dashboard_filters(self):
        widgets=(self.svm_filter_profile,self.svm_filter_scope,self.svm_filter_kernel,self.svm_filter_normalization)
        for widget in widgets: widget.blockSignals(True)
        self.svm_filter_profile.clear()
        for combo in widgets[1:]: combo.setCurrentIndex(0)
        for widget in widgets: widget.blockSignals(False)
        self._svm_dashboard_sort = None
        self.svm_dashboard_table.horizontalHeader().setSortIndicatorShown(False)
        self._render_svm_dashboard()

    def _sort_svm_dashboard(self, column):
        previous=getattr(self,'_svm_dashboard_sort',None)
        descending=not previous[1] if previous and previous[0]==column else column==7
        self._svm_dashboard_sort=(column,descending)
        header=self.svm_dashboard_table.horizontalHeader()
        header.setSortIndicator(column,Qt.DescendingOrder if descending else Qt.AscendingOrder)
        header.setSortIndicatorShown(True)
        self._render_svm_dashboard()

    def _svm_dashboard_filtered_records(self):
        records=list(getattr(self,'_svm_all_dashboard_records',[]))
        profile=self.svm_filter_profile.text().strip().casefold() if hasattr(self,'svm_filter_profile') else ''
        scope=self.svm_filter_scope.currentData() if hasattr(self,'svm_filter_scope') else ''
        kernel=self.svm_filter_kernel.currentData() if hasattr(self,'svm_filter_kernel') else ''
        norm=self.svm_filter_normalization.currentData() if hasattr(self,'svm_filter_normalization') else ''
        records=[r for r in records if (not profile or profile in r['profile'].casefold())
                 and (not scope or r.get('training_scope','complete')==scope)
                 and (not kernel or str(r['payload'].get('svm_type','')).lower()==kernel)
                 and (not norm or r['payload'].get('normalization_mode','none')==norm)]
        order=getattr(self,'_svm_dashboard_sort',None)
        self._svm_flat_dashboard_view=bool(profile or scope or kernel or norm or order)
        if order:
            column,descending=order
            def key(r):
                p=r['payload'];bands=np.asarray(p.get('selected_wavelengths',[]),float)
                if column==0:return r['profile'].casefold()
                if column==1:return 0 if r.get('training_scope')=='biological' else 1
                if column==2:return str(p.get('svm_type','')).casefold()
                if column==3:
                    c=p.get('rbf_c') if p.get('svm_type')=='rbf' else p.get('linear_c')
                    return (float(c) if c is not None else -np.inf,str(p.get('rbf_gamma','')))
                if column==4:return (len(bands),tuple(bands))
                if column==5:return str(p.get('normalization_mode','none')).casefold()
                if column==6:return self._svm_record_dataset_state(r)
                return float((r.get('report') or {})['best_score'])
            if column==7:
                scored=[];missing=[]
                for record in records:
                    value=(record.get('report') or {}).get('best_score')
                    (scored if value is not None and np.isfinite(float(value)) else missing).append(record)
                records=sorted(scored,key=key,reverse=descending)+missing
            else: records.sort(key=key,reverse=descending)
        else: records.sort(key=lambda r:r['profile'].casefold())
        return records

    def _read_svm_dashboard_record(self, path: Path) -> dict:
        record = {"path": path, "error": None, "compatibility_warnings": []}
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", InconsistentVersionWarning)
                payload = joblib.load(path)
            record["compatibility_warnings"] = [
                str(item.message)
                for item in caught
                if issubclass(item.category, InconsistentVersionWarning)
            ]
            if not isinstance(payload, dict):
                raise ValueError("unsupported model file")
            record["payload"] = payload
        except Exception as exc:
            record["payload"] = {}
            record["error"] = str(exc)
        payload = record["payload"]
        profile = self._svm_profile_from_filename(path)
        biological = bool(re.search(r"_svm_tissue(?:_|$)", path.stem, re.IGNORECASE))
        dataset_value = payload.get("dataset_path") or (payload.get("dataset_signature") or {}).get("path")
        dataset_path = resolve_stored_path(
            dataset_value,
            Path(self.roi_out_dir) / f"{profile}.npz",
        )
        if biological and dataset_value:
            # A missing derived dataset must never silently fall back to the
            # full four-class ROI dataset, especially for dashboard retraining.
            dataset_path = resolve_project_path(dataset_value)
        record["profile"] = profile
        record["training_scope"] = "biological" if biological else "complete"
        record["is_generic_family_file"] = path.stem == f"{profile}_svm"
        record["dataset_path"] = dataset_path
        record["report"] = payload.get("grouped_training") or {}
        return record

    def _refresh_svm_dashboard(self) -> None:
        table = getattr(self, "svm_dashboard_table", None)
        if table is None:
            return
        selected_path = None
        row = table.currentRow()
        if row >= 0:
            item = table.item(row, 0)
            selected_path = item.data(Qt.UserRole) if item else None

        paths = sorted(
            (
                path for path in Path(self.svm_models_dir).glob("*.joblib")
                if "_backup_" not in path.stem.lower()
            ),
            key=lambda candidate: candidate.stat().st_mtime,
            reverse=True,
        )
        records = [self._read_svm_dashboard_record(path) for path in paths]
        self._svm_all_dashboard_records = self._deduplicate_svm_dashboard_records(records)
        if hasattr(self,'svm_filter_normalization'):
            combo=self.svm_filter_normalization
            selected=combo.currentData()
            combo.blockSignals(True);combo.clear();combo.addItem('Any normalization','')
            from shared.svm_transforms import NORMALIZATION_OPTIONS
            labels=dict((mode,label) for label,mode in NORMALIZATION_OPTIONS)
            modes=sorted({r['payload'].get('normalization_mode','none') for r in self._svm_all_dashboard_records})
            for mode in modes:combo.addItem(labels.get(mode,mode),mode)
            combo.setCurrentIndex(max(0,combo.findData(selected)));combo.blockSignals(False)
        self._render_svm_dashboard(selected_path)

    def _render_svm_dashboard(self, selected_path=None):
        table=getattr(self,'svm_dashboard_table',None)
        if table is None:return
        if selected_path is None and table.currentRow()>=0:
            item=table.item(table.currentRow(),0)
            selected_path=item.data(Qt.UserRole) if item else None
        self._svm_dashboard_records=self._svm_dashboard_filtered_records()
        table.blockSignals(True)
        table.setRowCount(len(self._svm_dashboard_records))
        selected_row = 0 if self._svm_dashboard_records else -1
        for row_index, record in enumerate(self._svm_dashboard_records):
            payload = record["payload"]
            report = record["report"]
            svm_type = str(payload.get("svm_type", "unknown")).upper()
            wavelengths = np.asarray(payload.get("selected_wavelengths", []), dtype=float)
            if wavelengths.size:
                spectral_text = f"{wavelengths.size} · {wavelengths.min():g}–{wavelengths.max():g} nm"
            else:
                spectral_text = "bands unknown"
            if svm_type == "LINEAR":
                parameter_text = f"C={payload.get('linear_c', '—')}"
            elif svm_type == "RBF":
                parameter_text = (
                    f"C={payload.get('rbf_c', '—')}, γ={payload.get('rbf_gamma', '—')}"
                )
            else:
                parameter_text = "parameters unknown"
            scope = "Legacy class subset" if record.get("training_scope") == "biological" else "All classes"
            norm = str(payload.get("normalization_mode", "none"))
            norm_text = {"snv": "SNV + ratio", "snv_three_ratios": "SNV + 3 ratios", "snv_no_ratio": "SNV without ratio", "log": "−log10(R)"}.get(norm, norm)
            variant = f"{scope} · {svm_type} · {parameter_text} · {spectral_text} · {norm_text}"
            dataset_state = self._svm_record_dataset_state(record)
            score = report.get("best_score")
            if score is not None and not np.isfinite(float(score)):
                score = None
            values = [
                record["profile"],
                scope,
                svm_type,
                parameter_text,
                spectral_text,
                norm_text,
                dataset_state,
                f"{float(score):.4f}" if score is not None else "Not evaluated",
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 0:
                    item.setData(Qt.UserRole, str(record["path"]))
                    item.setToolTip(str(record["path"]))
                    if record["error"]:
                        item.setToolTip(record["error"])
                elif column > 0:
                    item.setTextAlignment(Qt.AlignCenter)
                    if column == 1:
                        classes = ', '.join(str(name) for name in payload.get('class_names', []))
                        item.setToolTip(f"{variant}\nClasses: {classes}\nFile: {record['path'].name}")
                    elif column == 4:
                        item.setToolTip("Selected wavelengths (nm): " + ", ".join(f"{value:g}" for value in wavelengths))
                    elif column == 6:
                        item.setToolTip(f"Training dataset: {record['dataset_path']}")
                    elif column == 7 and score is None:
                        item.setToolTip(self._svm_validation_unavailable_reason(record))
                    elif column == 7:
                        item.setToolTip("Fat vs non-fat macro-F1 after fat-only Platt · nested leave-one-cube-out validation" if
                            (report.get('calibration') or {}).get('target') == 'fat' else
                            "After Platt · nested cube-grouped validation" if report.get("prediction_rule")
                            in ("Platt probability argmax", "Platt fat probability threshold 0.5") else "Legacy raw SVM · cube-grouped validation · not calibrated")
                table.setItem(row_index, column, item)
            if selected_path and str(record["path"]) == str(selected_path):
                selected_row = row_index
        table.blockSignals(False)
        self._apply_svm_dataset_groups()
        if hasattr(self,'svm_filter_count'):
            shown=len(self._svm_dashboard_records);total=len(self._svm_all_dashboard_records)
            self.svm_filter_count.setText(f'{shown} / {total} model variants match' +
                (' · all matching variants shown' if self._svm_flat_dashboard_view else ' · grouped by specimen profile'))
            self.btn_toggle_svm_variants.setEnabled(bool(shown) and not self._svm_flat_dashboard_view)
        if selected_row >= 0:
            if table.isRowHidden(selected_row):
                profile=self._svm_dashboard_records[selected_row]['profile']
                selected_row=next(i for i,r in enumerate(self._svm_dashboard_records) if r['profile']==profile)
            table.blockSignals(True)
            table.selectRow(selected_row)
            table.blockSignals(False)
            self._show_selected_svm_model()
        else:
            self._clear_svm_dashboard_details("No model matches these filters." if getattr(self,'_svm_all_dashboard_records',[]) else "No SVM model found.")

    def _apply_svm_dataset_groups(self):
        """Keep real model rows and their action targets; hide sibling variants."""
        groups = {}
        for row, record in enumerate(self._svm_dashboard_records):
            groups.setdefault(record['profile'], []).append(row)
        expanded = getattr(self, '_svm_expanded_profiles', set())
        table = self.svm_dashboard_table
        if getattr(self,'_svm_flat_dashboard_view',False):
            for row,record in enumerate(self._svm_dashboard_records):
                table.setRowHidden(row,False)
                table.item(row,0).setText(record['profile'])
            return
        for profile, rows in groups.items():
            for offset, row in enumerate(rows):
                table.setRowHidden(row, offset > 0 and profile not in expanded)
                item = table.item(row, 0)
                if offset == 0:
                    marker = '−' if profile in expanded else '+'
                    item.setText(f'{marker} {profile} ({len(rows)} variants)' if len(rows) > 1 else profile)
                    item.setToolTip(str(self._svm_dashboard_records[row]['path']) +
                        '\nDouble-click to expand/collapse this dataset. Columns describe this model; actions apply to the selected row.')
                else:
                    item.setText('    ' + profile)

    def _toggle_svm_dataset_variants(self):
        if getattr(self,'_svm_flat_dashboard_view',False):return
        row = self.svm_dashboard_table.currentRow()
        if row < 0:
            return
        profile = self._svm_dashboard_records[row]['profile']
        expanded = set(getattr(self, '_svm_expanded_profiles', set()))
        if profile in expanded:
            expanded.remove(profile)
            first = next(i for i, record in enumerate(self._svm_dashboard_records) if record['profile'] == profile)
            self.svm_dashboard_table.selectRow(first)
        else:
            expanded.add(profile)
        self._svm_expanded_profiles = expanded
        self._apply_svm_dataset_groups()

    @staticmethod
    def _svm_validation_unavailable_reason(record):
        report = record.get('report') or {}
        calibration = report.get('calibration') or {}
        return calibration.get('reason') or report.get('reason') or "No cube-grouped validation saved for this model."

    @staticmethod
    def _svm_record_dataset_state(record: dict) -> str:
        dataset_path = record.get("dataset_path")
        signature = (record.get("payload") or {}).get("dataset_signature") or {}
        if not isinstance(dataset_path, Path) or not dataset_path.is_file():
            return "Missing"
        if not signature:
            return "Unknown"
        stat = dataset_path.stat()
        same_size = int(signature.get("size_bytes", -1)) == int(stat.st_size)
        same_mtime = int(signature.get("mtime_ns", -1)) == int(stat.st_mtime_ns)
        return "Current" if same_size and same_mtime else "Outdated"

    @staticmethod
    def _svm_configuration_key(record: dict) -> tuple:
        payload = record.get("payload") or {}
        svm_type = str(payload.get("svm_type", "unknown")).lower()
        wavelengths = tuple(
            round(float(value), 4)
            for value in np.asarray(payload.get("selected_wavelengths", []), dtype=float)
        )
        if svm_type == "linear":
            params = (payload.get("linear_c"),)
        else:
            params = (payload.get("rbf_c"), str(payload.get("rbf_gamma")))
        signature = payload.get("dataset_signature") or {}
        dataset_version = (
            signature.get("size_bytes"), signature.get("mtime_ns")
        )
        return (
            record.get("profile"), svm_type,
            record.get("training_scope", "complete"),
            tuple(str(name) for name in payload.get("class_names", [])),
            str(record.get("dataset_path", "")),
            str(payload.get("normalization_mode", "none")),
            wavelengths, params, dataset_version,
        )

    def _deduplicate_svm_dashboard_records(self, records: list[dict]) -> list[dict]:
        grouped = {}
        for record in records:
            if record.get("error"):
                grouped[(str(record.get("path")),)] = [record]
            else:
                grouped.setdefault(self._svm_configuration_key(record), []).append(record)

        output = []
        for duplicates in grouped.values():
            # Prefer the compact band-count filename over the legacy generic
            # filename, while retaining the richest validation report.
            chosen = next(
                (item for item in duplicates if re.search(r"_b\d+$", item["path"].stem)),
                duplicates[0],
            )
            best_report = max(
                (item.get("report") or {} for item in duplicates),
                key=lambda report: (
                    report.get("confusion_matrix") is not None,
                    report.get("best_score") is not None,
                ),
            )
            chosen["report"] = best_report
            chosen["equivalent_files"] = (
                [str(item["path"]) for item in duplicates]
            )
            output.append(chosen)
        return sorted(output, key=lambda item: item["path"].stat().st_mtime, reverse=True)

    def _clear_svm_dashboard_details(self, message: str = "—") -> None:
        if hasattr(self, "svm_calibration"):
            self.svm_calibration.set_record(None)
        if hasattr(self, "svm_visualization"):
            self.svm_visualization.set_record(None)
        for label in getattr(self, "svm_dashboard_labels", {}).values():
            label.setText("—")
        table = getattr(self, "svm_class_distribution_table", None)
        if table is not None:
            table.setRowCount(0)
        if hasattr(self, "svm_retrain_status"):
            self.svm_retrain_status.setText(message)
        self._draw_svm_confusion(None, None, message)

    def _show_selected_svm_model(self) -> None:
        row = self.svm_dashboard_table.currentRow()
        if row < 0 or row >= len(getattr(self, "_svm_dashboard_records", [])):
            self._clear_svm_dashboard_details()
            return
        record = self._svm_dashboard_records[row]
        payload = record["payload"]
        report = record["report"]
        self.svm_visualization.set_record(record)
        self.svm_calibration.set_record(record)
        if record["error"]:
            self._clear_svm_dashboard_details(f"Cannot read model: {record['error']}")
            self.svm_dashboard_labels["file"].setText(record["path"].name)
            return

        names = [str(value) for value in payload.get("class_names", [])]
        cubes = [str(value) for value in payload.get("dataset_cube_ids", report.get("cube_ids", []))]
        selected_wavelengths = np.asarray(payload.get("selected_wavelengths", []), dtype=float)
        svm_type = str(payload.get("svm_type", "unknown")).lower()
        if svm_type == "linear":
            params = f"C={payload.get('linear_c', '—')}"
        else:
            params = f"C={payload.get('rbf_c', '—')}, gamma={payload.get('rbf_gamma', '—')}"
        score = report.get("best_score")
        folds = report.get("n_splits")
        if score is not None:
            cv_text = f"macro-F1 {float(score):.4f} · {folds} folds grouped by cube"
            cv_text += " · after Platt" if str(report.get("prediction_rule", "")).startswith("Platt") else " · raw SVM"
        else:
            cv_text = "Not evaluated — retrain this model to generate grouped validation statistics"
        pca_components = payload.get("pca_n_components")
        pca_explained = payload.get("pca_explained_variance")
        pca_text = "Disabled"
        if payload.get("use_pca"):
            pca_text = f"{pca_components or '—'} components"
            if pca_explained is not None:
                pca_text += f" · {100.0 * float(pca_explained):.3f}% variance"
        samples = payload.get("n_training_samples", report.get("n_final_training_samples"))
        labels = self.svm_dashboard_labels
        labels["file"].setText(record["path"].name)
        labels["dataset"].setText(str(record["dataset_path"]))
        labels["kernel"].setText(svm_type.upper())
        labels["params"].setText(params)
        labels["normalization"].setText(
            str(payload["normalization_mode"]) if payload.get("normalization_mode") is not None
            else "Unknown (not recorded in model)"
        )
        labels["samples"].setText(f"{int(samples):,}" if samples is not None else "—")
        labels["classes"].setText(", ".join(names) or "—")
        labels["cubes"].setText("\n".join(cubes) or "—")
        labels["bands"].setText(
            f"{selected_wavelengths.size} · {selected_wavelengths.min():g}–{selected_wavelengths.max():g} nm"
            if selected_wavelengths.size else "—"
        )
        labels["pca"].setText(pca_text)
        labels["cv"].setText(cv_text)
        calibrated = bool(payload.get("platt_calibration"))
        self.svm_enable_platt.setChecked(calibrated)
        labels["calibration"].setText(
            ("Platt · only P(fat) calibrated · threshold 50%; other labels from raw SVM scores"
             if (payload.get('platt_calibration') or {}).get('target') == 'fat' else
             "Platt · unweighted cube-held-out scores · class = highest probability")
            if calibrated else "Platt disabled or unavailable · raw SVM predictions")
        after_platt = str(report.get("prediction_rule", "")).startswith("Platt")
        self.svm_validation_caption.setText(
            "After Platt · nested cube-grouped validation. Evaluated cubes were excluded from both the SVM "
            "and calibrator. Metrics cover annotated ROIs, not the spatial mask. "
            "HERAMAN uses the highest calibrated probability across all classes."
            if after_platt else
            "Raw SVM metrics, when available, use validation grouped by cube.")
        labels["updated"].setText(
            datetime.fromtimestamp(record["path"].stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        )
        compatibility_warnings = record.get("compatibility_warnings", [])
        if compatibility_warnings:
            version_pairs = set()
            for warning_text in compatibility_warnings:
                match = re.search(
                    r"from version ([^ ]+) when using version ([^ ]+)", warning_text
                )
                if match:
                    version_pairs.add((match.group(1), match.group(2).rstrip(".")))
            details = ", ".join(
                f"trained with sklearn {old}, running {current}"
                for old, current in sorted(version_pairs)
            )
            labels["compatibility"].setText(
                "⚠ Version mismatch — retraining recommended"
                + (f" ({details})" if details else "")
            )
            labels["compatibility"].setStyleSheet("color:#ffcc66;")
        else:
            labels["compatibility"].setText("Compatible with the current environment")
            labels["compatibility"].setStyleSheet("color:#8bd17c;")
        dataset_summary = self._populate_svm_class_distribution(record["dataset_path"])
        self._populate_svm_retrain_cubes(record["dataset_path"])
        model_cube_ids = set(cubes)
        current_cube_ids = set(dataset_summary.get("cube_ids", []))
        current_samples = dataset_summary.get("samples")
        missing_from_model = sorted(current_cube_ids - model_cube_ids)
        extra_in_model = sorted(model_cube_ids - current_cube_ids)
        model_is_current = (
            current_samples is not None
            and samples is not None
            and int(current_samples) == int(samples)
            and current_cube_ids == model_cube_ids
        )
        if model_is_current:
            labels["dataset_status"].setText(
                f"Current · {int(current_samples):,} pixels · {len(current_cube_ids)} cubes"
            )
            labels["dataset_status"].setStyleSheet("color:#8bd17c;")
        elif current_samples is None:
            labels["dataset_status"].setText("Dataset unavailable")
            labels["dataset_status"].setStyleSheet("color:#ffcc66;")
        else:
            differences = [
                f"model: {int(samples):,} pixels / {len(model_cube_ids)} cubes"
                if samples is not None else f"model: unknown pixels / {len(model_cube_ids)} cubes",
                f"dataset now: {int(current_samples):,} pixels / {len(current_cube_ids)} cubes",
            ]
            if missing_from_model:
                differences.append("not trained: " + ", ".join(missing_from_model))
            if extra_in_model:
                differences.append("absent from current dataset: " + ", ".join(extra_in_model))
            labels["dataset_status"].setText("⚠ OUTDATED · " + " · ".join(differences))
            labels["dataset_status"].setStyleSheet("color:#ffcc66;")
        self.combo_svm_retrain_kernel.setCurrentIndex(1 if svm_type == "rbf" else 0)
        self.combo_svm_retrain_kernel.setEnabled(False)
        self.combo_svm_retrain_kernel.setToolTip("Retraining preserves the selected model kernel and preprocessing.")
        self.svm_retrain_bands.setText("Selected model bands: " + ", ".join(f"{value:g} nm" for value in selected_wavelengths))
        self.svm_retrain_band_weights.setReadOnly(True)
        self.svm_retrain_band_weights.setText(", ".join(f"{k}={v:g}" for k,v in payload.get("band_weights_by_wavelength", {}).items()))
        can_retrain = record["dataset_path"].is_file()
        training_running = self._svm_training_is_running()
        self.btn_retrain_svm_dashboard.setEnabled(can_retrain and not training_running)
        self.btn_delete_svm_model.setEnabled(not training_running)
        if not training_running:
            self.svm_retrain_status.setText(
                "Ready to retrain this model." if can_retrain else "Associated ROI dataset not found."
            )
        self._draw_svm_confusion(
            report.get("confusion_matrix"), report.get("class_names", names),
            "This model has not been cross-validated yet. Retrain it from the Retraining tab "
            "to generate a cube-grouped confusion matrix.",
        )

    def _populate_svm_class_distribution(self, dataset_path: Path) -> dict:
        table = self.svm_class_distribution_table
        table.setRowCount(0)
        if not dataset_path.is_file():
            return {}
        try:
            with np.load(dataset_path, allow_pickle=True) as dataset:
                y = np.asarray(dataset["y"]).astype(str)
                dataset_cube_ids = (
                    [str(value) for value in np.asarray(dataset["cube_ids"], dtype=object).tolist()]
                    if "cube_ids" in dataset.files else []
                )
                roi_ids = np.asarray(dataset["roi_id"]) if "roi_id" in dataset.files else None
                cube_index = (
                    np.asarray(dataset["sample_cube_index"])
                    if "sample_cube_index" in dataset.files else None
                )
            classes, counts = np.unique(y, return_counts=True)
            table.setRowCount(len(classes))
            total = max(int(y.size), 1)
            for row, (class_name, count) in enumerate(zip(classes, counts)):
                selected = y == class_name
                values = [
                    str(class_name), f"{int(count):,}", f"{100.0 * int(count) / total:.1f}%",
                    str(int(np.unique(roi_ids[selected]).size)) if roi_ids is not None else "—",
                    str(int(np.unique(cube_index[selected]).size)) if cube_index is not None else "—",
                ]
                for column, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    if column:
                        item.setTextAlignment(Qt.AlignCenter)
                    table.setItem(row, column, item)
            return {"samples": int(y.size), "cube_ids": dataset_cube_ids}
        except Exception as exc:
            table.setRowCount(1)
            table.setItem(0, 0, QTableWidgetItem(f"Unavailable: {exc}"))
            return {}

    def _populate_svm_retrain_cubes(self, dataset_path: Path) -> None:
        """List current dataset cubes with persistent per-dataset inclusion choices."""
        table = self.svm_retrain_cubes_table
        table.setRowCount(0)
        if not dataset_path.is_file():
            return
        try:
            with np.load(dataset_path, allow_pickle=True) as dataset:
                cube_ids = [str(value) for value in np.asarray(dataset["cube_ids"], dtype=object)]
                cube_index = np.asarray(dataset["sample_cube_index"], dtype=np.int64)
            if cube_index.size == 0 or cube_index.min() < 0 or cube_index.max() >= len(cube_ids):
                raise ValueError("invalid or missing cube provenance")
        except Exception as exc:
            table.setRowCount(1)
            table.setItem(0, 1, QTableWidgetItem(f"Unavailable: {exc}"))
            return

        selections = getattr(self, "_svm_retrain_cube_selections", {})
        selection_key = str(dataset_path.resolve())
        checked_cube_ids = selections.get(selection_key, set(cube_ids))
        table.setRowCount(len(cube_ids))
        for row, cube_id in enumerate(cube_ids):
            enabled_item = QTableWidgetItem()
            enabled_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            enabled_item.setCheckState(Qt.Checked if cube_id in checked_cube_ids else Qt.Unchecked)
            enabled_item.setData(Qt.UserRole, cube_id)
            table.setItem(row, 0, enabled_item)
            cube_item = QTableWidgetItem(cube_id)
            cube_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            table.setItem(row, 1, cube_item)
            count_item = QTableWidgetItem(f"{int(np.sum(cube_index == row)):,}")
            count_item.setTextAlignment(Qt.AlignCenter)
            count_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            table.setItem(row, 2, count_item)

    def _selected_svm_retrain_cube_ids(self, dataset_path: Path) -> list[str]:
        table = self.svm_retrain_cubes_table
        selected = []
        for row in range(table.rowCount()):
            item = table.item(row, 0)
            if item is not None and item.checkState() == Qt.Checked:
                selected.append(str(item.data(Qt.UserRole)))
        selections = getattr(self, "_svm_retrain_cube_selections", {})
        selections[str(dataset_path.resolve())] = set(selected)
        self._svm_retrain_cube_selections = selections
        return selected

    def _selected_svm_retrain_band_weights(self) -> dict[int, float] | None:
        text = self.svm_retrain_band_weights.text().strip()
        if not text:
            return {}
        try:
            weights = {
                int(float(part.split("=", 1)[0].strip())): float(part.split("=", 1)[1].strip())
                for part in text.split(",") if part.strip()
            }
        except (IndexError, ValueError):
            QMessageBox.warning(self, "SVM band weights", "Use comma-separated nm=weight pairs, for example: 620=1.5, 680=0.7")
            return None
        if not weights or any(not np.isfinite(weight) or weight <= 0 for weight in weights.values()):
            QMessageBox.warning(self, "SVM band weights", "Every weight must be a finite number greater than zero.")
            return None
        return weights

    def _draw_svm_confusion(self, matrix, class_names, missing_text: str) -> None:
        figure = getattr(self, "svm_confusion_figure", None)
        if figure is None:
            return
        figure.clear()
        ax = figure.add_subplot(111)
        figure.patch.set_facecolor((0.16, 0.16, 0.16))
        ax.set_facecolor((0.18, 0.18, 0.18))
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#888888")
        try:
            values = np.asarray(matrix, dtype=float)
        except Exception:
            values = np.empty((0, 0))
        names = [str(value) for value in class_names] if class_names is not None else []
        if values.ndim != 2 or not values.size or values.shape[0] != values.shape[1]:
            ax.axis("off")
            ax.text(0.5, 0.5, missing_text, color="#dddddd", ha="center", va="center", wrap=True)
        else:
            row_sums = values.sum(axis=1, keepdims=True)
            normalized = np.divide(values, row_sums, out=np.zeros_like(values), where=row_sums > 0)
            image = ax.imshow(normalized, vmin=0.0, vmax=1.0, cmap="Blues")
            figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized rate")
            tick_names = names if len(names) == values.shape[0] else [str(i) for i in range(values.shape[0])]
            ax.set_xticks(range(len(tick_names)), tick_names, rotation=35, ha="right", color="white")
            ax.set_yticks(range(len(tick_names)), tick_names, color="white")
            ax.set_xlabel("Predicted class", color="white")
            ax.set_ylabel("True class", color="white")
            for y in range(values.shape[0]):
                for x in range(values.shape[1]):
                    color = "white" if normalized[y, x] > 0.55 else "black"
                    ax.text(x, y, f"{normalized[y, x] * 100:.1f}%\n{int(values[y, x]):,}",
                            ha="center", va="center", fontsize=8, color=color)
        figure.tight_layout()
        self.svm_confusion_canvas.draw_idle()

    def _update_svm_training_steps(self) -> None:
        label = getattr(self, "svm_training_steps", None)
        if label is None:
            return
        states = getattr(self, "_svm_training_steps_state", {})
        symbols = {"pending": "○", "running": "◉", "done": "✓", "failed": "✕"}
        colors = {"pending": "#888", "running": "#ffcc66", "done": "#8bd17c", "failed": "#ff7777"}
        captions = [
            ("validation", "Cube-grouped validation"),
            ("fit", "Final model fitting"),
            ("save", "Model and statistics saved"),
        ]
        parts = []
        for key, caption in captions:
            state = states.get(key, "pending")
            parts.append(
                f'<span style="color:{colors[state]}; font-weight:600">'
                f'{symbols[state]} {caption}</span>'
            )
        label.setText("&nbsp;&nbsp;&nbsp;".join(parts))

    def _set_svm_training_stage(self, stage: str, *, failed: bool = False) -> None:
        states = self._svm_training_steps_state
        previous = dict(states)
        if failed:
            for key, value in list(states.items()):
                if value == "running":
                    states[key] = "failed"
        elif stage == "validation":
            states.update(validation="running", fit="pending", save="pending")
        elif stage == "fit":
            states.update(validation="done", fit="running", save="pending")
        elif stage == "save":
            states.update(validation="done", fit="done", save="running")
        elif stage == "done":
            states.update(validation="done", fit="done", save="done")
        self._update_svm_training_steps()
        if previous != states and stage in {"fit", "save", "done"}:
            messages = {
                "fit": "✓ Grouped validation completed; fitting final model.",
                "save": "✓ Final model fitted; saving model and statistics.",
                "done": "✓ Model and validation statistics saved.",
            }
            self.svm_retrain_log.appendPlainText("[DASHBOARD] " + messages[stage])

    @staticmethod
    def _format_svm_elapsed(seconds: float) -> str:
        total = max(0, int(seconds))
        minutes, secs = divmod(total, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours} h {minutes:02d} min {secs:02d} s"
        if minutes:
            return f"{minutes} min {secs:02d} s"
        return f"{secs} s"

    def _update_svm_training_elapsed(self) -> None:
        if not self._svm_training_is_running():
            timer = getattr(self, "_svm_training_ui_timer", None)
            if timer is not None:
                timer.stop()
            return
        elapsed = time.monotonic() - getattr(self, "_svm_training_started_at", time.monotonic())
        kernel = getattr(self, "_svm_training_kernel", "SVM")
        pulse = "●" if int(elapsed) % 2 == 0 else "◉"
        self.svm_training_indicator.setText(
            f"{pulse} Training {kernel} · {self._format_svm_elapsed(elapsed)}"
        )
        self.svm_training_indicator.setStyleSheet("color:#ffcc66; font-weight:700;")

    def _svm_training_is_running(self) -> bool:
        process = getattr(self, "_svm_dashboard_process", None)
        return isinstance(process, QProcess) and process.state() != QProcess.NotRunning

    def _svm_profile_bands_path(self, profile: str) -> Path:
        return Path(self.svm_models_dir) / f"{profile}_svm_bands.json"

    def _load_svm_profile_bands(self, profile: str) -> list[float] | None:
        path = self._svm_profile_bands_path(profile)
        if not path.is_file():
            return None
        try:
            values = json.loads(path.read_text(encoding="utf-8")).get("selected_wavelengths_nm")
            return [float(value) for value in values] if values else None
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None

    def _configure_selected_svm_bands(self) -> None:
        row = self.svm_dashboard_table.currentRow()
        if row < 0:
            QMessageBox.information(self, "SVM bands", "Select an SVM model first.")
            return
        record = self._svm_dashboard_records[row]
        dataset_path = record["dataset_path"]
        if not dataset_path.is_file():
            QMessageBox.warning(self, "SVM bands", f"Dataset not found:\n{dataset_path}")
            return
        with np.load(dataset_path, allow_pickle=False) as data:
            if "wavelengths" not in data.files:
                QMessageBox.warning(self, "SVM bands", "The dataset has no saved wavelength axis.")
                return
            available = [float(value) for value in np.asarray(data["wavelengths"]).reshape(-1)]
        current = self._load_svm_profile_bands(record["profile"])
        initial = ", ".join(f"{value:g}" for value in (current or available))
        text, accepted = QInputDialog.getText(
            self, "SVM spectral bands",
            "Wavelengths in nm, comma-separated. Leave empty to use all dataset bands.\n"
            f"Available: {', '.join(f'{value:g}' for value in available)}",
            text=initial,
        )
        if not accepted:
            return
        requested = [part.strip() for part in text.split(",") if part.strip()]
        try:
            selected = [float(value) for value in requested] if requested else available
        except ValueError:
            QMessageBox.warning(self, "SVM bands", "Use only comma-separated numeric wavelengths.")
            return
        missing = [value for value in selected if not any(np.isclose(value, available, atol=1.0, rtol=0.0))]
        if missing or len(set(selected)) != len(selected):
            QMessageBox.warning(self, "SVM bands", f"Invalid or duplicate bands: {missing or selected}")
            return
        self._svm_profile_bands_path(record["profile"]).write_text(
            json.dumps({"selected_wavelengths_nm": selected}, indent=2), encoding="utf-8",
        )
        self.svm_retrain_bands.setText("Bands: " + ", ".join(f"{value:g} nm" for value in selected))
        self.svm_retrain_status.setText("Band profile saved. Retrain before classifying with this configuration.")
        if hasattr(self, '_queue_pp_model_check'):
            self._queue_pp_model_check()

    def _retrain_selected_svm_model(self) -> None:
        row = self.svm_dashboard_table.currentRow()
        if row < 0 or self._svm_training_is_running():
            return
        record = self._svm_dashboard_records[row]
        dataset_path = record["dataset_path"]
        if not dataset_path.is_file():
            QMessageBox.warning(self, "SVM training", f"Dataset not found:\n{dataset_path}")
            return
        profile = record["profile"]
        suffix = "_svm_tissue.joblib" if record.get("training_scope") == "biological" else "_svm.joblib"
        model_path = Path(self.svm_models_dir) / f"{profile}{suffix}"
        kernel = str(record["payload"].get("svm_type", "linear"))
        normalization = str(record["payload"].get("normalization_mode", "none"))
        selected_bands = np.asarray(record["payload"].get("selected_wavelengths", []), dtype=float).tolist() or None
        selected_cube_ids = self._selected_svm_retrain_cube_ids(dataset_path)
        selected_band_weights = {}  # Weights and preprocessing are cloned from the selected model.
        if not selected_cube_ids:
            QMessageBox.warning(self, "SVM training", "Select at least one cube for training.")
            return
        warning = "\n\nRBF grouped validation may take several minutes." if kernel == "rbf" else ""
        answer = QMessageBox.question(
            self,
            "Retrain SVM",
            f"Retrain profile '{profile}' with the {kernel.upper()} kernel?\n\n"
            f"Dataset: {dataset_path}\n"
            f"Included cubes ({len(selected_cube_ids)}): " + ", ".join(selected_cube_ids)
            + warning,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        args = [
            "-u", str(PROJECT_ROOT / "processing" / "svm_classifier_runner.py"),
            "--train-only", "--dataset_path", str(dataset_path),
            "--model_path", str(model_path), "--normalization_mode", normalization,
            "--svm_type", kernel, "--force_retrain",
            "--keep_related_models", "--refresh_validation",
            "--reference_model", str(record["path"]),
        ]
        if self.svm_enable_platt.isChecked(): args.append("--platt")
        if selected_bands:
            args += ["--selected_wavelengths_nm", *[f"{value:g}" for value in selected_bands]]
        for cube_id in selected_cube_ids:
            args += ["--included_cube_id", cube_id]
        for wavelength, weight in sorted(selected_band_weights.items()):
            args += ["--band_weight", f"{wavelength}={weight:g}"]
        process = QProcess(self)
        from gui.runtime import configure_qt_worker
        configure_qt_worker(process)
        process.setWorkingDirectory(str(PROJECT_ROOT))
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.setProgram(str(self._get_python64_path()))
        process.setArguments(args)
        process.readyReadStandardOutput.connect(self._read_svm_dashboard_training_output)
        process.finished.connect(self._svm_dashboard_training_finished)
        process.errorOccurred.connect(self._svm_dashboard_training_error)
        self._svm_dashboard_process = process
        self._svm_training_cancelled = False
        self._svm_training_started_at = time.monotonic()
        self._svm_training_kernel = kernel.upper()
        self._svm_training_output_buffer = ""
        self._svm_training_decoder = utf8_decoder()
        self._svm_training_steps_state = {
            "validation": "running", "fit": "pending", "save": "pending"
        }
        self._update_svm_training_steps()
        self.svm_retrain_log.clear()
        self.svm_retrain_status.setText(
            f"Training {profile} ({kernel.upper()}) — grouped validation may be silent while a fold is running."
        )
        self.svm_training_indicator.setText(f"● Training {kernel.upper()} · 0 s")
        self.svm_training_indicator.setStyleSheet("color:#ffcc66; font-weight:700;")
        self.btn_retrain_svm_dashboard.setEnabled(False)
        self.btn_cancel_svm_training.setEnabled(True)
        self.btn_delete_svm_model.setEnabled(False)
        self.combo_svm_retrain_kernel.setEnabled(False)
        process.start()
        self._svm_training_ui_timer.start()

    def _read_svm_dashboard_training_output(self) -> None:
        process = getattr(self, "_svm_dashboard_process", None)
        if not isinstance(process, QProcess):
            return
        chunk = self._svm_training_decoder.decode(bytes(process.readAllStandardOutput()))
        buffer = getattr(self, "_svm_training_output_buffer", "") + chunk
        lines = buffer.splitlines(keepends=True)
        self._svm_training_output_buffer = ""
        if lines and not lines[-1].endswith(("\n", "\r")):
            self._svm_training_output_buffer = lines.pop()
        for line in lines:
            self._handle_svm_training_line(line.strip())

    def _handle_svm_training_line(self, text: str) -> None:
        if not text:
            return
        self.svm_retrain_log.appendPlainText(text)
        if "[SVM][GROUP-CV]" in text and (
            "fold" in text.lower() or "params=" in text or "fallback" in text.lower()
        ):
            self.svm_retrain_status.setText(text)
        if "[SVM][PLATT]" in text:
            self.svm_retrain_status.setText(text)
        if "[SVM][TRAIN] Training new SVM model" in text:
            self._set_svm_training_stage("fit")
            self.svm_retrain_status.setText("Grouped validation complete — fitting final model…")
        elif "[SVM][TRAIN] Saved model:" in text:
            self._set_svm_training_stage("save")
            self.svm_retrain_status.setText("Final model fitted — saving validation statistics…")
        elif "[SVM][TRAIN] Done in" in text:
            self._set_svm_training_stage("done")
        bar = self.svm_retrain_log.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _svm_dashboard_training_finished(self, exit_code: int, _status) -> None:
        self._read_svm_dashboard_training_output()
        trailing = getattr(self, "_svm_training_output_buffer", "").strip()
        if trailing:
            self._handle_svm_training_line(trailing)
        self._svm_training_output_buffer = ""
        self._svm_training_ui_timer.stop()
        process = getattr(self, "_svm_dashboard_process", None)
        self._svm_dashboard_process = None
        if isinstance(process, QProcess):
            process.deleteLater()
        cancelled = bool(getattr(self, "_svm_training_cancelled", False))
        if cancelled:
            self._set_svm_training_stage("", failed=True)
            self.svm_training_indicator.setText("■ Training cancelled")
            self.svm_training_indicator.setStyleSheet("color:#ffcc66; font-weight:700;")
            self.svm_retrain_status.setText("Training was cancelled. Existing saved models were preserved.")
            self.btn_retrain_svm_dashboard.setEnabled(True)
        elif int(exit_code) == 0:
            self._set_svm_training_stage("done")
            elapsed = time.monotonic() - getattr(self, "_svm_training_started_at", time.monotonic())
            self.svm_training_indicator.setText(
                f"✓ Completed · {self._format_svm_elapsed(elapsed)}"
            )
            self.svm_training_indicator.setStyleSheet("color:#8bd17c; font-weight:700;")
            self.svm_retrain_status.setText("Training completed. Dashboard refreshed.")
            self._refresh_svm_dashboard()
            self._refresh_svm_model_choices()
            row = self.svm_dashboard_table.currentRow()
            if row >= 0:
                record = self._svm_dashboard_records[row]
                if record["report"].get("best_score") is None:
                    self.svm_retrain_status.setText("Model fitted; validation unavailable: " + self._svm_validation_unavailable_reason(record))
        else:
            self._set_svm_training_stage("", failed=True)
            self.svm_training_indicator.setText("✕ Training failed")
            self.svm_training_indicator.setStyleSheet("color:#ff7777; font-weight:700;")
            self.svm_retrain_status.setText(f"Training failed (exit code {exit_code}).")
            self.btn_retrain_svm_dashboard.setEnabled(True)
        self.combo_svm_retrain_kernel.setEnabled(False)
        self.btn_cancel_svm_training.setEnabled(False)
        self.btn_delete_svm_model.setEnabled(True)

    def _svm_dashboard_training_error(self, _error) -> None:
        process = getattr(self, "_svm_dashboard_process", None)
        if not isinstance(process, QProcess) or process.state() != QProcess.NotRunning:
            return
        self._svm_training_ui_timer.stop()
        self._set_svm_training_stage("", failed=True)
        self.svm_training_indicator.setText("✕ Could not start training")
        self.svm_training_indicator.setStyleSheet("color:#ff7777; font-weight:700;")
        self.svm_retrain_status.setText(process.errorString())
        self.btn_retrain_svm_dashboard.setEnabled(True)
        self.combo_svm_retrain_kernel.setEnabled(False)
        self.btn_cancel_svm_training.setEnabled(False)
        self.btn_delete_svm_model.setEnabled(True)

    def _cancel_svm_training(self) -> None:
        process = getattr(self, "_svm_dashboard_process", None)
        if not isinstance(process, QProcess) or process.state() == QProcess.NotRunning:
            return
        self._svm_training_cancelled = True
        self.svm_retrain_status.setText("Cancelling training safely…")
        self.btn_cancel_svm_training.setEnabled(False)
        process.terminate()

        def _force_stop_if_needed():
            current = getattr(self, "_svm_dashboard_process", None)
            if isinstance(current, QProcess) and current.state() != QProcess.NotRunning:
                current.kill()

        QTimer.singleShot(1500, _force_stop_if_needed)

    def _delete_selected_svm_model(self) -> None:
        row = self.svm_dashboard_table.currentRow()
        if row < 0 or self._svm_training_is_running():
            return
        record = self._svm_dashboard_records[row]
        paths = [Path(value) for value in record.get("equivalent_files", [record["path"]])]
        existing = [path for path in paths if path.is_file()]
        if not existing:
            self._refresh_svm_dashboard()
            return
        answer = QMessageBox.question(
            self,
            "Delete SVM model",
            "Delete this model configuration?\n\n"
            + "\n".join(path.name for path in existing)
            + "\n\nThe ROI dataset and training statistics will not be deleted.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        failures = []
        for path in existing:
            try:
                path.unlink()
            except OSError as exc:
                failures.append(f"{path.name}: {exc}")
        self._refresh_svm_dashboard()
        self._refresh_svm_model_choices()
        if failures:
            QMessageBox.warning(self, "Delete SVM model", "\n".join(failures))
