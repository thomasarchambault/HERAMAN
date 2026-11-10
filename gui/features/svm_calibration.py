"""Reliability diagrams from saved nested held-out predictions, never training scores."""
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
import numpy as np


class SvmCalibrationWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.description = QLabel()
        self.description.setWordWrap(True)
        layout.addWidget(self.description)
        row = QHBoxLayout()
        row.addWidget(QLabel("Class:"))
        self.classes = QComboBox()
        row.addWidget(self.classes)
        row.addWidget(QLabel("Evaluation:"))
        self.cubes = QComboBox()
        row.addWidget(self.cubes, 1)
        layout.addLayout(row)
        self.metrics = QLabel()
        self.metrics.setWordWrap(True)
        layout.addWidget(self.metrics)
        self.figure = Figure(figsize=(7, 6), dpi=100, constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(360)
        toolbar = NavigationToolbar2QT(self.canvas, self)
        toolbar.setStyleSheet("background:#eeeeee; color:#222222;")
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas, 1)
        self.classes.currentIndexChanged.connect(self.redraw)
        self.cubes.currentIndexChanged.connect(self.redraw)
        self.set_record(None)

    def set_record(self, record):
        payload = (record or {}).get("payload") or {}
        report = (record or {}).get("report") or payload.get("grouped_training") or {}
        self.report = report.get("calibration") or {}
        available = self.report.get("status") == "evaluated"
        self.classes.blockSignals(True)
        self.cubes.blockSignals(True)
        self.classes.clear()
        self.cubes.clear()
        if available:
            for curve in self.report.get("curves", []):
                self.classes.addItem(curve["class_name"])
            fat = self.classes.findText("fat")
            self.classes.setCurrentIndex(max(0, fat))
            self.cubes.addItem("All held-out cubes (pooled pixels)", -1)
            for i, cube in enumerate(self.report.get("per_cube", [])):
                self.cubes.addItem(cube["cube_id"], i)
        self.classes.blockSignals(False)
        self.cubes.blockSignals(False)
        self.classes.setEnabled(available)
        self.cubes.setEnabled(available)
        if available:
            self.description.setText(("Multiclass SVM; only P(fat) is calibrated. Other class scores remain uncalibrated. " if self.report.get("target") == "fat" else "") + "SVM + Platt · nested cube-grouped validation. Each evaluated cube was excluded "
                "from both SVM fitting and calibration. ROI annotations only; spatial mask not evaluated. "
                "Calibration is unweighted. No confidence intervals; adjacent pixels are correlated.")
        else:
            reason = self.report.get("reason") or payload.get("calibration_unavailable_reason")
            self.description.setText("Calibration validation unavailable. " + (str(reason) if reason else
                "Retrain this model to generate Platt calibration and held-out reliability diagrams.") +
                " At least three suitably annotated independent cubes are needed for nested validation.")
        self.redraw()

    def redraw(self, *_):
        self.figure.clear()
        if self.report.get("status") != "evaluated" or self.classes.currentIndex() < 0:
            self.metrics.setText("No held-out calibration metrics available.")
            ax = self.figure.add_subplot(111)
            ax.axis("off")
            ax.text(.5, .5, "No calibration curve saved", ha="center", va="center")
            self.canvas.draw_idle()
            return
        cube_index = self.cubes.currentData()
        data = (self.report["per_cube"][cube_index] if cube_index is not None and cube_index >= 0 else self.report)
        curve = data["curves"][self.classes.currentIndex()]
        metric_scope = "fat vs non-fat" if self.report.get("target") == "fat" else "all classes"
        self.metrics.setText(f"{data['n_samples']:,} evaluated ROI pixels · Brier ({curve['class_name']} vs rest): "
            f"{curve['brier']:.6f} · Log-loss ({metric_scope}): {data['log_loss']:.6f}"
            + (f"\nEqual-weight mean across cubes: Brier ({metric_scope})={self.report['per_cube_mean_brier']:.6f} · "
               f"Log-loss={self.report['per_cube_mean_log_loss']:.6f}"
               if data is self.report else ""))
        top, lower = self.figure.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]})
        top.plot([0, 1], [0, 1], "--", color="gray", label="Ideal calibration: y = x")
        top.plot(curve["predicted"], curve["observed"], "o-", label="SVM + Platt (held-out)")
        top.set(xlim=(0, 1), ylim=(0, 1), xlabel=f"Mean predicted P({curve['class_name']})",
                ylabel=f"Observed fraction {curve['class_name']}", title="Reliability diagram · 10 equal-width bins")
        top.grid(alpha=.2)
        top.legend(fontsize=8)
        edges = np.asarray(data["bin_edges"])
        lower.bar(edges[:-1], curve["counts"], width=np.diff(edges), align="edge", edgecolor="white")
        lower.set(xlim=(0, 1), xlabel=f"P({curve['class_name']})", ylabel="ROI pixels")
        self.canvas.draw_idle()
