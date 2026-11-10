# Rôle : Présenter les métriques de validation et diagnostics de qualité des modèles SVM.

"""At-a-glance model diagnostics; keep validation and resubstitution distinct."""
from __future__ import annotations

import numpy as np
import textwrap
from sklearn.metrics import confusion_matrix
from PyQt5.QtCore import QObject, QRunnable, QThreadPool, Qt, pyqtSignal
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTabWidget, QFrame, QScrollArea, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView
from PyQt5.QtGui import QColor, QPalette
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from gui.features.svm_visualization import SvmVisualizationWidget, load_model_dataset
from shared.svm_calibration import predict_encoded

GREEN, AMBER, RED = "#69c99a", "#edbd62", "#ec8585"
OVERVIEW_MAX_PIXELS_PER_CLASS = 50_000
# Match the shared Qt stylesheet: page #2b2b2b, raised controls #3c3f41.
BG, PANEL, TEXT, MUTED = "#2b2b2b", "#3c3f41", "#f0f0f0", "#b8b8b8"


def compact_tooltip(text):
    """Keep native Qt tooltips readable, including explicit paragraph breaks."""
    return "\n".join(textwrap.fill(line, width=58, break_long_words=False,
                                  break_on_hyphens=False)
                     for line in str(text).split("\n"))


def summarize_confusion(matrix, names):
    values = np.asarray(matrix, dtype=float)
    if (values.shape != (len(names), len(names)) or not len(names)
            or not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0):
        raise ValueError("No usable evaluation matrix.")
    totals, predicted = values.sum(axis=1), values.sum(axis=0)
    diagonal = np.diag(values)
    recall = np.divide(diagonal, totals, out=np.full(len(names), np.nan), where=totals > 0)
    precision = np.divide(diagonal, predicted, out=np.full(len(names), np.nan), where=predicted > 0)
    f1 = np.divide(2*diagonal, totals+predicted, out=np.zeros(len(names)), where=totals+predicted > 0)
    f1[totals == 0] = np.nan
    pairs = [(float(values[i, j]/totals[i]), str(names[i]), str(names[j]))
             for i in range(len(names)) for j in range(len(names))
             if i != j and totals[i] > 0 and values[i, j] > 0]
    return dict(names=list(names), matrix=values, totals=totals, recall=recall,
                precision=precision, f1=f1, macro=float(np.nanmean(f1)),
                accuracy=float(diagonal.sum()/values.sum()),
                pairs=sorted(pairs, reverse=True)[:4])


def build_health(record):
    payload = record["payload"]
    report = record.get("report") or payload.get("grouped_training") or {}
    matrix = report.get("confusion_matrix")
    if matrix is not None:
        result = summarize_confusion(matrix, report.get("class_names", payload["class_names"]))
        stage = ("Multiclass SVM + fat-only Platt (fat vs non-fat metrics)" if
                 (report.get('calibration') or {}).get('target') == 'fat' else
                 "SVM + Platt" if report.get("prediction_rule") == "Platt probability argmax" else "Raw SVM")
        result.update(validated=True, excluded=0, source=f"{stage} · cube-grouped validation · {report.get('n_splits', '?')} folds")
        return result
    x, y = load_model_dataset(payload, record["dataset_path"])
    names = [str(value) for value in payload["class_names"]]
    valid = np.isfinite(x).all(axis=1) & np.isin(y, names)
    rng = np.random.default_rng(42)
    chosen = []
    for name in names:
        indices = np.flatnonzero(valid & (y == name))
        chosen.extend(rng.choice(indices, min(len(indices), OVERVIEW_MAX_PIXELS_PER_CLASS), replace=False))
    if not chosen:
        raise ValueError("No compatible samples.")
    ids = np.asarray(chosen, dtype=int)
    predicted = np.concatenate([predict_encoded(payload, x[ids[i:i+4096]]) for i in range(0, len(ids), 4096)])
    encoder = payload.get("label_encoder")
    labels = encoder.inverse_transform(predicted) if encoder is not None else np.asarray(names)[predicted.astype(int)]
    result = summarize_confusion(confusion_matrix(y[ids], labels, labels=names), names)
    result.update(validated=False, excluded=int((~valid).sum()),
                  source=f"ROI preview · {len(ids):,} pixels · max. {OVERVIEW_MAX_PIXELS_PER_CLASS:,} / class")
    return result


class _HealthSignals(QObject):
    done = pyqtSignal(int, object, str)


class _HealthJob(QRunnable):
    def __init__(self, token, record):
        super().__init__()
        self.token, self.record = token, record
        self.signals = _HealthSignals()

    def run(self):
        try:
            self.signals.done.emit(self.token, build_health(self.record), "")
        except Exception as exc:
            self.signals.done.emit(self.token, None, str(exc))


def score_color(value):
    return MUTED if not np.isfinite(value) else GREEN if value >= .9 else AMBER if value >= .75 else RED


def draw_health(figure, result):
    figure.clear()
    figure.set_layout_engine("constrained")
    figure.set_facecolor(BG)
    upper, lower = figure.subplots(2, 1, gridspec_kw={"height_ratios": [1.25, 1]})
    for ax in [upper, lower]:
        ax.set_facecolor(BG)
        ax.tick_params(colors=MUTED, length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_axisbelow(True)
    order = np.argsort(np.nan_to_num(result["f1"], nan=-1))
    labels = [result["names"][i] for i in order]
    upper.set_title("Performance by class", color=TEXT, loc="left", fontsize=12, pad=15)
    for row, i in enumerate(order):
        value = result["f1"][i]
        upper.barh(row, 100, height=.25, color=PANEL)
        if np.isfinite(value):
            upper.barh(row, value*100, height=.25, color=score_color(value))
            upper.text(102, row, f"{value:.1%}", va="center", color=TEXT, fontsize=11)
        else:
            upper.text(102, row, "—", va="center", color=MUTED)
    upper.set_yticks(range(len(labels)), labels, color=TEXT)
    upper.invert_yaxis()
    upper.set_xlim(0, 113)
    upper.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100 %"])
    upper.set_xlabel("F1 · balance between detection and false alarms", color=MUTED, fontsize=9)
    upper.grid(axis="x", alpha=.12)
    lower.set_title("Main confusions", color=TEXT, loc="left", fontsize=12, pad=15)
    pairs = result["pairs"]
    if not pairs:
        lower.set_axis_off()
        lower.text(.02, .55, "✓ No confusion observed", color=GREEN, fontsize=13, transform=lower.transAxes)
    else:
        for row, (rate, source, target) in enumerate(pairs):
            lower.barh(row, rate*100, color=RED if rate > .1 else AMBER, height=.48)
            lower.text(rate*100 + 1, row, f"{rate:.1%}", color=TEXT, va="center")
        lower.set_yticks(range(len(pairs)), [f"{a}  →  {b}" for _, a, b in pairs], color=TEXT)
        lower.invert_yaxis()
        lower.set_xlim(0, 112)
        lower.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100 %"])
        lower.set_xlabel("True class → predicted class · % of the true class", color=MUTED, fontsize=9)
        lower.grid(axis="x", alpha=.12)


class SvmHealthWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.token = 0
        self.jobs = {}
        self.record = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        page = QWidget()
        page.setObjectName("svmHealthSummary")
        page.setAutoFillBackground(True)
        self.page = page
        box = QVBoxLayout(page)
        self.badge = QLabel("Select a model")
        self.badge.setWordWrap(True)
        box.addWidget(self.badge)
        cards = QHBoxLayout()
        self.cards = []
        for title in ["Macro F1", "Strongest class", "Lowest class F1"]:
            frame = QFrame()
            frame.setStyleSheet(f"QFrame {{background:{PANEL}; border-radius:8px;}}")
            inner = QVBoxLayout(frame)
            caption = QLabel(title)
            caption.setToolTip("Macro F1 gives each evaluated class equal weight. It is not the percentage of correct pixels." if title == "Macro F1" else "Class ranked by F1 on the displayed evaluation data. A relative ranking, not a universal quality verdict.")
            caption.setStyleSheet(f"color:{MUTED}; font-size:12px;")
            value = QLabel("—")
            value.setTextFormat(Qt.PlainText)
            value.setWordWrap(True)
            inner.addWidget(caption)
            inner.addWidget(value)
            cards.addWidget(frame, 1)
            self.cards.append(value)
        box.addLayout(cards)
        self.source = QLabel()
        self.source.setWordWrap(True)
        self.source.setStyleSheet(f"color:{MUTED};")
        box.addWidget(self.source)
        guide = QLabel(
            "<b>How to read this page</b><br>"
            "The SVM learns from the tissue labels in your ROIs, then assigns a class "
            "to each pixel from its spectrum. Here, its predictions are compared with "
            "your ROI labels.<br>"
            "<b>Recall:</b> does it find the tissue? "
            "<b>Precision:</b> are its tissue labels correct? "
            "<b>F1:</b> how well does it do both?<br>"
            "A high score on pixels used for learning does not tell us how well it "
            "will work on a new specimen. Hover over ⓘ for examples."
        )
        guide.setWordWrap(True)
        box.addWidget(guide)
        self.figure = Figure(figsize=(7, 5), dpi=100, facecolor=BG)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setFixedHeight(340)
        box.addWidget(self.canvas, 1)
        legend = QLabel("● ≥ 90 %     ● 75–89 %     ● < 75 %")
        legend.setText(f'<span style="color:{GREEN}">● ≥ 90 %</span>　'
                       f'<span style="color:{AMBER}">● 75–89 %</span>　'
                       f'<span style="color:{RED}">● &lt; 75 %</span>')
        legend.setToolTip("Visual F1 guides, not universal quality thresholds.")
        box.addWidget(legend)
        def category(title, explanation):
            label = QLabel(title + "  ⓘ")
            label.setStyleSheet("font-weight:600; font-size:14px; margin-top:12px;")
            label.setToolTip(explanation)
            box.addWidget(label)
        category("Detection and reliability by class", "Each row is one tissue or background class.\n\nRecall: does the model find this class?\nPrecision: can you trust labels of this class?\nF1: balances these two abilities.\n\nGreen: ≥90%; amber: 75–89%; red: <75%.\nThese colors help you compare scores; they do not prove that a model is suitable for your use.")
        self.metrics = QTableWidget(0, 5)
        self.metrics.setStyleSheet("""
            QTableWidget { background:#2b2b2b; color:#f0f0f0;
                           gridline-color:#454545; }
            QHeaderView::section { background-color:#3c3f41;
                color:#f0f0f0; border:1px solid #555555;
                padding:6px; font-weight:600; }
            QTableCornerButton::section { background:#3c3f41; }
        """)
        self.metrics.setHorizontalHeaderLabels(["Class", "Recall ⓘ", "Precision ⓘ", "F1 ⓘ", "Pixels evaluated ⓘ"])
        tips = [
            "The reference label you assigned in a ROI.\nThe model prediction is compared with this label.",
            "RECALL — Does it find the tissue?\n\nExample: 80% recall for fat means that, out of 100 actual fat pixels, 80 are found and 20 are missed.\n\nCalculated as correct detections divided by actual pixels of this class.",
            "PRECISION — Are its labels correct?\n\nExample: 95% precision for fat means that, out of 100 pixels labelled fat by the model, 95 really are fat and 5 are another class.\n\nCalculated as correct detections divided by predictions of this class.\nA dash means the model never predicted this class.",
            "F1 — Does it both find and correctly label the tissue?\n\nA high F1 requires both high recall and high precision. A low value in either lowers this score.\n\nF1 = 2 × precision × recall / (precision + recall).\n90% F1 does NOT mean 90% of pixels are correct.\nA dash means there are no reference pixels to evaluate this class.",
            "How many reference pixels of this class were evaluated?\n\nThese are labelled pixels, not a count of independent specimens. Many neighboring pixels can come from a single ROI.\nMore pixels alone do not guarantee a stronger evaluation."
        ]
        tips = [compact_tooltip(tip) for tip in tips]
        for i, tip in enumerate(tips):
            self.metrics.horizontalHeaderItem(i).setToolTip(tip)
        self.metric_tips = tips
        self.metrics.verticalHeader().hide()
        self.metrics.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.metrics.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.metrics.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        box.addWidget(self.metrics)
        category("Observed strengths and weaknesses", "Missed pixels: the tissue was present, but the model called it something else.\n\nIncorrect labels: the model named this tissue, but the reference label says otherwise.\n\nExample: fat → meat = actual fat labelled meat. The percentage is taken out of actual fat pixels.\n\nThese errors do not explain their cause by themselves.")
        self.diagnosis = QLabel("—")
        self.diagnosis.setTextFormat(Qt.PlainText)
        self.diagnosis.setWordWrap(True)
        box.addWidget(self.diagnosis)
        category("Evaluation coverage and robustness", "ROI preview: checks the current labelled pixels, which may already have been used for learning.\n\nGrouped validation: learns without some cubes, then checks the ROIs in those excluded cubes. A fold is one repetition of this process.\n\nStability: does performance stay similar between cubes? This needs separate scores for each cube.\n\nTraining / validation gap: does performance drop on excluded data? This needs both scores.\n\nOnly annotated regions are checked, not the entire cube. Unavailable means missing evidence, not a bad score.")
        self.coverage = QLabel("—")
        self.coverage.setTextFormat(Qt.PlainText)
        self.coverage.setWordWrap(True)
        box.addWidget(self.coverage)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        self.tabs.addTab(scroll, "Quick overview")
        self.technical = SvmVisualizationWidget()
        self.technical_tab_index = self.tabs.addTab(self.technical, "Model · details")
        self._wrap_tooltips()

    def _wrap_tooltips(self):
        for widget in self.page.findChildren(QWidget):
            if widget.toolTip():
                widget.setToolTip(compact_tooltip(widget.toolTip()))

    def set_record(self, record):
        self.record = record
        self.token += 1
        payload = (record or {}).get("payload", {})
        svm_type = str(payload.get("svm_type", "")).lower()
        model = payload.get("model")
        svm = getattr(model, "named_steps", {}).get("svm") if model is not None else None
        rbf_map = svm_type == "rbf" and int(getattr(svm, "n_features_in_", 0)) == 2
        self.tabs.setTabText(
            self.technical_tab_index,
            "Confidence & decisions",
        )
        self.technical.set_record(record)
        self.metrics.setRowCount(0)
        self.diagnosis.setText("—")
        self.coverage.setText("—")
        self.figure.clear()
        self.canvas.draw_idle()
        for value in self.cards:
            value.setText("—")
        self.source.clear()
        self.badge.setText("Analyzing…" if record else "No model available")
        self.badge.setStyleSheet(f"color:{MUTED};padding:8px;")
        if not record or record.get("error"):
            return
        job = _HealthJob(self.token, record)
        self.jobs[self.token] = job
        job.signals.done.connect(self.finish)
        QThreadPool.globalInstance().start(job)

    def finish(self, token, result, error):
        self.jobs.pop(token, None)
        if token != self.token:
            return
        if error:
            self.badge.setText("Overview unavailable · missing or incompatible data")
            self.badge.setToolTip(compact_tooltip(error))
            return
        validated = result["validated"]
        self.badge.setText(("● SVM + Platt · cube-grouped validation" if result["source"].startswith("SVM + Platt")
                           else "● Raw SVM · cube-grouped validation") if validated else "● Not validated · ROI data preview only")
        self.badge.setStyleSheet(f"color:{GREEN if validated else AMBER};background:{PANEL};padding:10px;border-radius:6px;")
        self.badge.setToolTip("Cross-validation on held-out cube groups, not an external test. Parameter selection may also use these results."
                              if validated else "These pixels may have been used for training. High scores do not prove generalization.")
        values = result["f1"]
        best, worst = int(np.nanargmax(values)), int(np.nanargmin(values))
        self.cards[0].setText(f"{result['macro']:.1%}")
        tied = np.all(np.isfinite(values)) and np.allclose(values, values[best], atol=1e-6)
        self.cards[1].setText(f"All classes · {values[best]:.1%}" if tied
                              else f"{result['names'][best]} · {values[best]:.1%}")
        self.cards[2].setText(f"{result['names'][worst]} · {values[worst]:.1%}")
        self.cards[0].setToolTip("Mean per-class F1 from the displayed matrix. Each class has equal weight.")
        for label, index in [(self.cards[1], best), (self.cards[2], worst)]:
            label.setToolTip(f"{result['names'][index]} · recall {result['recall'][index]:.1%} · "
                             f"precision {result['precision'][index]:.1%} · {result['totals'][index]:,.0f} pixels")
        for label, score in zip(self.cards, [result["macro"], values[best], values[worst]]):
            label.setStyleSheet(f"color:{score_color(score)};font-size:20px;font-weight:600;")
        source = result["source"]
        if not validated:
            source += " · exploratory scores"
        if result["excluded"]:
            source += f" · {result['excluded']:,} excluded pixels"
        missing = [result["names"][i] for i, total in enumerate(result["totals"]) if total == 0]
        if missing:
            source += " · unevaluated classes: " + ", ".join(missing)
        signature = self.record["payload"].get("dataset_signature") or {}
        try:
            stat = self.record["dataset_path"].stat()
            matches = (signature.get("size_bytes") == stat.st_size
                       and signature.get("mtime_ns") == stat.st_mtime_ns)
        except (OSError, AttributeError, KeyError):
            matches = False
        if not matches:
            source += " · data changed or provenance unknown"
        if self.record["payload"].get("normalization_mode") == "snv_offset" and not validated:
            source += " · batch-dependent normalization"
        self.source.setText(source)
        self.metrics.setRowCount(len(values))
        statements = []
        for row, name in enumerate(result["names"]):
            p, r, f = (result[key][row] for key in ("precision", "recall", "f1"))
            cells = [name, r, p, f, result["totals"][row]]
            for col, value in enumerate(cells):
                display = str(value) if col == 0 else f"{value:,.0f}" if col == 4 else f"{value:.1%}" if np.isfinite(value) else "—"
                item = QTableWidgetItem(display)
                item.setToolTip(self.metric_tips[col])
                if col in (1, 2, 3):
                    item.setForeground(QColor(score_color(value)))
                self.metrics.setItem(row, col, item)
            if not np.isfinite(r):
                statements.append(f"{name}: no annotated pixels; class not evaluated.")
            elif not np.isfinite(p):
                statements.append(f"{name}: never predicted; all {result['totals'][row]:,.0f} annotated pixels missed.")
            else:
                statements.append(f"{name}: {1-r:.1%} of actual pixels missed; {1-p:.1%} of predictions of this class incorrect.")
        self.metrics.setFixedHeight(40 + 32 * min(len(values), 8))
        self.diagnosis.setText("\n".join(statements))
        report = self.record.get("report") or self.record["payload"].get("grouped_training") or {}
        context = (f"Grouped validation · {report.get('n_cubes', '?')} cubes · {report.get('n_splits', '?')} folds" if validated else f"Current ROI preview · up to {OVERVIEW_MAX_PIXELS_PER_CLASS:,} pixels per class · may include training data")
        self.coverage.setText(f"{context}\n{sum(result['totals']):,.0f} annotated pixels evaluated · {sum(result['totals'] > 0)}/{len(values)} classes represented\nEvaluated ROI count: not recorded in this summary.\nPer-cube stability: unavailable; separate cube scores are not recorded.\nTraining / validation gap: unavailable; paired scores are not recorded.")
        draw_health(self.figure, result)
        background = self.page.palette().color(QPalette.Window).name()
        self.figure.set_facecolor(background)
        for ax in self.figure.axes:
            ax.set_facecolor(background)
        self.canvas.draw_idle()
        self._wrap_tooltips()
