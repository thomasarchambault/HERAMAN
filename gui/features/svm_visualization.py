# Rôle : Produire des visualisations en lecture seule de la géométrie de décision SVM.

"""Read-only SVM diagnostics. Geometry lives AFTER the saved preprocessing."""
from __future__ import annotations

import copy
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC
from PyQt5.QtCore import QObject, QRunnable, QThreadPool, Qt, pyqtSignal
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
                            QPushButton, QCheckBox)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT


BG, TEXT, MUTED = "#2b2b2b", "#f0f0f0", "#b8b8b8"

RBF_VIEW_HELP = (
    "Colored dots: sampled pixels from the current ROI dataset, grouped by true class.\n"
    "Horizontal position: model score for the selected class pair, not a probability.\n"
    "Zero: decision boundary. Yellow band: margin from -1 to +1.\n"
    "Red crosses: wrong side of this boundary, not necessarily multiclass errors.\n"
    "White circles on a separate line: saved training support vectors that help define\n"
    "the model boundaries (up to 600 shown). They do not outline the colored dots.\n"
    "Support vectors can lie on or inside the margin, or on the wrong side. In a\n"
    "multiclass model, some support other comparisons and may lie beyond this margin.\n"
    "Vertical spacing is only for readability; overlapping circles can hide each other.\n"
    "Percentages describe the sampled ROI pixels, not the support vectors."
)
LINEAR_VIEW_HELP = (
    "Background colors: final classes predicted by the saved Linear SVM.\n"
    "Dots: sampled pixels from the current ROI dataset, colored by their true class.\n"
    "White lines: decision boundaries. Red crosses: incorrectly classified pixels.\n"
    "Axes: the first two components of the model space. When the saved model has exactly\n"
    "two components, this is its complete decision space and the boundaries are exact.\n"
    "The displayed ROI pixels are training data, not an independent test set."
)
RBF_MAP_HELP = (
    "Background colors: final classes predicted directly by the saved RBF SVM.\n"
    "Dots: sampled pixels from the current ROI dataset, colored by their true class.\n"
    "White curves: boundaries between adjacent predicted regions. Red crosses: errors.\n"
    "Axes: the two PCA components used by the saved model; no display projection is added.\n"
    "Optional white circles: actual saved training support vectors (up to 1,000 shown).\n"
    "The curves are traced from a dense grid of direct model predictions, so their visible\n"
    "smoothness depends slightly on grid resolution, but the colored regions come from the model.\n"
    "The displayed ROI pixels are training data, not an independent test set."
)
LINEAR_PAIR_HELP = (
    "Each axis is an actual decision score produced by the saved LinearSVC.\n"
    "The diagonal is the exact tie boundary between the two selected classes:\n"
    "above it, the vertical-axis class wins; below it, the horizontal-axis class wins.\n"
    "Dots are sampled ROI pixels from those two true classes. Red crosses indicate\n"
    "pixels for which another class has the highest final multiclass score.\n"
    "No display PCA, reconstructed plane, or prediction grid is used."
)


class _Signals(QObject):
    finished = pyqtSignal(int, object, str)


class _Job(QRunnable):
    def __init__(self, token, payload, path, a, b):
        super().__init__()
        self.signals = _Signals()
        self.args = payload, path, a, b
        self.token = token

    def run(self):
        try:
            result = build_confidence_view(self.args[0], self.args[1], self.args[2])
            self.signals.finished.emit(self.token, result, "")
        except Exception as exc:
            self.signals.finished.emit(self.token, None, str(exc))


class SvmVisualizationWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.record = None
        self.token = 0
        self.busy = False
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        self.mode = QLabel()
        self.first = QComboBox()
        self.second = QComboBox()
        self.generate = QPushButton("Generate visualization")
        self.support_toggle = QCheckBox("Support vectors")
        self.support_toggle.setToolTip(RBF_VIEW_HELP)
        self.view_help = QLabel("ⓘ How to read this view")
        self.view_help.setToolTip(RBF_VIEW_HELP)
        controls.addWidget(self.mode)
        controls.addWidget(self.first)
        controls.addWidget(self.second)
        controls.addWidget(self.generate)
        controls.addWidget(self.support_toggle)
        controls.addWidget(self.view_help)
        controls.addStretch()
        layout.addLayout(controls)
        self.status = QLabel("Select a model.")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.PlainText)
        layout.addWidget(self.status)
        self.figure = Figure(figsize=(9, 4), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        layout.addWidget(self.canvas, 1)
        self.point_details = QLabel("Click a point to inspect its actual score.")
        self.point_details.setTextFormat(Qt.PlainText)
        self.point_details.setWordWrap(True)
        layout.addWidget(self.point_details)
        self.canvas.mpl_connect("pick_event", self.inspect_point)
        self.canvas.setToolTip(RBF_VIEW_HELP)
        self.generate.clicked.connect(self.start)
        self.first.currentIndexChanged.connect(self.invalidate)
        self.second.currentIndexChanged.connect(self.invalidate)
        self.support_toggle.toggled.connect(self.redraw)

    def redraw(self, *_):
        result = getattr(self, "result", None)
        if result is not None:
            draw_visualization(self.figure, result, self.support_toggle.isChecked())
            self.canvas.draw_idle()

    def invalidate(self, *_):
        self.result = None
        self.token += 1
        self.figure.clear()
        self.canvas.draw_idle()
        linear = bool(
            self.record and self.record.get("payload", {}).get("svm_type") == "linear"
        )
        linear_pair = bool(getattr(self, "linear_pair_mode", False))
        rbf_map = bool(getattr(self, "rbf_map_mode", False))
        self.point_details.setText(
            "Diagonal = exact score tie · red crosses = a third class wins"
            if linear_pair else
            "Background = model prediction · dots = true ROI class · red crosses = errors"
            if linear or rbf_map else
            "Groups on opposite sides = good separation · red crosses = wrong side"
        )
        self.status.setText(
            "Choose two classes, then generate their exact score boundary · ROI data"
            if linear_pair else
            "Generate the final multiclass decision map · ROI data"
            if linear or rbf_map else
            "Choose classes, then generate the view · ROI data"
        )
        self.status.setText("Choose a class, then show confidence and decisions · all model components")
        self.point_details.setText("ROI preview, not independent validation · saved Platt calibration when available")

    def set_record(self, record):
        self.record = record
        for combo in [self.first, self.second]:
            combo.blockSignals(True)
            combo.clear()
        payload = record.get("payload", {}) if record else {}
        names = [str(v) for v in payload.get("class_names", [])]
        self.first.addItems(names)
        self.second.addItems(names)
        default_a = names.index("fat") if "fat" in names else 0
        default_b = names.index("meat") if "meat" in names else (1 if len(names) > 1 else -1)
        self.first.setCurrentIndex(default_a)
        self.second.setCurrentIndex(default_b)
        rbf = payload.get("svm_type") == "rbf"
        model = payload.get("model")
        svm = getattr(model, "named_steps", {}).get("svm") if model is not None else None
        linear_pair = isinstance(svm, LinearSVC) and svm.coef_.shape[1] != 2
        rbf_map = isinstance(svm, SVC) and int(getattr(svm, "n_features_in_", 0)) == 2
        self.linear_pair_mode = linear_pair
        self.rbf_map_mode = rbf_map
        help_text = RBF_MAP_HELP if rbf_map else RBF_VIEW_HELP if rbf else LINEAR_PAIR_HELP if linear_pair else LINEAR_VIEW_HELP
        self.view_help.setToolTip(help_text)
        self.canvas.setToolTip(help_text)
        self.support_toggle.setVisible(rbf)
        self.second.setVisible((rbf and not rbf_map) or linear_pair)
        self.first.setVisible((rbf and not rbf_map) or linear_pair)
        self.mode.setText("Compare:" if rbf and not rbf_map else "Exact pairwise boundary:" if linear_pair else "Decision map:")
        self.generate.setText("Show")
        if (not rbf or rbf_map) and not linear_pair:
            self.mode.setText("Final classification · all classes")
        for combo in [self.first, self.second]:
            combo.blockSignals(False)
        self.invalidate()
        self.first.setVisible(True)
        self.second.setVisible(False)
        self.support_toggle.setVisible(False)
        self.mode.setText("Class to inspect:")
        self.view_help.setToolTip("All saved model components are used. No 2D PCA slice. Saved Platt calibration is applied when available. ROI previews may include training data.")
        self.canvas.setToolTip(self.view_help.toolTip())
        self.status.setText("Show calibrated confidence and decisions across true ROI classes · all model components")
        self.point_details.setText("ROI preview, not independent validation. Fat decisions use 25% / 75% thresholds before spatial cleanup.")
        self.generate.setEnabled(bool(names) and not self.busy)
        if not names:
            self.status.setText("No readable SVM selected.")

    def start(self):
        if not self.record or self.busy:
            return
        a, b = self.first.currentIndex(), self.second.currentIndex()
        payload = self.record["payload"]
        model = payload["model"]
        svm = model.named_steps["svm"]
        rbf_pair = isinstance(svm, SVC) and int(getattr(svm, "n_features_in_", 0)) != 2
        pair_selection = False
        if pair_selection:
            if a == b:
                self.status.setText("Select two different classes.")
                return
            a, b = sorted([a, b])
            self.first.setCurrentIndex(a)
            self.second.setCurrentIndex(b)
        self.token += 1
        self.busy = True
        self.generate.setEnabled(False)
        self.status.setText("Computing the ROI sample and decision map in the background…")
        self.job = _Job(self.token, self.record["payload"], self.record["dataset_path"], a, b)
        self.job.signals.finished.connect(self.finish)
        QThreadPool.globalInstance().start(self.job)

    def finish(self, token, result, error):
        self.busy = False
        self.generate.setEnabled(bool(self.record and self.first.count()))
        if token != self.token:
            return
        if error:
            self.figure.clear()
            self.canvas.draw_idle()
            self.status.setText("Visualization unavailable: " + error)
            return
        self.result = result
        self.redraw()
        if result.get('confidence_view'):
            self.status.setText(f"{result['source']} · {len(result['y']):,} ROI pixels · all model components")
            self.point_details.setText("ROI preview may include training pixels. Displayed decisions precede spatial cleanup. Other classes in fat-only Platt models retain raw SVM routing, not calibrated probabilities.")
            return
        if result.get("linear_pair"):
            self.status.setText(
                f"Exact LinearSVC boundary · {result['class_a']} vs {result['class_b']} · "
                f"{len(result['y']):,} sampled ROI pixels"
            )
            self.status.setToolTip(
                f"Showing {len(result['y']):,}/{result['eligible']:,} finite pixels from the "
                f"two selected true classes; {result['omitted']:,} other or invalid pixels excluded. "
                "Both axes are saved-model decision scores. The diagonal is the exact pairwise tie. "
                "These are training pixels, not an independent test set."
            )
            self.point_details.setText(
                "Diagonal = exact score tie · red crosses = a third class wins final classification"
            )
            self.canvas.setToolTip(LINEAR_PAIR_HELP)
            return
        if result.get("linear_map"):
            qualifier = "exact model space" if result["exact_model_space"] else "2D model-space slice"
            self.status.setText(
                f"Linear decision map · {len(result['xy']):,} sampled ROI pixels · {qualifier}"
            )
            self.status.setToolTip(
                f"Showing {len(result['xy']):,}/{result['eligible']:,} finite, known-class pixels. "
                f"{result['omitted']:,} pixels were excluded. The display retains "
                f"{100 * result['variance']:.2f}% of the model-space variance. "
                "Background = final model prediction; point color = true ROI label. "
                "These are training pixels, not an independent test set."
            )
            self.point_details.setText(
                "Background = model prediction · dots = true ROI class · red crosses = errors"
            )
            self.canvas.setToolTip(LINEAR_VIEW_HELP)
            return
        if result.get("rbf_map"):
            self.status.setText(
                f"RBF decision map · {len(result['xy']):,} sampled ROI pixels · exact 2D model space"
            )
            self.status.setToolTip(
                f"Showing {len(result['xy']):,}/{result['eligible']:,} finite, known-class pixels. "
                f"{result['omitted']:,} pixels were excluded. The saved PCA retains "
                f"{100 * result['variance']:.2f}% of spectral variance. "
                "Every colored grid cell is a direct prediction by the saved RBF SVM; "
                "white curves separate adjacent predicted regions. These ROI pixels are training data."
            )
            self.point_details.setText(
                "Background = RBF prediction · dots = true ROI class · red crosses = errors"
            )
            self.canvas.setToolTip(RBF_MAP_HELP)
            return
        signed = result["signed"]
        signature = self.record["payload"].get("dataset_signature") or {}
        try:
            stat = self.record["dataset_path"].stat()
            matches = (signature.get("size_bytes") == stat.st_size
                       and signature.get("mtime_ns") == stat.st_mtime_ns)
        except OSError:
            matches = False
        state = "signature matches saved dataset" if matches else "dataset changed or provenance unknown"
        distance = ""
        if result["norm"] is not None and result["norm"] > 0:
            distance = f" Linear model-space distance to boundary = f(x)/{result['norm']:.4g}."
        self.status.setToolTip(
            f"Current ROI data — {state}. Negative: {result['negative']}; positive: {result['positive']}. "
            f"Showing {len(signed):,}/{result['eligible']:,} eligible samples; "
            f"{result['omitted']:,} excluded (other/unknown classes or non-finite spectra). "
            f"Wrong side: {100*np.mean(signed < 0):.1f}%; inside/on margin, including errors: "
            f"{100*np.mean(signed <= 1):.1f}%. Display PCA retains {100*result['variance']:.1f}% variance. "
            f"Saved support vectors shown: {len(result['support_scores'])}/{result['support_total']}."
            + distance + (" Batch-dependent offset normalization: scores depend on the loaded dataset."
                          if result['normalization'] == 'snv_offset' else "")
        )
        self.status.setText(
            f"ROI · {len(signed):,} points · "
            + ("two-class comparison" if result['norm'] is None else "one boundary, not the final multiclass prediction")
        )

    def inspect_point(self, event):
        details = getattr(event.artist, "svm_details", None)
        if details and len(event.ind):
            self.point_details.setText(details[int(event.ind[0])])


def decision_view(svm, a, b):
    """Return a score oriented toward b (SVC pair), or a (LinearSVC OVR)."""
    n = len(svm.classes_)
    if isinstance(svm, SVC):
        if not 0 <= a < b < n:
            raise ValueError("Select two distinct classes in model order.")
        estimator = copy.copy(svm)
        estimator.decision_function_shape = "ovo"
        column = sum(n - i - 1 for i in range(a)) + b - a - 1

        def score(z):
            values = estimator.decision_function(z)
            return values if n == 2 else -values[:, column]
        return score, None
    if isinstance(svm, LinearSVC):
        row = 0 if n == 2 else a
        sign = (-1 if a == 0 else 1) if n == 2 else 1
        w = sign * svm.coef_[row]
        intercept = sign * svm.intercept_[row]
        return lambda z: z @ w + intercept, float(np.linalg.norm(w))
    raise ValueError("Only fitted SVC and LinearSVC models are supported.")


def load_model_dataset(payload, dataset_path):
    from processing.pipeline.svm_classifier import prepare_svm_features

    with np.load(dataset_path, allow_pickle=True) as data:
        x = np.asarray(data["X"], dtype=np.float32)
        y = np.asarray(data["y"]).astype(str)
        bands = np.asarray(payload["band_indices"], dtype=int)
        if x.ndim != 2 or y.shape != (len(x),):
            raise ValueError("Dataset must contain X (samples × bands) and y (class names).")
        if "wavelengths" not in data.files:
            raise ValueError("Dataset wavelengths are missing; spectral alignment cannot be verified.")
        wavelengths = np.asarray(data["wavelengths"])
        expected = np.asarray(payload["selected_wavelengths"])
        if (len(wavelengths) != x.shape[1] or len(bands) != len(expected)
                or np.any(bands < 0) or np.any(bands >= x.shape[1])
                or not np.allclose(wavelengths[bands], expected, atol=0.1, rtol=0)):
            raise ValueError("Dataset spectral bands do not match this model.")
        # Normalize before sampling: legacy offset normalization depends on the batch.
        x = prepare_svm_features(
            x[:, bands], expected, payload.get("normalization_mode", "none"),
            include_ratio=payload.get("ratio_feature") is not None,
        )
    return x, y


def build_confidence_view(payload, dataset_path, selected=0, limit=50_000):
    """Evaluate actual full-dimensional predictions, never a reconstructed PCA plane."""
    from shared.svm_calibration import platt_probabilities, calibration_decision_scores, fat_membership_labels
    x,y=load_model_dataset(payload,dataset_path)
    names=[str(v) for v in payload['class_names']]
    if not 0 <= selected < len(names): raise ValueError('Select a known class.')
    valid=np.isfinite(x).all(axis=1)&np.isin(y,names)
    rng=np.random.default_rng(42)
    # Keep natural class proportions in this preview so precision is interpretable.
    eligible=np.flatnonzero(valid)
    ids=np.sort(rng.choice(eligible,min(len(eligible),limit),replace=False))
    if not len(ids): raise ValueError('No compatible ROI pixels.')
    model=payload['model']; cal=payload.get('platt_calibration')
    calibrated=bool(cal) and (cal.get('target')!='fat' or selected==int(cal['fat_index']))
    values=[]; raw_predictions=[]
    for start in range(0,len(ids),4096):
        features=x[ids[start:start+4096]]
        if calibrated:
            p=platt_probabilities(cal,calibration_decision_scores(model,features,cal))
            values.append(p[:,selected])
        else:
            scores=np.asarray(model.decision_function(features))
            values.append(scores[:,selected] if scores.ndim==2 else scores*(1 if selected==1 else -1))
            raw_predictions.append(np.asarray(model.predict(features)))
    values=np.concatenate(values); truth=y[ids]==names[selected]
    is_fat=names[selected].lower()=='fat'
    if calibrated and is_fat:
        decision=fat_membership_labels(values)
    elif calibrated:
        decision=(values>=.5).astype(np.int8)
    else:
        decision=(np.concatenate(raw_predictions)==model.classes_[selected]).astype(np.int8)
    thresholds=np.linspace(0,1,101) if calibrated else np.linspace(values.min(),values.max(),101)
    precision=[];recall=[]
    for threshold in thresholds:
        chosen=values>=threshold;tp=np.sum(chosen&truth)
        precision.append(tp/chosen.sum() if chosen.any() else np.nan)
        recall.append(tp/truth.sum() if truth.any() else np.nan)
    composition=np.array([[np.mean(decision[y[ids]==name]==code) if np.any(y[ids]==name) else np.nan
                           for code in (0,2,1)] for name in names])
    source='Saved Platt probabilities' if calibrated else 'Raw SVM scores · no calibrated probability for this class'
    return dict(confidence_view=True,names=names,selected=names[selected],y=y[ids],values=values,
                calibrated=calibrated,is_fat=is_fat,decision=decision,composition=composition,
                thresholds=thresholds,precision=np.asarray(precision),recall=np.asarray(recall),source=source)


def draw_confidence_view(figure,result):
    figure.clear();figure.set_layout_engine('constrained');figure.set_facecolor(BG)
    grid=figure.add_gridspec(2,2,height_ratios=[1.1,1])
    hist=figure.add_subplot(grid[0,:]);bars=figure.add_subplot(grid[1,0]);curve=figure.add_subplot(grid[1,1])
    axes=(hist,bars,curve)
    for ax in axes:
        ax.set_facecolor(BG);ax.tick_params(colors=MUTED,labelsize=9)
        for spine in ax.spines.values(): spine.set_color('#555555')
        ax.grid(alpha=.12)
    calibrated=result['calibrated'];values=result['values'];names=result['names'];selected=result['selected']
    bins=np.linspace(0,1,41) if calibrated else np.linspace(values.min()-.001,values.max()+.001,41)
    colors=__import__('matplotlib').colormaps['tab10']
    for i,name in enumerate(names):
        v=values[result['y']==name]
        if len(v): hist.hist(v,bins=bins,weights=np.full(len(v),100/len(v)),histtype='step',linewidth=1.8,color=colors(i%10),label=f'True {name} · {len(v):,}')
    selective=calibrated and result['is_fat']
    if selective:
        hist.axvspan(.25,.75,color='#edbd62',alpha=.16,label='Unsure (25–75%)')
        for t in (.25,.75):hist.axvline(t,color='#edbd62',linestyle='--',linewidth=1)
    hist.set_title(f'How confidently does the model recognize {selected}?',color=TEXT,loc='left')
    xlabel=f'P({selected}) · saved Platt calibration' if calibrated else f'Raw decision score for {selected} · not probability'
    hist.set_xlabel(xlabel,color=MUTED);hist.set_ylabel('% within each true class',color=MUTED)
    hist.legend(fontsize=8,facecolor=BG,labelcolor=TEXT)
    left=np.zeros(len(names))
    for column,(label,color) in enumerate(((f'Non-{selected}','#69a6cf'),('Unsure','#edbd62'),(selected,'#69c99a'))):
        width=100*np.nan_to_num(result['composition'][:,column])
        bars.barh(names,width,left=left,color=color,label=label)
        for row,w in enumerate(width):
            if w>=8:bars.text(left[row]+w/2,row,f'{w:.0f}%',ha='center',va='center',fontsize=8,color='#202020')
        left+=width
    bars.set_xlim(0,100);bars.invert_yaxis()
    bars.set_title('Decisions by true class',color=TEXT,loc='left')
    bars.set_xlabel('% of ROI pixels · before spatial cleanup',color=MUTED)
    bars.legend(fontsize=8,facecolor=BG,labelcolor=TEXT,loc='upper center',bbox_to_anchor=(.5,-.2),ncol=3)
    scale=100 if calibrated else 1
    curve.plot(result['thresholds']*scale,result['precision']*100,color='#69c99a',label='Precision: predicted positives that are correct')
    curve.plot(result['thresholds']*scale,result['recall']*100,color='#69a6cf',label='Recall: true positives recovered')
    if selective:curve.axvline(75,color='#edbd62',linestyle='--',label='Current fat threshold: 75%')
    curve.set_ylim(0,103);curve.set_title('Effect of the decision threshold',color=TEXT,loc='left')
    curve.set_xlabel('Probability threshold (%)' if calibrated else 'Raw score threshold',color=MUTED)
    curve.set_ylabel('%',color=MUTED);curve.legend(fontsize=7,facecolor=BG,labelcolor=TEXT)
    figure.suptitle('Full model · all PCA components · ROI preview, not independent validation',color=MUTED,fontsize=10)


def build_visualization(payload, dataset_path, a=0, b=1, limit=None):
    model = payload["model"]
    svm = model.named_steps["svm"]
    if limit is None:
        limit = 12000
    names = [str(v) for v in payload["class_names"]]
    score, norm = decision_view(svm, a, b)
    x, y = load_model_dataset(payload, dataset_path)
    valid = np.isfinite(x).all(axis=1) & np.isin(y, names)
    rbf_map = isinstance(svm, SVC) and int(getattr(svm, "n_features_in_", 0)) == 2
    if isinstance(svm, SVC) and not rbf_map:
        valid &= np.isin(y, [names[a], names[b]])
    linear_pair = isinstance(svm, LinearSVC) and svm.coef_.shape[1] != 2
    if linear_pair:
        valid &= np.isin(y, [names[a], names[b]])
    rng = np.random.default_rng(42)
    indices = []
    present = np.unique(y[valid])
    for name in present:
        candidates = np.flatnonzero(valid & (y == name))
        indices.extend(rng.choice(candidates, min(len(candidates), max(1, limit // len(present))), replace=False))
    indices = np.asarray(indices, dtype=int)
    if len(indices) < 3:
        raise ValueError("At least three finite, known-class samples are required.")
    z = model[:-1].transform(x[indices]) if len(model.steps) > 1 else x[indices]
    y = y[indices]
    if isinstance(svm, LinearSVC):
        raw = svm.decision_function(z)
        matrix = np.c_[-raw, raw] if raw.ndim == 1 else raw
        if linear_pair:
            if not 0 <= a < b < matrix.shape[1]:
                raise ValueError("Select two distinct classes in model order.")
            predicted = svm.predict(z)
            class_to_column = {value: column for column, value in enumerate(svm.classes_)}
            predicted_columns = np.asarray(
                [class_to_column[value] for value in predicted], dtype=int
            )
            predicted_names = np.asarray(names, dtype=str)[predicted_columns]
            pair_winner = np.where(matrix[:, b] > matrix[:, a], names[b], names[a])
            return dict(
                linear_pair=True, score_a=matrix[:, a], score_b=matrix[:, b], y=y,
                class_a=names[a], class_b=names[b], pair_winner=pair_winner,
                predicted_names=predicted_names,
                other_winner=~np.isin(predicted_names, [names[a], names[b]]),
                eligible=int(valid.sum()), omitted=int(len(x) - valid.sum()),
                normalization=payload.get("normalization_mode", "none"),
            )
        if z.shape[1] < 2 or not np.isfinite(z).all():
            raise ValueError("A 2D decision map requires at least two finite model-space features.")
        exact_model_space = z.shape[1] == 2
        if exact_model_space:
            xy = z
            projection = None
            saved_pca = model.named_steps.get("pca")
            variance = float(np.sum(getattr(saved_pca, "explained_variance_ratio_", [1.0])))
            axis_prefix = "Model PCA component" if saved_pca is not None else "Model feature"
        else:
            projection = PCA(n_components=2, svd_solver="full").fit(z)
            xy = projection.transform(z)
            variance = float(projection.explained_variance_ratio_.sum())
            axis_prefix = "Display PCA component"
        lo, hi = np.quantile(xy, [0.005, 0.995], axis=0)
        pad = np.maximum((hi - lo) * .10, .1)
        gx, gy = np.meshgrid(
            np.linspace(lo[0] - pad[0], hi[0] + pad[0], 180),
            np.linspace(lo[1] - pad[1], hi[1] + pad[1], 140),
        )
        grid_xy = np.c_[gx.ravel(), gy.ravel()]
        plane = grid_xy if projection is None else projection.inverse_transform(grid_xy)
        predicted = svm.predict(z)
        surface_predictions = svm.predict(plane)
        class_to_column = {value: column for column, value in enumerate(svm.classes_)}
        predicted_columns = np.asarray([class_to_column[value] for value in predicted], dtype=int)
        surface = np.asarray(
            [class_to_column[value] for value in surface_predictions], dtype=int
        ).reshape(gx.shape)
        true_columns = np.asarray([names.index(label) for label in y], dtype=int)
        predicted_names = np.asarray(names, dtype=str)[predicted_columns]
        return dict(
            linear_map=True, xy=xy, y=y, predicted_names=predicted_names,
            correct=predicted_columns == true_columns, gx=gx, gy=gy, surface=surface,
            class_names=names, exact_model_space=exact_model_space, variance=variance,
            axis_prefix=axis_prefix,
            eligible=int(valid.sum()), omitted=int(len(x) - valid.sum()),
            normalization=payload.get("normalization_mode", "none"),
            support_scores=np.empty(0), support_names=np.empty(0, dtype=str),
        )
    if rbf_map:
        if z.shape[1] != 2 or not np.isfinite(z).all():
            raise ValueError("The saved RBF model does not expose a finite 2D model space.")
        xy = z
        lo, hi = np.quantile(xy, [0.005, 0.995], axis=0)
        pad = np.maximum((hi - lo) * .10, .1)
        gx, gy = np.meshgrid(
            np.linspace(lo[0] - pad[0], hi[0] + pad[0], 260),
            np.linspace(lo[1] - pad[1], hi[1] + pad[1], 190),
        )
        grid = np.c_[gx.ravel(), gy.ravel()]

        def predict_batched(values, batch=512):
            return np.concatenate([
                svm.predict(values[start:start + batch])
                for start in range(0, len(values), batch)
            ])

        predicted = svm.predict(z)
        surface_predictions = predict_batched(grid)
        class_to_column = {value: column for column, value in enumerate(svm.classes_)}
        predicted_columns = np.asarray(
            [class_to_column[value] for value in predicted], dtype=int
        )
        surface = np.asarray(
            [class_to_column[value] for value in surface_predictions], dtype=int
        ).reshape(gx.shape)
        true_columns = np.asarray([names.index(label) for label in y], dtype=int)
        predicted_names = np.asarray(names, dtype=str)[predicted_columns]
        labels = np.repeat(np.asarray(names, dtype=str), svm.n_support_)
        support_total = len(svm.support_vectors_)
        support_ids = rng.choice(
            support_total, min(support_total, 1000), replace=False
        ) if support_total else np.empty(0, dtype=int)
        saved_pca = model.named_steps.get("pca")
        variance = float(np.sum(getattr(saved_pca, "explained_variance_ratio_", [1.0])))
        return dict(
            rbf_map=True, xy=xy, y=y, predicted_names=predicted_names,
            correct=predicted_columns == true_columns, gx=gx, gy=gy, surface=surface,
            class_names=names, exact_model_space=True, variance=variance,
            axis_prefix="Model PCA component" if saved_pca is not None else "Model feature",
            eligible=int(valid.sum()), omitted=int(len(x) - valid.sum()),
            normalization=payload.get("normalization_mode", "none"),
            support_xy=svm.support_vectors_[support_ids],
            support_names=labels[support_ids], support_total=support_total,
        )
    if z.shape[1] < 2 or not np.isfinite(z).all():
        raise ValueError("A 2D view requires at least two finite model-space features.")
    if not np.any(np.var(z, axis=0) > 0):
        raise ValueError("No variation in the selected model-space data.")
    projection = PCA(n_components=2, svd_solver="full").fit(z)
    xy = projection.transform(z)
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    pad = np.maximum((hi - lo) * .12, .1)
    gx, gy = np.meshgrid(np.linspace(*(lo[0]-pad[0], hi[0]+pad[0]), 45),
                         np.linspace(*(lo[1]-pad[1], hi[1]+pad[1]), 45))
    plane = projection.inverse_transform(np.c_[gx.ravel(), gy.ravel()])
    # Bounded batches avoid large kernel working arrays.
    evaluate = lambda values: np.concatenate([score(values[i:i+128]) for i in range(0, len(values), 128)])
    scores = evaluate(z)
    positive = names[b] if isinstance(svm, SVC) else names[a]
    signed = np.where(y == positive, 1, -1) * scores
    support_xy = np.empty((0, 2))
    support_scores = np.empty(0)
    support_names = np.empty(0, dtype=str)
    support_total = 0
    if isinstance(svm, SVC):
        # These are saved training vectors, NOT indices into today's ROI dataset.
        labels = np.repeat(np.asarray(names), svm.n_support_)
        ids = np.flatnonzero(np.isin(labels, [names[a], names[b]]))
        support_total = len(ids)
        ids = rng.choice(ids, min(len(ids), 600), replace=False)
        vectors = svm.support_vectors_[ids]
        support_xy = projection.transform(vectors)
        support_scores = evaluate(vectors)
        support_names = labels[ids]
    return dict(xy=xy, y=y, scores=scores, signed=signed, gx=gx, gy=gy,
                surface=evaluate(plane).reshape(gx.shape), positive=positive,
                negative=names[a] if isinstance(svm, SVC) else "Other classes",
                support_xy=support_xy, support_scores=support_scores,
                support_names=support_names, support_total=support_total,
                variance=float(projection.explained_variance_ratio_.sum()),
                norm=norm, eligible=int(valid.sum()), omitted=int(len(x)-valid.sum()),
                normalization=payload.get("normalization_mode", "none"))


def margin_percentages(signed):
    """Largest-remainder rounding in tenths; displayed categories sum to 100.0%."""
    signed = np.asarray(signed, dtype=float)
    if signed.ndim != 1 or not signed.size or not np.isfinite(signed).all():
        raise ValueError("Margin scores must be a non-empty finite vector.")
    counts = np.array([np.count_nonzero(signed > 1),
                       np.count_nonzero((signed >= 0) & (signed <= 1)),
                       np.count_nonzero(signed < 0)], dtype=np.int64)
    units, remainders = np.divmod(counts * 1000, signed.size)
    missing = int(1000 - units.sum())
    units[np.argsort(-remainders, kind="stable")[:missing]] += 1
    return units / 10.0


def draw_visualization(figure, result, show_support=False):
    """Draw the visualization appropriate for the saved SVM kernel."""
    if result.get('confidence_view'):
        return draw_confidence_view(figure, result)
    if result.get("linear_pair"):
        return draw_linear_pair(figure, result)
    if result.get("linear_map") or result.get("rbf_map"):
        return draw_decision_map(figure, result, show_support)
    figure.clear()
    figure.set_layout_engine("constrained")
    figure.set_facecolor(BG)
    ax = figure.subplots()
    ax.set_facecolor(BG)
    colors = ["#66b6ff", "#f1ac70", "#7ad5aa", "#c5a1ee", "#ead479"]
    names = list(np.unique(result["y"]))
    names = [n for n in names if n != result["positive"]] + [result["positive"]]
    names = [n for n in names if np.any(result["y"] == n)]
    all_scores = result["scores"]
    if show_support and len(result["support_scores"]):
        all_scores = np.r_[all_scores, result["support_scores"]]
    lo, hi = min(float(all_scores.min()), -1.5), max(float(all_scores.max()), 1.5)
    pad = (hi - lo) * .08
    ax.set_xlim(lo-pad, hi+pad)
    ax.axvspan(-1, 1, color="#edbd62", alpha=.18)
    for value in [-1, 1]:
        ax.axvline(value, color="#edbd62", linestyle="--", linewidth=1)
    ax.axvline(0, color="white", linewidth=1.5)
    ax.text(0, 1.02, "Boundary", ha="center", color="white", transform=ax.get_xaxis_transform())
    ax.text(.01, 1.10, "← " + result["negative"], color=MUTED, transform=ax.transAxes)
    ax.text(.99, 1.10, result["positive"] + " →", ha="right", color=MUTED, transform=ax.transAxes)
    rng = np.random.default_rng(42)
    labels = []
    ticks = []
    for row, name in enumerate(names):
        mask = result["y"] == name
        scores, signed = result["scores"][mask], result["signed"][mask]
        color = colors[row % len(colors)]
        labels.append(f"{name}\n← expected" if name != result["positive"] else f"{name}\nexpected →")
        ticks.append(row)
        positions = row + rng.uniform(-.14, .14, len(scores))
        points = ax.scatter(scores, positions, s=18, alpha=.65, color=color, picker=4)
        points.svm_details = [
            f"{name} · score {value:.3f} · " +
            ("wrong side" if margin < 0 else "within margin" if margin <= 1 else "beyond margin")
            for value, margin in zip(scores, signed)
        ]
        wrong = signed < 0
        ax.scatter(scores[wrong], positions[wrong], s=36, marker="x", color="#ff7777")
        beyond, within, wrong_percent = margin_percentages(signed)
        ax.text(1.01, row, f"{beyond:.1f}% beyond margin\n"
                f"{within:.1f}% within margin\n"
                f"{wrong_percent:.1f}% wrong side",
                transform=ax.get_yaxis_transform(), va="center", color="#dce2eb", fontsize=9)
        if show_support:
            support = result["support_names"] == name
            support_row = row + .36
            support_scores = result["support_scores"][support]
            ax.axhline(support_row, color="white", alpha=.12, linewidth=.7)
            points = ax.scatter(support_scores,
                                np.full(len(support_scores), support_row),
                                facecolors="none", edgecolors="white", s=28,
                                linewidths=.7, picker=4)
            points.svm_details = [
                f"{name} · saved training support vector · score {value:.3f}"
                for value in support_scores
            ]
            ticks.append(support_row)
            labels.append(f"Support vectors ({len(support_scores)})")
    ax.set_yticks(ticks, labels, color=TEXT)
    ax.set_ylim(len(names)-(.35 if show_support else .5), -.5)
    ax.tick_params(axis="x", colors=MUTED)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Actual model score · yellow band = margin (−1 to +1)", color=MUTED, labelpad=14)
    ax.grid(axis="x", alpha=.1)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_linear_pair(figure, result):
    """Plot two exact LinearSVC decision scores and their analytical tie boundary."""
    figure.clear()
    figure.set_layout_engine("constrained")
    figure.set_facecolor(BG)
    ax = figure.subplots()
    ax.set_facecolor(BG)
    a, b = result["class_a"], result["class_b"]
    x, y = result["score_a"], result["score_b"]
    low = min(float(np.min(x)), float(np.min(y)))
    high = max(float(np.max(x)), float(np.max(y)))
    pad = max((high - low) * .08, .1)
    low, high = low - pad, high + pad
    axis = np.linspace(low, high, 200)
    ax.fill_between(axis, low, axis, color="#533333", alpha=.55, label=f"{a} score > {b}")
    ax.fill_between(axis, axis, high, color="#334a56", alpha=.55, label=f"{b} score > {a}")
    ax.plot(axis, axis, color="#f1f3f4", linewidth=1.6, label="Exact tie boundary")
    colors = ["#efa2b5", "#e45b5b"]
    for name, color in zip([a, b], colors):
        mask = result["y"] == name
        points = ax.scatter(
            x[mask], y[mask], s=13, color=color, alpha=.48,
            linewidths=0, picker=4, label=f"True {name}",
        )
        points.svm_details = [
            f"True: {name} · final prediction: {prediction} · pair winner: {winner} · "
            f"score({a})={sa:.3f} · score({b})={sb:.3f}"
            for prediction, winner, sa, sb in zip(
                result["predicted_names"][mask], result["pair_winner"][mask],
                x[mask], y[mask],
            )
        ]
    other = result["other_winner"]
    if np.any(other):
        ax.scatter(
            x[other], y[other], s=28, marker="x", color="#ff8b83",
            alpha=.8, linewidths=.8, label="Another class wins final prediction",
        )
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Exact LinearSVC score boundary · {a} vs {b}", color=TEXT, loc="left", pad=10)
    ax.set_xlabel(f"Decision score · {a}", color=MUTED, labelpad=9)
    ax.set_ylabel(f"Decision score · {b}", color=MUTED, labelpad=9)
    ax.tick_params(colors=MUTED)
    ax.grid(alpha=.12)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#697078")
    legend = ax.legend(
        loc="best", frameon=True, facecolor="#303030",
        edgecolor="#697078", fontsize=8,
    )
    for label in legend.get_texts():
        label.set_color(TEXT)


def _draw_pairwise_region_boundaries(ax, result):
    """Trace each adjacent class pair once, independent of numeric class IDs."""
    surface = result["surface"]
    for first in range(len(result["class_names"])):
        for second in range(first + 1, len(result["class_names"])):
            active = (surface == first) | (surface == second)
            if not np.any(surface == first) or not np.any(surface == second):
                continue
            pair = np.ma.masked_where(~active, (surface == second).astype(float))
            ax.contour(
                result["gx"], result["gy"], pair, levels=[.5],
                colors="#e1e5e8", linewidths=1.15, alpha=.95,
            )


def draw_decision_map(figure, result, show_support=False):
    """Draw final multiclass regions in the complete two-dimensional model space."""
    figure.clear()
    figure.set_layout_engine("constrained")
    figure.set_facecolor(BG)
    ax = figure.subplots()
    ax.set_facecolor(BG)
    names = list(result["class_names"])
    point_colors = ["#c8a24e", "#efa2b5", "#e45b5b", "#55add8", "#8bcf91"]
    region_colors = ["#4a4025", "#563642", "#552d2d", "#294653", "#304c35"]
    levels = np.arange(len(names) + 1) - .5
    ax.contourf(
        result["gx"], result["gy"], result["surface"], levels=levels,
        colors=region_colors[:len(names)], alpha=.72,
    )
    if len(names) > 1:
        _draw_pairwise_region_boundaries(ax, result)
    for index, name in enumerate(names):
        mask = result["y"] == name
        correct = result["correct"][mask]
        xy = result["xy"][mask]
        accuracy = float(np.mean(correct)) if len(correct) else float("nan")
        points = ax.scatter(
            xy[:, 0], xy[:, 1], s=11,
            color=point_colors[index % len(point_colors)], alpha=.42,
            linewidths=0, picker=4, label=f"{name} · {accuracy:.1%} correct",
        )
        predicted = result["predicted_names"][mask]
        points.svm_details = [
            f"True: {name} · predicted: {prediction} · "
            f"component 1: {coords[0]:.3f} · component 2: {coords[1]:.3f}"
            for prediction, coords in zip(predicted, xy)
        ]
        wrong_xy = xy[~correct]
        if len(wrong_xy):
            ax.scatter(
                wrong_xy[:, 0], wrong_xy[:, 1], s=18, marker="x",
                color="#ff8b83", alpha=.55, linewidths=.7,
            )
    if show_support and result.get("rbf_map") and len(result["support_xy"]):
        support = result["support_xy"]
        points = ax.scatter(
            support[:, 0], support[:, 1], s=22, facecolors="none",
            edgecolors="#f4f5f6", alpha=.42, linewidths=.55, picker=4,
            label=f"Saved support vectors · {len(support):,}/{result['support_total']:,}",
        )
        points.svm_details = [
            f"Saved training support vector · class group: {name} · "
            f"component 1: {coords[0]:.3f} · component 2: {coords[1]:.3f}"
            for name, coords in zip(result["support_names"], support)
        ]
    exact = result["exact_model_space"]
    family = "RBF" if result.get("rbf_map") else "Linear"
    ax.set_title(
        f"Final {family} SVM decision regions · actual ROI data"
        + ("" if exact else " · 2D slice"),
        color=TEXT, loc="left", pad=10,
    )
    ax.set_xlabel(result["axis_prefix"] + " 1", color=MUTED, labelpad=9)
    ax.set_ylabel(result["axis_prefix"] + " 2", color=MUTED, labelpad=9)
    ax.tick_params(colors=MUTED)
    ax.grid(alpha=.12)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#697078")
    legend = ax.legend(
        loc="upper right", title="True ROI class", frameon=True,
        facecolor="#303030", edgecolor="#697078", fontsize=8,
    )
    legend.get_title().set_color(TEXT)
    for label in legend.get_texts():
        label.set_color(TEXT)
