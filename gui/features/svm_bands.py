# Rôle : Calculer et afficher l’importance exploratoire des bandes pour les modèles SVM.

"""Exploratory per-class permutation importance of normalized spectral bands."""
import numpy as np
from sklearn.metrics import f1_score
from PyQt5.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QComboBox, QCheckBox, QHBoxLayout, QFileDialog, QMessageBox
import csv
from gui.features.spectral_statistics import analyze_spectra
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from gui.features.svm_visualization import load_model_dataset
from shared.svm_calibration import predict_encoded


def band_importance(payload, path, repeats=3, per_class=100):
    x, y = load_model_dataset(payload, path)
    names = np.asarray(payload['class_names']).astype(str)
    rng = np.random.default_rng(42)
    valid = np.isfinite(x).all(axis=1)
    ids = []
    for name in names:
        candidates = np.flatnonzero(valid & (y == name))
        if not len(candidates):
            raise ValueError('Every model class needs finite ROI samples for band comparison.')
        ids.extend(rng.choice(candidates, min(len(candidates), per_class), replace=False))
    x, y = x[ids], y[ids]
    def evaluate(values):
        prediction = np.concatenate([predict_encoded(payload, values[i:i+128])
                                     for i in range(0, len(values), 128)])
        encoder = payload.get('label_encoder')
        labels = encoder.inverse_transform(prediction) if encoder is not None else names[prediction.astype(int)]
        return f1_score(y, labels, labels=names, average=None, zero_division=0)
    baseline = evaluate(x)
    drops = np.empty((x.shape[1], repeats, len(names)))
    for band in range(x.shape[1]):
        for repeat in range(repeats):
            shuffled = x.copy()
            shuffled[:, band] = x[rng.permutation(len(x)), band]
            drops[band, repeat] = (baseline - evaluate(shuffled))*100
    return dict(mean=drops.mean(axis=1), std=drops.std(axis=1), names=names,
                wavelengths=np.asarray(payload['selected_wavelengths']), baseline=baseline,
                samples=len(x), repeats=repeats)


class _Signals(QObject):
    done = pyqtSignal(int, object, str)


class _Job(QRunnable):
    def __init__(self, token, record, mode='roi', exclude=True):
        super().__init__()
        self.token, self.record, self.signals = token, record, _Signals()
        self.mode, self.exclude = mode, exclude

    def run(self):
        try:
            self.signals.done.emit(self.token, analyze_spectra(self.record['payload'], self.record['dataset_path'], self.mode, self.exclude), '')
        except Exception as exc:
            self.signals.done.emit(self.token, None, str(exc))


class BandImportanceWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.record, self.token, self.jobs, self.result = None, 0, {}, None
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        self.exclude = QCheckBox('Tissue classes only')
        self.exclude.setToolTip('Exclude background, petri, black ink and yellow ink. Uncheck to include every available model class.')
        self.exclude.setChecked(True)
        controls.addWidget(self.exclude)
        layout.addLayout(controls)
        self.unit_help = QLabel()
        self.unit_help.setWordWrap(True)
        layout.addWidget(self.unit_help)
        self.button = QPushButton('Analyze wavelengths')
        self.button.clicked.connect(self.start)
        layout.addWidget(self.button)
        self.caption = QLabel('Notebook method: Kruskal–Wallis + pairwise effects + redundancy')
        self.caption.setWordWrap(True)
        layout.addWidget(self.caption)
        self.view = QComboBox()
        self.view.addItems(['Which wavelengths separate two classes?', 'Which wavelengths distinguish classes overall?', 'Which wavelengths carry similar information?'])
        self.view.currentIndexChanged.connect(self.draw)
        layout.addWidget(self.view)
        self.reading_help = QLabel()
        self.reading_help.setWordWrap(True)
        layout.addWidget(self.reading_help)
        self.pair = QComboBox()
        self.pair.currentIndexChanged.connect(self.draw)
        layout.addWidget(self.pair)
        self.export = QPushButton('Export spectral statistics (CSV)')
        self.export.setEnabled(False)
        self.export.clicked.connect(self.export_csv)
        layout.addWidget(self.export)
        self.figure = Figure(figsize=(7, 3.5), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(330)
        self.canvas.hide()
        layout.addWidget(self.canvas)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.canvas.mpl_connect('motion_notify_event', self.hover)
        self.exclude.toggled.connect(self.invalidate)
        self.update_help()

    def update_help(self):
        self.unit_help.setText('Each ROI contributes its median spectrum: the middle pixel value at each wavelength, after spectral normalization.')
        guides = [
            (
                'Whether one class consistently has higher spectral values than the other (|Cliff’s delta|).',
                'High points or dark green cells. 1 means complete ordering; 0 means neither class tends to be higher, not necessarily identical distributions.',
                'Compare every ROI in class A with every ROI in B. Count A-higher minus B-higher, divide by all comparisons, then take the absolute value. Ties contribute zero.'
            ),
            (
                'Blue: differences between class ranks (Kruskal–Wallis eta²_H). Yellow: information about class labels (mutual information).',
                'Wavelength regions where both curves are high. A peak may distinguish only some classes: check the pair view.',
                'Blue ranks ROI values and measures between-class rank differences. Yellow estimates how often nearby spectral values share a class; it is divided by its largest value here, so 1 does not mean perfect separation.'
            ),
            (
                'How similarly two wavelengths vary across ROIs (absolute Spearman correlation).',
                'Dark blue blocks away from the diagonal: these bands may repeat information. The diagonal is each band compared with itself.',
                'Rank the ROIs at each wavelength, then correlate those rankings. 1 means matching or reversed rankings; 0 means no rank correlation, not proven independence. Blank cells mean an undefined correlation.'
            )
        ]
        what, seek, calculation = guides[self.view.currentIndex()]
        self.reading_help.setText(
            f'<b>What you see:</b> {what}<br>'
            f'<b>What to look for:</b> {seek}<br>'
            f'<b>How it is calculated:</b> {calculation}'
        )
        self.reading_help.setToolTip(
            'These are spectral statistics, not SVM accuracy. ROI medians from the same specimen can remain dependent. '
            'Dunn p-values on hover compare pooled mean ranks with tie correction; BH correction covers all bands and pairs. '
            'Those nominal p-values are exploratory and are not the values drawn in the pair chart.'
        )
        for i, guide in enumerate(guides):
            self.view.setItemData(i, guide[0] + '\nLook for: ' + guide[1], Qt.ToolTipRole)
        self.pair.setVisible(self.view.currentIndex() == 0 and self.result is not None)

    def invalidate(self, *_):
        self.token += 1
        self.result = None
        self.update_help()
        self.export.setEnabled(False)
        self.canvas.hide()
        self.summary.clear()
        self.caption.setText('Settings changed · run analysis')
        self.button.setEnabled(bool(self.record) and not self.jobs)

    def set_record(self, record):
        self.record = record
        self.invalidate()

    def start(self):
        if not self.record or self.jobs:
            return
        self.caption.setText('Computing spectral statistics…')
        self.button.setEnabled(False)
        job = _Job(self.token, self.record, 'roi', self.exclude.isChecked())
        self.jobs[self.token] = job
        job.signals.done.connect(self.finish)
        QThreadPool.globalInstance().start(job)

    def finish(self, token, result, error):
        self.jobs.pop(token, None)
        self.button.setEnabled(bool(self.record))
        if token != self.token:
            return
        if error:
            self.caption.setText('Analysis unavailable: ' + error)
            return
        self.result = result
        self.pair.blockSignals(True)
        self.pair.clear()
        self.pair.addItems([p.replace(' / ', ' vs ') for p in result['pairs']])
        self.pair.addItem(f"All included pairs ({len(result['pairs'])}) · heatmap")
        self.pair.blockSignals(False)
        self.export.setEnabled(True)
        m = result['metadata']
        unit = 'ROI medians' if m['mode']=='roi' else 'pixels'
        self.caption.setText(f"Spectral separation · {m['per_class']} {unit}/class · {m['sources'] if m['sources'] is not None else '?'} sources · model spectral normalization")
        self.caption.setText(self.caption.text() + '\nIncluded: ' + ', '.join(result['names'])
                             + f" · {len(result['pairs'])} pair(s)"
                             + (' · uncheck Tissue classes only to include background / petri / inks' if self.exclude.isChecked() else ''))
        self.caption.setToolTip('Not classifier importance. ROI medians grouped by (source, class, ROI ID). Equal class counts; fixed sample across all bands. Nominal Kruskal/Dunn p-values assume independence, which clustered ROIs/pixels do not establish. FDR covers all bands and all pairs. Values are descriptive, not proof of biological significance.')
        selected = ', '.join(f"{result['wavelengths'][j]:g}" for j in result['selected']) or 'none'
        self.summary.setText('Suggested bands to investigate (nm): ' + selected)
        self.summary.setToolTip('Exploratory MI minus mean absolute Spearman correlation heuristic. Not classical mRMR or a validated filter set. Neighboring bands may still be redundant.')
        self.draw()

    def export_csv(self):
        if self.result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Export spectral statistics', 'spectral_statistics.csv', 'CSV (*.csv)')
        if not path:
            return
        r = self.result
        fields = ['wavelength_nm', 'rank_eta_H_squared', 'kruskal_H', 'nominal_kw_q_BH', 'mutual_information', 'heuristic_rank', 'unit', 'observations_per_class', 'sources', 'normalization']
        fields += [f'{pair}_{metric}' for pair in r['pairs'] for metric in ['cliffs_delta', 'nominal_dunn_q_BH']]
        try:
            with open(path, 'w', newline='', encoding='utf-8-sig') as stream:
                writer = csv.writer(stream)
                writer.writerow(fields)
                for band, wavelength in enumerate(r['wavelengths']):
                    row = [wavelength, r['eta'][band], r['h'][band], r['kw_q'][band], r['mi'][band],
                           r['selected'].index(band)+1 if band in r['selected'] else '', r['metadata']['mode'],
                           r['metadata']['per_class'], r['metadata']['sources'], self.record['payload'].get('normalization_mode', 'none')]
                    row += [v for pair in range(len(r['pairs'])) for v in [r['delta'][pair,band], r['dunn_q'][pair,band]]]
                    writer.writerow(row)
        except OSError as exc:
            QMessageBox.warning(self, 'Export failed', str(exc))

    def draw(self, *_):
        self.update_help()
        if self.result is None:
            return
        r = self.result
        self.figure.clear()
        self.figure.set_facecolor('#24282e')
        self.figure.set_layout_engine('constrained')
        mode = self.view.currentIndex()
        ax = self.figure.subplots()
        ax.set_facecolor('#24282e')
        if mode == 0:
            values = np.abs(r['delta'])
            row = self.pair.currentIndex()
            if 0 <= row < len(r['pairs']):
                ax.plot(r['wavelengths'], values[row], color='#75d2a5', marker='o', markersize=4)
                ax.fill_between(r['wavelengths'], 0, values[row], color='#75d2a5', alpha=.15)
                ax.set_ylim(0, 1.05)
                ax.set_yticks([0,.5,1], ['0 · No rank dominance', '0.5', '1 · Complete ordering'], fontsize=8)
                ax.set_title(r['pairs'][row].replace(' / ', ' vs ') + " (Cliff’s delta; Dunn post-hoc on hover)", fontsize=10)
                ax.grid(alpha=.15)
            else:
                image = ax.imshow(values, cmap='YlGn', vmin=0, vmax=1, aspect='auto')
                ax.set_xticks(range(len(r['wavelengths'])), [f'{w:g}' for w in r['wavelengths']], rotation=60, fontsize=8)
                ax.set_yticks(range(len(r['pairs'])), r['pairs'], fontsize=9)
                ax.set_title('Darker green = stronger rank separation (|Cliff’s delta|)', fontsize=10)
                bar = self.figure.colorbar(image, ax=ax, fraction=.03, pad=.02)
                bar.set_label('Rank separation (0–1)', color='white')
                bar.ax.tick_params(colors='white')
            ax.set_xlabel('Wavelength (nm)')
        elif mode == 1:
            ax.plot(r['wavelengths'], r['eta'], marker='o', color='#66b6ff', label='Class differences (rank effect)')
            mi = r['mi']/r['mi'].max() if r['mi'].max()>0 else np.zeros_like(r['mi'])
            ax.plot(r['wavelengths'], mi, color='#edbd62', label='Class information (relative MI)')
            ax.set_ylim(0, 1.05)
            ax.set_xlabel('Wavelength (nm)')
            ax.set_title('Class differences (Kruskal–Wallis eta²_H) + information (MI)', fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(alpha=.15)
        else:
            image = ax.imshow(np.ma.masked_invalid(np.abs(r['corr'])), cmap='Blues', vmin=0, vmax=1, aspect='auto')
            ticks = np.arange(0,len(r['wavelengths']),2)
            ax.set_xticks(ticks, [f'{r["wavelengths"][i]:g}' for i in ticks], rotation=60, fontsize=8)
            ax.set_yticks(ticks, [f'{r["wavelengths"][i]:g}' for i in ticks], fontsize=8)
            ax.set_title('Bands that vary together (|Spearman correlation|)', fontsize=10)
            ax.set_xlabel('Wavelength (nm)')
        ax.tick_params(colors='white')
        ax.title.set_color('white')
        ax.xaxis.label.set_color('white')
        self.canvas.show()
        self.canvas.draw_idle()

    def hover(self, event):
        if self.result is None or event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        r = self.result
        column, row = int(round(event.xdata)), int(round(event.ydata))
        if self.view.currentIndex() == 0 and self.pair.currentIndex() < len(r['pairs']):
            row = self.pair.currentIndex()
            column = int(np.argmin(np.abs(r['wavelengths']-event.xdata)))
        if self.view.currentIndex()==0 and 0 <= column < len(r['wavelengths']) and 0 <= row < len(r['pairs']):
            delta = r['delta'][row,column]
            self.canvas.setToolTip(f"{r['pairs'][row]} · {r['wavelengths'][column]:g} nm\nCliff's delta: {delta:.3f} (positive: first class higher)\nNominal Dunn q: {r['dunn_q'][row,column]:.3g} (all bands/pairs BH)\nNominal KW q: {r['kw_q'][column]:.3g}\nClustered observations: q-values are exploratory, not independent-specimen evidence.")
