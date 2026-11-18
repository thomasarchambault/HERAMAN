"""Review labelled extraction regions and their measured spectra before export."""
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Polygon, Rectangle
from matplotlib.font_manager import FontProperties
from matplotlib.colors import to_rgba
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox

from gui.inspection_widgets import attach_spectrum_cursor


class SpectraExportViewer(QDialog):
    def __init__(self, parent, image_bgr, wavelengths, cube_path, summaries, save):
        super().__init__(parent)
        self.setWindowTitle('Extracted spectra — ROI review')
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)
        self.resize(1200, 760)
        self.image = image_bgr[..., ::-1].copy()
        self.wavelengths = np.asarray(wavelengths).copy()
        self.cube_path = str(cube_path)
        self.summaries = summaries
        layout = QVBoxLayout(self)
        cube_label = QLabel('Cube: '+self.cube_path)
        cube_label.setWordWrap(True)
        layout.addWidget(cube_label)
        row = QHBoxLayout()
        self.choice = QComboBox()
        self.choice.addItem('All ROIs', -1)
        for index, r in enumerate(summaries):
            self.choice.addItem(f"{r['id']} | {r['shape']} | {r['pixel_count']} pixels | bounds {r['bounds_xyxy']}", index)
        row.addWidget(self.choice, 1)
        export = QPushButton('Export CSV / NPZ…')
        export.clicked.connect(lambda: save(self))
        row.addWidget(export)
        close = QPushButton('Close')
        close.clicked.connect(self.accept)
        row.addWidget(close)
        layout.addLayout(row)
        self.figure = Figure(constrained_layout=True)
        self.figure.suptitle('Cube: '+Path(self.cube_path).name)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.image_ax, self.spectrum_ax = self.figure.subplots(1, 2)
        layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        layout.addWidget(self.canvas, 1)
        self.status = QLabel('Select a ROI in the list or click its outline. Spectra: mean ± 1 population standard deviation; original cube values.')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.choice.currentIndexChanged.connect(self.draw)
        self.canvas.mpl_connect('pick_event', self.pick_roi)
        self.canvas.mpl_connect('resize_event', self.draw)
        self.draw()

    def pick_roi(self, event):
        index = getattr(event.artist, '_roi_index', None)
        if index is not None:
            self.choice.setCurrentIndex(index+1)

    def draw(self, *_):
        self.image_ax.clear()
        self.spectrum_ax.clear()
        self.image_ax.imshow(self.image)
        chosen = self.choice.currentData()
        roi_patches = []
        for index, r in enumerate(self.summaries):
            color = f'C{index % 10}'
            active = chosen == -1 or chosen == index
            if r['shape'] == 'mask':
                self._draw_mask(self.image_ax, r, color, .35 if active else .1)
            points = np.asarray(r['vertices_xy'])
            if r['shape'] == 'polygon':
                patch = Polygon(points, closed=True, fill=False)
            else:
                x0, y0, x1, y1 = r['bounds_xyxy']
                patch = Rectangle((x0-.5, y0-.5), x1-x0+1, y1-y0+1, fill=False)
            patch.set_edgecolor(color)
            patch.set_linewidth(2.5 if active else 1)
            patch.set_alpha(1 if active else .3)
            patch.set_picker(5)
            patch._roi_index = index
            self.image_ax.add_patch(patch)
            roi_patches.append(patch)
            if active:
                self.spectrum_ax.plot(self.wavelengths, r['mean'], color=color, label=r['id'])
                self.spectrum_ax.fill_between(self.wavelengths, r['mean']-r['std'], r['mean']+r['std'], color=color, alpha=.15)
        self.image_ax.set_title('ROI positions · coordinates start at 0')
        self.image_ax.set_xlabel('x (pixel)')
        self.image_ax.set_ylabel('y (pixel)')
        self.spectrum_ax.set_title('ROI mean spectra ± standard deviation')
        self.spectrum_ax.set_xlabel('Wavelength (nm)')
        self.spectrum_ax.set_ylabel('Original cube value')
        self.spectrum_ax.grid(alpha=.25)
        self.spectrum_ax.legend()
        attach_spectrum_cursor(self.canvas, self.spectrum_ax)
        self.canvas.draw()
        add_roi_labels(self.image_ax, self.canvas, self.summaries, roi_patches)
        self.canvas.draw_idle()

    def save_annotation(self, image_path):
        """Render a standalone figure so all image borders and axes are retained."""
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        height, width = self.image.shape[:2]
        figure = Figure(figsize=(8*max(1, width/height), 8*max(1, height/width)),
                        dpi=150, constrained_layout=True)
        canvas = FigureCanvasAgg(figure)
        axes = figure.subplots()
        axes.imshow(self.image)
        patches = []
        for index, roi in enumerate(self.summaries):
            if roi['shape'] == 'mask':
                self._draw_mask(axes, roi, f'C{index % 10}', .35)
            if roi['shape'] == 'polygon':
                patch = Polygon(roi['vertices_xy'], closed=True, fill=False)
            else:
                x0, y0, x1, y1 = roi['bounds_xyxy']
                patch = Rectangle((x0-.5, y0-.5), x1-x0+1, y1-y0+1, fill=False)
            patch.set_edgecolor(f'C{index % 10}')
            patch.set_linewidth(1.5)
            axes.add_patch(patch)
            patches.append(patch)
        axes.set_title('ROI positions · '+Path(self.cube_path).name)
        axes.set_xlabel('x (pixel)')
        axes.set_ylabel('y (pixel)')
        canvas.draw()
        add_roi_labels(axes, canvas, self.summaries, patches)
        destination = Path(image_path)
        temporary = destination.with_suffix('.png.tmp')
        try:
            figure.savefig(temporary, format='png', dpi=150, bbox_inches='tight', pad_inches=.2)
            temporary.replace(destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        figure.clear()

    def _draw_mask(self, axes, roi, color, alpha):
        overlay = np.zeros((*self.image.shape[:2], 4), dtype=np.float32)
        xy = roi['coordinates_xy']
        overlay[xy[:, 1], xy[:, 0]] = to_rgba(color, alpha)
        axes.imshow(overlay)


def add_roi_labels(axes, canvas, summaries, patches):
    from gui.roi_labels import place_roi_labels, nearest_outline_point
    renderer = canvas.get_renderer()
    bounds = [QRectF(p.get_window_extent().x0, p.get_window_extent().y0,
                     p.get_window_extent().width, p.get_window_extent().height) for p in patches]
    sizes = []
    for r in summaries:
        width, height, _ = renderer.get_text_width_height_descent(r['id'], FontProperties(size=9), False)
        sizes.append((width+10, max(height, 12)+8))
    area = axes.get_window_extent()
    placements = place_roi_labels(bounds, sizes, QRectF(area.x0, area.y0, area.width, area.height).adjusted(3, 3, -3, -3))
    for index, (r, placement) in enumerate(zip(summaries, placements)):
        if placement is None:
            continue
        rect, start, _ = placement
        if r['shape'] == 'polygon':
            start = nearest_outline_point(axes.transData.transform(r['vertices_xy']).tolist(), rect.center())
        anchor = axes.transData.inverted().transform((start.x(), start.y()))
        position = axes.transData.inverted().transform((rect.center().x(), rect.center().y()))
        annotation = axes.annotate(
            r['id'], anchor, xytext=position, color='white', fontsize=9, ha='center', va='center',
            bbox=dict(facecolor=f'C{index % 10}', alpha=.85, edgecolor='none', pad=2),
            arrowprops=dict(arrowstyle='-', color=f'C{index % 10}', linewidth=.8), picker=True)
        annotation._roi_index = index
