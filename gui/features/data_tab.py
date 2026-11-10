"""Cube import and pixel exports using the existing postprocessing viewer."""
from pathlib import Path
import csv
import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSlider, QFileDialog, QMessageBox, QComboBox, QSizePolicy
from shared.envi import load_spectral_cube
from shared.cube_preview import make_rgb_preview, stretch_display_band
from shared.envi_catalog import register_envi_source


class DataMixin:
    def _register_cube_source(self, path):
        return register_envi_source(path)

    def _make_cube_controls(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._data_cube = None
        self._data_pixels = []
        self.data_info = QLabel('Open a spectral cube to explore the image and spectra without classification.')
        self.data_info.setWordWrap(True)
        layout.addWidget(self.data_info)
        row = QHBoxLayout()
        self.data_display = QComboBox()
        self.data_display.addItem('RGB (nearest 650 / 550 / 450 nm)', 'rgb')
        self.data_display.addItem('Single band', 'band')
        self.data_display.currentIndexChanged.connect(self._draw_data)
        row.addWidget(self.data_display)
        self.data_band = QSlider(Qt.Horizontal)
        self.data_band.setEnabled(False)
        self.data_band.valueChanged.connect(self._draw_data)
        row.addWidget(self.data_band, 1)
        self.data_band_label = QLabel('No cube loaded')
        row.addWidget(self.data_band_label)
        layout.addLayout(row)
        row = QHBoxLayout()
        self.data_selection_info = QLabel('Click to hold a spectrum and select a pixel for export.')
        self.data_selection_info.setWordWrap(True)
        row.addWidget(self.data_selection_info, 1)
        export = QPushButton('Export selected pixels…')
        export.clicked.connect(self._export_data_pixels)
        clear = QPushButton('Clear selection')
        clear.clicked.connect(self._clear_data_pixels)
        row.addWidget(export)
        row.addWidget(clear)
        layout.addLayout(row)
        return panel

    def _view_selected_cube(self):
        path = self.combo_pp_cube.currentData()
        if not path:
            self._browse_pp_cube()
            return
        try:
            self._load_data_cube(path)
        except Exception as exc:
            QMessageBox.warning(self, 'Open cube', str(exc))

    def _set_cube_source(self, cube, wavelengths, path):
        """Synchronize controls/exports to the cube actually shown by the viewer."""
        path = str(Path(path).resolve()) if path else ''
        if path != getattr(self, '_data_path', None):
            self._data_pixels = []
        self._data_cube = cube
        self._data_wavelengths = np.asarray(wavelengths)
        self._data_path = path
        self._refresh_pp_cube_choices(selected_path=path)
        self.data_selection_info.setText(f'{len(self._data_pixels)} selected pixels — click to hold and select.')
        self.data_band.blockSignals(True)
        self.data_band.setRange(0, len(wavelengths)-1)
        self.data_band.blockSignals(False)
        self.data_info.setText(f'{Path(path).name} | {cube.shape} | {wavelengths[0]:g}–{wavelengths[-1]:g} nm | original values')
        self.pp_tissue_band.setValue(float(wavelengths[np.argmin(abs(wavelengths-self.pp_tissue_band.value()))]))
        self._draw_data()

    def _load_data_cube(self, path):
        cube, wavelengths = load_spectral_cube(path)
        self._show_unclassified_cube(cube, wavelengths, str(path))
        self._set_cube_source(cube, wavelengths, path)
        self._refresh_pp_cube_choices(selected_path=str(path))
        self.corrected_cube_path = str(path)
        self.corrected_file_input.setText(str(path))
        self.corrected_file_label.setText(Path(path).name)

    def _record_data_pixel(self, x, y):
        if self._data_cube is None:
            return
        if (x, y) not in self._data_pixels:
            self._data_pixels.append((x, y))
        self.data_selection_info.setText(f'{len(self._data_pixels)} selected pixels — last: x={x}, y={y}')

    def _clear_data_pixels(self):
        self._data_pixels = []
        self.data_selection_info.setText('0 selected pixels — click to hold and select.')
        self._render_spectral_image()

    def _write_pixel_export(self, path):
        coordinates = np.asarray(self._data_pixels, dtype=np.int64)
        spectra = np.asarray([self._data_cube[y, x, :] for x, y in coordinates])
        if Path(path).suffix.lower() == '.npz':
            np.savez(path, coordinates_xy=coordinates, spectra=spectra, wavelengths=self._data_wavelengths,
                     cube_path=self._data_path, coordinate_origin='zero-based')
        else:
            with open(path, 'w', newline='', encoding='utf-8') as stream:
                writer = csv.writer(stream)
                writer.writerow(['cube_path', 'x', 'y', *[f'{wl:g} nm' for wl in self._data_wavelengths]])
                for (x, y), values in zip(coordinates, spectra):
                    writer.writerow([self._data_path, int(x), int(y), *values])

    def _export_data_pixels(self):
        if not self._data_pixels:
            QMessageBox.information(self, 'Export', 'Click at least one pixel first.')
            return
        path, kind = QFileDialog.getSaveFileName(self, 'Export pixel spectra', '', 'CSV (*.csv);;NumPy (*.npz)')
        if not path:
            return
        if not Path(path).suffix:
            path += '.npz' if 'npz' in kind else '.csv'
        try:
            self._write_pixel_export(path)
        except Exception as exc:
            QMessageBox.warning(self, 'Export', str(exc))

    def _draw_data(self, *_):
        """Change only the display background; retain class maps and raw spectra."""
        if self._data_cube is None:
            return
        rgb = self.data_display.currentData() == 'rgb'
        self.data_band.setEnabled(not rgb)
        if rgb:
            self._ab_base_bgr, actual = make_rgb_preview(self._data_cube, self._data_wavelengths)
            caption = 'RGB ' + ' / '.join(f'{value:g}' for value in actual) + ' nm'
        else:
            index = self.data_band.value()
            channel = stretch_display_band(self._data_cube[:, :, index])
            self._ab_base_bgr = cv2.cvtColor(channel, cv2.COLOR_GRAY2BGR)
            caption = f'{self._data_wavelengths[index]:g} nm'
        self.data_band_label.setText(caption)
        self._render_spectral_image()
