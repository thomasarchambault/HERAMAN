"""Cube import and pixel exports using the existing postprocessing viewer."""
from pathlib import Path
import cv2
import numpy as np
from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSlider, QFileDialog, QMessageBox, QComboBox, QSizePolicy, QInputDialog
from shared.envi import load_spectral_cube
from shared.cube_preview import make_rgb_preview, stretch_display_band
from shared.envi_catalog import register_envi_source
from shared.paths import ENVI_FILES_DIR


class DataMixin:
    def _register_cube_source(self, path):
        return register_envi_source(path)

    def _make_cube_controls(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._data_circle_radius = 10
        self._data_circle_center = None
        self._data_cube = None
        self._data_pixels = []
        self._data_rois = []
        self._data_selection_mask = None
        self._data_draft = []
        self._data_drag_end = None
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
        self.data_selection_mode = QComboBox()
        for label, mode in (('Pixel', 'pixel'), ('Rectangle', 'rectangle'), ('Polygon', 'polygon'), ('Circle', 'circle')):
            self.data_selection_mode.addItem(label, mode)
        self.data_selection_mode.currentIndexChanged.connect(self._change_data_selection_mode)
        row.addWidget(QLabel('Extract:'))
        row.addWidget(self.data_selection_mode)
        layout.addLayout(row)
        row = QHBoxLayout()
        self.data_selection_info = QLabel('Click to select a pixel for export.')
        self.data_selection_info.setWordWrap(True)
        row.addWidget(self.data_selection_info, 1)
        export = QPushButton('Review / export ROI spectra…')
        export.clicked.connect(self._export_data_pixels)
        load = QPushButton('Load ROIs…')
        load.setToolTip('Choose the cube folder to find its exported ROIs and open their source cube.')
        load.clicked.connect(self._import_data_rois)
        clear = QPushButton('Clear selection')
        clear.clicked.connect(self._clear_data_pixels)
        row.addWidget(load)
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
            self._data_circle_center = None
            self._data_pixels = []
            self._data_rois = []
            self._data_selection_mask = None
            self._data_draft = []
            self._data_drag_end = None
        self._data_cube = cube
        self._data_circle_radius = max(1, min(self._data_circle_radius, max(cube.shape[:2])))
        self._data_wavelengths = np.asarray(wavelengths)
        self._data_path = path
        self._refresh_pp_cube_choices(selected_path=path)
        self._update_data_selection_info()
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
            self._register_data_roi('pixel', [(x, y)], np.array([[x, y]], dtype=np.int64))
        self._ensure_data_selection_mask()[y, x] = True
        self._update_data_selection_info()

    def _ensure_data_selection_mask(self):
        shape = self._data_cube.shape[:2]
        if self._data_selection_mask is None or self._data_selection_mask.shape != shape:
            self._data_selection_mask = np.zeros(shape, dtype=bool)
            if self._data_pixels:
                xy = np.asarray(self._data_pixels)
                self._data_selection_mask[xy[:, 1], xy[:, 0]] = True
        return self._data_selection_mask

    def _update_data_selection_info(self):
        mode = self.data_selection_mode.currentData()
        hint = {'pixel': 'Click to select a pixel.',
                'rectangle': 'Drag with the left button to select a rectangle.',
                'polygon': 'Left-click vertices; right-click to close. Esc cancels.',
                'circle': f'Left-click to place a circle. Hold right button + wheel to resize (radius {self._data_circle_radius} px).'}[mode]
        self.data_selection_info.setText(f'{len(self._data_rois)} ROIs · {len(self._data_pixels)} unique pixels — {hint}')

    def _change_data_selection_mode(self, *_):
        self._data_circle_center = None
        self._data_draft = []
        self._data_drag_end = None
        self._update_data_selection_info()
        if hasattr(self, '_render_spectral_image'):
            self._render_spectral_image()

    def _add_data_region(self, points, rectangle=False):
        if self._data_cube is None:
            return
        selected = self._ensure_data_selection_mask()
        region = np.zeros(selected.shape, dtype=np.uint8)
        if rectangle:
            cv2.rectangle(region, points[0], points[1], 1, -1)
        else:
            cv2.fillPoly(region, [np.asarray(points, dtype=np.int32)], 1)
        roi_y, roi_x = np.nonzero(region)
        if len(roi_x):
            self._register_data_roi('rectangle' if rectangle else 'polygon', points,
                                    np.column_stack((roi_x, roi_y)))
        yy, xx = np.nonzero((region != 0) & ~selected)
        self._data_pixels.extend(zip(xx.tolist(), yy.tolist()))
        selected[yy, xx] = True
        self._data_draft = []
        self._data_drag_end = None
        self._update_data_selection_info()
        self._render_spectral_image()

    def _add_data_circle(self, center):
        region = np.zeros(self._data_cube.shape[:2], np.uint8)
        cv2.circle(region, center, self._data_circle_radius, 1, -1)
        yy, xx = np.nonzero(region)
        self._register_data_roi('circle', [center], np.column_stack((xx, yy)))
        self._data_rois[-1]['radius_px'] = self._data_circle_radius
        selected = self._ensure_data_selection_mask()
        yy, xx = np.nonzero((region != 0) & ~selected)
        self._data_pixels.extend(zip(xx.tolist(), yy.tolist()))
        selected[yy, xx] = True
        self._update_data_selection_info()
        self._render_spectral_image()

    def _register_data_roi(self, shape, vertices, coordinates):
        existing = {roi['id'] for roi in self._data_rois}
        number = len(existing) + 1
        while f'ROI{number}' in existing:
            number += 1
        self._data_rois.append(dict(id=f'ROI{number}', shape=shape,
                                   vertices_xy=np.asarray(vertices).tolist(), coordinates_xy=coordinates))

    def _finish_data_polygon(self):
        if len(self._data_draft) >= 3:
            points = self._data_draft
            if abs(cv2.contourArea(np.asarray(points, dtype=np.int32))) > 0:
                self._add_data_region(points)

    def _handle_data_selection_event(self, event, xy):
        """Right-click closes a pending polygon; middle drag remains available."""
        if self._data_cube is None:
            return False
        mode = self.data_selection_mode.currentData()
        kind = event.type()
        if kind == QEvent.KeyPress and event.key() == Qt.Key_Escape:
            self._change_data_selection_mode()
            return True
        if mode == 'circle':
            if kind == QEvent.Leave:
                self._data_circle_center = None
                self._render_spectral_image()
            if kind == QEvent.Wheel and event.buttons() & Qt.RightButton:
                delta = event.angleDelta().y() / 120.
                self._data_circle_radius = max(1, min(max(self._data_cube.shape[:2]),
                    int(round(self._data_circle_radius * (1.15 ** delta))) + (1 if delta > 0 else -1 if delta < 0 else 0)))
                self._data_circle_center = xy
                self._update_data_selection_info()
                self._render_spectral_image()
                return True
            if kind in (QEvent.MouseButtonPress, QEvent.MouseButtonRelease) and event.button() == Qt.RightButton:
                self._data_circle_center = xy
                self._render_spectral_image()
                return True
            if kind == QEvent.MouseMove and not event.buttons() & Qt.MiddleButton:
                self._data_circle_center = xy
                self._render_spectral_image()
                return False
            if kind == QEvent.MouseButtonPress and event.button() == Qt.LeftButton and xy is not None:
                self._data_circle_center = xy
                self._add_data_circle(xy)
                return True
            if kind == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                return True
            return False
        if mode == 'pixel':
            return False
        if (mode == 'polygon' and self._data_draft and kind == QEvent.MouseButtonPress
                and event.button() == Qt.RightButton):
            self._finish_data_polygon()
            return True
        if kind == QEvent.MouseMove and event.buttons() & (Qt.RightButton | Qt.MiddleButton):
            return False
        if kind == QEvent.MouseButtonPress and event.button() == Qt.LeftButton and xy is not None:
            if mode == 'rectangle':
                self._data_draft = [xy]
                self._data_drag_end = xy
            elif not self._data_draft or self._data_draft[-1] != xy:
                self._data_draft.append(xy)
            self._update_data_selection_info()
            self._render_spectral_image()
            return True
        if kind == QEvent.MouseMove and self._data_draft:
            if xy is not None:
                self._data_drag_end = xy
                self._render_spectral_image()
            return True
        if kind == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
            if mode == 'rectangle' and self._data_draft:
                self._add_data_region([self._data_draft[0], xy or self._data_drag_end], rectangle=True)
            return True
        return False

    def _draw_data_selection(self, image):
        from gui.roi_labels import roi_outline_paths
        self.ab_view_label.roi_annotations = []
        if self._data_cube is None:
            return image
        mask = self._data_selection_mask
        if (mask is not None and mask.any()) or self._data_draft or self._data_circle_center is not None:
            image = image.copy()
        if mask is not None and mask.shape == image.shape[:2] and mask.any():
            image[mask] = (image[mask] * .6 + np.array([0, 220, 255]) * .4).astype(np.uint8)
        for roi in self._data_rois:
            points = np.asarray(roi['vertices_xy'], dtype=np.int32)
            if roi['shape'] == 'polygon':
                cv2.polylines(image, [points], True, (0, 220, 255), 1)
            elif roi['shape'] == 'rectangle':
                cv2.rectangle(image, tuple(points[0]), tuple(points[1]), (0, 220, 255), 1)
            elif roi['shape'] == 'circle':
                cv2.circle(image, tuple(points[0]), roi['radius_px'], (0, 220, 255), 1)
            elif roi['shape'] == 'mask':
                region = np.zeros(image.shape[:2], np.uint8)
                xy = roi['coordinates_xy']
                region[xy[:, 1], xy[:, 0]] = 1
                contours, _ = cv2.findContours(region, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image, contours, -1, (0, 220, 255), 1)
            xy = roi['coordinates_xy']
            self.ab_view_label.roi_annotations.append(
                (roi['id'], (*xy.min(axis=0), *xy.max(axis=0)),
                 roi_outline_paths(roi)))
        if self._data_circle_center is not None and self.data_selection_mode.currentData() == 'circle':
            cv2.circle(image, self._data_circle_center, self._data_circle_radius, (0, 255, 255), 1)
        if self._data_draft:
            points = self._data_draft
            if self.data_selection_mode.currentData() == 'rectangle':
                cv2.rectangle(image, points[0], self._data_drag_end or points[0], (0, 255, 255), 1)
            else:
                preview = points + ([self._data_drag_end] if self._data_drag_end is not None else [])
                cv2.polylines(image, [np.asarray(preview, np.int32)], False, (0, 255, 255), 1)
                for point in points:
                    cv2.circle(image, point, 2, (0, 255, 255), -1)
        return image

    def _clear_data_pixels(self):
        self._data_circle_center = None
        self._data_pixels = []
        self._data_rois = []
        self._data_selection_mask = None
        self._data_draft = []
        self._data_drag_end = None
        self._update_data_selection_info()
        self._render_spectral_image()

    def _write_pixel_export(self, path):
        from shared.spectra_export import write_selection
        coordinates = np.asarray(self._data_pixels, dtype=np.int64)
        return write_selection(path, self._data_cube, self._data_wavelengths, self._data_path,
                               coordinates, self._data_rois)

    def _read_data_rois(self, path):
        from shared.spectra_import import read_selection
        if self._data_cube is None:
            raise ValueError('Open the source cube before loading ROIs.')
        return read_selection(path, self._data_path, self._data_cube.shape, self._data_wavelengths)

    def _restore_data_rois(self, coordinates, rois):
        self._data_circle_center = None
        self._data_pixels = [tuple(xy) for xy in coordinates.tolist()]
        self._data_rois = rois
        self._data_selection_mask = np.zeros(self._data_cube.shape[:2], dtype=bool)
        self._data_selection_mask[coordinates[:, 1], coordinates[:, 0]] = True
        self._data_draft = []
        self._data_drag_end = None
        self._update_data_selection_info()
        self._render_spectral_image()

    def _import_data_rois(self):
        selected_folder = QFileDialog.getExistingDirectory(self, 'Choose cube folder — load ROIs', str(ENVI_FILES_DIR))
        if not selected_folder:
            return
        folder = Path(selected_folder)
        try:
            from shared.spectra_import import find_selections, selection_source, locate_source, read_selection
            from shared.envi_catalog import registered_sources
            exports = find_selections(folder)
            if not exports:
                raise ValueError('No exported ROIs found in this folder or its extracted spectra subfolder.')
            path = exports[0]
            if len(exports) > 1:
                labels = [str(item.relative_to(folder.resolve())) for item in exports]
                choice, accepted = QInputDialog.getItem(
                    self, 'Load ROIs', 'Several ROI exports found — choose a selection (newest first):', labels, 0, False)
                if not accepted:
                    return
                path = exports[labels.index(choice)]
            pixel_path, source = selection_source(path)
            known = registered_sources()
            if getattr(self, '_data_path', ''):
                known.append(self._data_path)
            adjacent = folder / Path(source.replace('\\', '/')).name
            cube_path = adjacent.resolve() if adjacent.is_file() else locate_source(pixel_path, source, known)
            if cube_path is None:
                selected, _ = QFileDialog.getOpenFileName(
                    self, f'Locate source cube: {Path(source).name}', str(pixel_path.parent), 'ENVI header (*.hdr)')
                if not selected:
                    return
                cube_path = Path(selected).resolve()
            # Validate against the associated cube before replacing the displayed cube or selection.
            cube, wavelengths = load_spectral_cube(cube_path)
            try:
                coordinates, rois = read_selection(pixel_path, cube_path, cube.shape, wavelengths)
            finally:
                mapping = getattr(cube, '_mmap', None)
                if mapping is not None:
                    mapping.close()
        except Exception as exc:
            QMessageBox.warning(self, 'Load ROIs', str(exc))
            return
        if self._data_rois and QMessageBox.question(
                self, 'Load ROIs', 'Replace the current selection with the loaded ROIs?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        if str(cube_path) != getattr(self, '_data_path', None):
            try:
                self._load_data_cube(cube_path)
            except Exception as exc:
                QMessageBox.warning(self, 'Load ROIs', str(exc))
                return
        self._restore_data_rois(coordinates, rois)

    def _export_data_pixels(self):
        if not self._data_pixels:
            QMessageBox.information(self, 'Export', 'Select at least one pixel, rectangle or polygon first.')
            return
        from shared.spectra_export import roi_summaries
        from gui.spectra_export_viewer import SpectraExportViewer
        viewer = SpectraExportViewer(self, self._ab_base_bgr, self._data_wavelengths,
                                     self._data_path, roi_summaries(self._data_cube, self._data_rois),
                                     self._save_data_pixels)
        viewer.exec_()
        viewer.deleteLater()

    def _save_data_pixels(self, viewer=None):
        source = Path(self._data_path)
        export_dir = source.parent / 'extracted spectra'
        try:
            export_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            QMessageBox.warning(self, 'Export', str(exc))
            return
        suggested = export_dir / f'{source.stem}_pixel_spectra'
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export ROI selection', str(suggested), 'CSV (*.csv)')
        if not path:
            return
        if not Path(path).suffix:
            path += '.csv'
        try:
            self._write_pixel_export(path)
            if viewer is not None:
                viewer.status.setText(f'Exported: {path}\nOne CSV contains pixel spectra and complete ROI geometry. Use Load ROIs to restore it.')
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
