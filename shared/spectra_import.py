"""Restore exported ROI membership without trusting saved spectral values."""
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from shared.spectra_export import export_paths, MAX_METADATA_CHARS

# Geometry is stored once in a CSV cell; large polygon selections can exceed
# the csv module's default 128 KiB field limit.
csv.field_size_limit(MAX_METADATA_CHARS)


def _selection_metadata(path, first_row):
    if 'selection_metadata' in first_row:
        metadata = json.loads(first_row['selection_metadata'])
        if (not isinstance(metadata, dict) or metadata.get('format') != 'HERAMAN ROI selection'
                or metadata.get('version') != 2):
            raise ValueError('Unsupported ROI selection format or version.')
        return metadata
    # Legacy exports stored geometry in a companion JSON file.
    geometry_path = export_paths(path)['geometry']
    return json.loads(geometry_path.read_text(encoding='utf-8')) if geometry_path.is_file() else {}


def find_selections(folder):
    """Find pixel exports and index targets, without counting companions twice."""
    folder = Path(folder)
    found = set()
    for directory in (folder, folder / 'extracted spectra'):
        for path in directory.glob('*.csv'):
            try:
                with path.open(newline='', encoding='utf-8-sig') as stream:
                    fields = next(csv.reader(stream), [])
            except (OSError, UnicodeError, csv.Error):
                continue
            if {'cube_name', 'x', 'y', 'roi_ids'}.issubset(fields):
                found.add(path.resolve())
            elif {'pixel_spectra_file', 'roi_id'}.issubset(fields):
                try:
                    pixel, _ = selection_source(path)
                except (OSError, ValueError, csv.Error):
                    continue
                found.add(pixel.resolve())
    return sorted(found, key=lambda path: (-path.stat().st_mtime, str(path)))


def selection_source(path):
    """Find the pixel file and source path/name recorded in an export."""
    path = Path(path)
    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if 'pixel_spectra_file' in fields and 'roi_id' in fields:
            names = {row['pixel_spectra_file'] for row in reader}
            if len(names) != 1:
                raise ValueError('The ROI index must reference one pixel spectra CSV.')
            name = names.pop()
            if Path(name).name != name or Path(name).suffix.lower() != '.csv':
                raise ValueError('Select the exported pixel spectra CSV for this index.')
            # Only one index level is supported.
            path = path.parent / name
            with path.open(newline='', encoding='utf-8-sig') as pixels:
                reader = csv.DictReader(pixels)
                if not {'x', 'y', 'roi_ids', 'cube_name'}.issubset(reader.fieldnames or []):
                    raise ValueError('The index does not reference a pixel spectra CSV.')
                first_row = next(reader, {})
                source = first_row.get('cube_name')
        else:
            if not {'x', 'y', 'roi_ids', 'cube_name'}.issubset(fields):
                raise ValueError('Select a pixel spectra CSV or ROI positions CSV.')
            first_row = next(reader, {})
            source = first_row.get('cube_name')
    if not source:
        raise ValueError('The CSV does not identify its source cube.')
    metadata = _selection_metadata(path, first_row)
    if metadata:
        recorded = str(metadata.get('cube_path', ''))
        if Path(recorded.replace('\\', '/')).name != Path(source.replace('\\', '/')).name:
            raise ValueError('The geometry and CSV identify different source cubes.')
        source = recorded
    return path, source


def locate_source(pixel_path, source, known_sources=()):
    """Prefer recorded/adjacent paths; avoid choosing among homonymous cubes."""
    path = Path(pixel_path)
    recorded = Path(source)
    if recorded.is_absolute() and recorded.is_file():
        return recorded.resolve()
    if not recorded.is_absolute() and (path.parent / recorded).is_file():
        return (path.parent / recorded).resolve()
    name = Path(source.replace('\\', '/')).name
    for candidate in (path.parent / name, path.parent.parent / name):
        if candidate.is_file():
            return candidate.resolve()
    matches = {Path(value).resolve() for value in known_sources
               if Path(value).name == name and Path(value).is_file()}
    return matches.pop() if len(matches) == 1 else None


def read_selection(path, cube_path, shape, wavelengths):
    """Read pixel CSV (or its positions index), validating before changing the GUI."""
    path, _ = selection_source(path)
    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if not {'cube_name', 'x', 'y', 'roi_ids'}.issubset(fields):
            raise ValueError('Select a HERAMAN pixel spectra CSV or ROI positions CSV. Mean spectra do not contain pixel coordinates.')
        if fields[:3] != ['cube_name', 'x', 'y'] or fields[-1] != 'roi_ids':
            raise ValueError('Invalid HERAMAN pixel CSV column order.')
        spectral_fields = fields[3:-1]
        if 'selection_metadata' in fields:
            if fields[-2] != 'selection_metadata':
                raise ValueError('Invalid HERAMAN metadata column position.')
            spectral_fields = fields[3:-2]
        try:
            axis = np.array([float(field.removesuffix(' nm')) for field in spectral_fields])
        except ValueError as exc:
            raise ValueError('Invalid CSV wavelength columns.') from exc
        if (len(axis) != len(wavelengths) or not np.isfinite(axis).all()
                or not np.allclose(axis, wavelengths, rtol=1e-5, atol=1e-6)):
            raise ValueError('The CSV spectral axis does not match the open cube.')
        coordinates, groups, seen = [], {}, set()
        source, first_row = None, None
        for number, row in enumerate(reader, 2):
            if first_row is None:
                first_row = row
            elif row.get('selection_metadata'):
                raise ValueError('Selection metadata must occur only in the first pixel row.')
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f'Row {number}: incorrect number of columns.')
            if row.get('cube_name'):
                source = Path(row['cube_name'].replace('\\', '/')).name
            if source != Path(cube_path).name:
                raise ValueError(f'Row {number}: CSV source cube does not match the open cube.')
            try:
                x, y = int(row['x']), int(row['y'])
            except (ValueError, TypeError) as exc:
                raise ValueError(f'Row {number}: invalid pixel coordinates.') from exc
            if not (0 <= x < shape[1] and 0 <= y < shape[0]):
                raise ValueError(f'Row {number}: pixel outside the open cube.')
            if (x, y) in seen:
                raise ValueError(f'Row {number}: duplicate pixel coordinates.')
            names = (row.get('roi_ids') or '').split(';')
            if any(not name.strip() for name in names) or len(set(names)) != len(names):
                raise ValueError(f'Row {number}: invalid ROI membership.')
            seen.add((x, y))
            coordinates.append((x, y))
            for name in names:
                groups.setdefault(name, []).append((x, y))
    if not coordinates:
        raise ValueError('The CSV contains no selected pixels.')
    geometry = {}
    metadata = _selection_metadata(path, first_row)
    if metadata:
        if metadata.get('coordinate_origin') != 'zero-based':
            raise ValueError('Unsupported ROI coordinate origin.')
        if 'cube_shape' in metadata and list(shape) != metadata['cube_shape']:
            raise ValueError('The ROI geometry dimensions do not match the open cube.')
        if Path(str(metadata.get('cube_path', '')).replace('\\', '/')).name != Path(cube_path).name:
            raise ValueError('The ROI geometry belongs to another cube.')
        geometry = {roi['id']: roi for roi in metadata['rois']}
        if len(geometry) != len(metadata['rois']):
            raise ValueError('ROI identifiers must be unique in selection metadata.')
        if set(geometry) != set(groups):
            raise ValueError('The geometry metadata and pixel rows contain different ROIs.')
    rois = []
    for name, pixels in groups.items():
        xy = np.asarray(pixels, dtype=np.int64)
        kind, vertices = 'mask', []
        if name in geometry:
            saved = geometry[name]
            kind = saved['shape']
            if kind == 'mask':
                rois.append(dict(id=name, shape=kind, vertices_xy=[], coordinates_xy=xy))
                continue
            vertices = np.asarray(saved['vertices_xy'], dtype=float)
            if (vertices.ndim != 2 or vertices.shape[1] != 2 or not np.isfinite(vertices).all()
                    or not np.equal(vertices, np.floor(vertices)).all()
                    or np.any(vertices < 0) or np.any(vertices[:, 0] >= shape[1])
                    or np.any(vertices[:, 1] >= shape[0])):
                raise ValueError(f'{name}: invalid geometry vertices.')
            vertices = vertices.astype(np.int32)
            mask = np.zeros(shape[:2], np.uint8)
            if kind == 'rectangle' and len(vertices) == 2:
                cv2.rectangle(mask, tuple(vertices[0]), tuple(vertices[1]), 1, -1)
            elif kind == 'polygon' and len(vertices) >= 3:
                cv2.fillPoly(mask, [vertices], 1)
            elif kind == 'pixel' and len(vertices) == 1:
                mask[vertices[0, 1], vertices[0, 0]] = 1
            elif kind != 'mask':
                raise ValueError(f'{name}: unsupported ROI geometry.')
            if kind != 'mask' and (int(mask.sum()) != len(xy) or not mask[xy[:, 1], xy[:, 0]].all()):
                raise ValueError(f'{name}: geometry does not match CSV pixels.')
            vertices = vertices.tolist()
        rois.append(dict(id=name, shape=kind, vertices_xy=vertices, coordinates_xy=xy))
    return np.asarray(coordinates, dtype=np.int64), rois
