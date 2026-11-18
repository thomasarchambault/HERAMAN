"""ROI identities and portable extraction exports, retaining overlapping regions."""
import csv
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

MAX_METADATA_CHARS = 16 * 1024 * 1024


def export_paths(path):
    """Companion names used by legacy exports; new exports only write one CSV."""
    path = Path(path)
    stem = path.stem.removesuffix('_pixel_spectra')
    return {key: path.with_name(stem+suffix) for key, suffix in (
        ('index', '_ROI_positions.csv'),
        ('means', '_ROI_mean_spectra.csv'),
        ('geometry', '_ROI_geometry.json'),
        ('image', '_image_with_ROIs.png'))}


def roi_summaries(cube, rois):
    result = []
    for roi in rois:
        xy = roi['coordinates_xy']
        # Bound temporary allocations independently of ROI size.
        total = np.zeros(cube.shape[2], np.float64)
        squares = total.copy()
        counts = total.copy()
        for start in range(0, len(xy), 8192):
            part = xy[start:start+8192]
            values = np.asarray(cube[part[:, 1], part[:, 0], :], dtype=np.float64)
            valid = np.isfinite(values)
            values = np.where(valid, values, 0)
            total += values.sum(axis=0)
            squares += (values*values).sum(axis=0)
            counts += valid.sum(axis=0)
        mean = np.divide(total, counts, out=np.full_like(total, np.nan), where=counts > 0)
        variance = np.divide(squares, counts, out=np.full_like(total, np.nan), where=counts > 0)-mean**2
        result.append(dict(roi, pixel_count=len(xy),
                           bounds_xyxy=[*xy.min(axis=0).tolist(), *xy.max(axis=0).tolist()],
                           centroid_xy=xy.mean(axis=0).tolist(),
                           mean=mean, std=np.sqrt(np.maximum(variance, 0))))
    return result


def write_roi_means(path, wavelengths, cube_path, summaries):
    """Export one mean spectrum per ROI, without individual pixel records."""
    path = Path(path)
    if path.suffix.lower() != '.csv':
        raise ValueError('ROI mean spectra can only be exported as CSV.')
    if not summaries:
        raise ValueError('There are no ROI spectra to export.')
    if any(np.asarray(roi['mean']).shape != (len(wavelengths),) for roi in summaries):
        raise ValueError('ROI mean spectra do not match the wavelength axis.')
    temporary = None
    try:
        with NamedTemporaryFile(mode='w', newline='', encoding='utf-8',
                                dir=path.parent, suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.writer(stream)
            writer.writerow(['cube_name', 'roi_id', 'pixel_count', *[f'{wl:.9g} nm' for wl in wavelengths]])
            for roi in summaries:
                writer.writerow([Path(cube_path).name, roi['id'], roi['pixel_count'], *roi['mean']])
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_selection(path, cube, wavelengths, cube_path, coordinates, rois):
    """Write one reloadable CSV, with geometry embedded in the first pixel row."""
    path = Path(path)
    if path.suffix.lower() != '.csv':
        raise ValueError('ROI selections can only be exported as CSV.')
    if not len(coordinates) or not rois:
        raise ValueError('Select at least one ROI before exporting.')
    membership = np.full(len(coordinates), '', dtype=object)
    lookup = np.full(cube.shape[:2], -1, dtype=np.int64)
    lookup[coordinates[:, 1], coordinates[:, 0]] = np.arange(len(coordinates))
    for roi in rois:
        xy = roi['coordinates_xy']
        rows = lookup[xy[:, 1], xy[:, 0]]
        if np.any(rows < 0):
            raise ValueError('ROI pixels are missing from the selection.')
        membership[rows] = np.where(membership[rows] == '', roi['id'], membership[rows]+';'+roi['id'])
    if np.any(membership == ''):
        raise ValueError('Selected pixels must belong to an ROI.')
    try:
        reference = Path(os.path.relpath(Path(cube_path).resolve(), path.parent.resolve())).as_posix()
    except ValueError:
        # Across drives, retain the filename rather than exposing an absolute personal path.
        reference = Path(cube_path).name
    metadata = json.dumps(dict(format='HERAMAN ROI selection', version=2,
                               cube_path=reference, cube_shape=list(cube.shape),
                               coordinate_origin='zero-based',
                               rois=[dict(id=r['id'], shape=r['shape'], vertices_xy=r['vertices_xy'],
                                          **({'radius_px': r['radius_px']} if r['shape'] == 'circle' else {}))
                                     for r in rois]), separators=(',', ':'))
    if len(metadata) > MAX_METADATA_CHARS:
        raise ValueError('Too much ROI geometry for one CSV. Export fewer ROIs at a time.')
    temporary = None
    try:
        with NamedTemporaryFile(mode='w', newline='', encoding='utf-8',
                                dir=path.parent, suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.writer(stream)
            writer.writerow(['cube_name', 'x', 'y', *[f'{wl:.9g} nm' for wl in wavelengths],
                             'selection_metadata', 'roi_ids'])
            for start in range(0, len(coordinates), 8192):
                xy = coordinates[start:start+8192]
                spectra = cube[xy[:, 1], xy[:, 0], :]
                for index, ((x, y), spectrum, names) in enumerate(zip(xy, spectra, membership[start:start+8192]), start):
                    writer.writerow([Path(cube_path).name if index == 0 else '', int(x), int(y),
                                     *spectrum, metadata if index == 0 else '', names])
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
