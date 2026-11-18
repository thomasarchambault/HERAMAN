"""ROI identities and portable extraction exports, retaining overlapping regions."""
import csv
import json
from pathlib import Path

import numpy as np


def export_paths(path):
    path = Path(path)
    stem = path.stem.removesuffix('_pixel_spectra')
    return {key: path.with_name(stem+suffix) for key, suffix in (
        ('index', '_ROI_positions.csv'),
        ('means', '_ROI_mean_spectra.csv'),
        ('geometry', '_ROI_geometry.json'),
        ('image', '_image_with_ROIs.png'))}


def compact_cube_rows(rows, cube_column):
    """Name the cube only on the first row of each consecutive cube group."""
    previous = None
    for values in rows:
        row = list(values)
        source = str(row[cube_column])
        row[cube_column] = Path(source.replace('\\', '/')).name if source != previous else ''
        previous = source
        yield row


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


def write_selection(path, cube, wavelengths, cube_path, coordinates, rois):
    """Write raw pixels plus an ROI index and mean spectra for either format."""
    path = Path(path)
    summaries = roi_summaries(cube, rois)
    membership = np.full(len(coordinates), '', dtype=object)
    lookup = np.full(cube.shape[:2], -1, dtype=np.int64)
    lookup[coordinates[:, 1], coordinates[:, 0]] = np.arange(len(coordinates))
    indices, offsets = [], [0]
    for roi in summaries:
        xy = roi['coordinates_xy']
        rows = lookup[xy[:, 1], xy[:, 0]]
        indices.append(rows)
        offsets.append(offsets[-1]+len(rows))
        membership[rows] = np.where(membership[rows] == '', roi['id'], membership[rows]+';'+roi['id'])
    if path.suffix.lower() == '.npz':
        spectra = np.asarray(cube[coordinates[:, 1], coordinates[:, 0], :])
        np.savez(path, coordinates_xy=coordinates, spectra=spectra, wavelengths=wavelengths,
                 cube_path=cube_path, coordinate_origin='zero-based', roi_ids=membership.astype(str),
                 roi_names=np.array([r['id'] for r in summaries]),
                 roi_shapes=np.array([r['shape'] for r in summaries]),
                 roi_mean_spectra=np.array([r['mean'] for r in summaries]),
                 roi_std_spectra=np.array([r['std'] for r in summaries]),
                 roi_bounds_xyxy=np.array([r['bounds_xyxy'] for r in summaries]),
                 roi_centroids_xy=np.array([r['centroid_xy'] for r in summaries]),
                 roi_pixel_indices=np.concatenate(indices), roi_pixel_offsets=np.array(offsets))
    else:
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['cube_name', 'x', 'y', *[f'{wl:g} nm' for wl in wavelengths], 'roi_ids'])
            def pixel_rows():
                for start in range(0, len(coordinates), 8192):
                    xy = coordinates[start:start+8192]
                    spectra = cube[xy[:, 1], xy[:, 0], :]
                    for (x, y), spectrum, names in zip(xy, spectra, membership[start:start+8192]):
                        yield [cube_path, int(x), int(y), *spectrum, names]
            writer.writerows(compact_cube_rows(pixel_rows(), 0))
    paths = export_paths(path)
    index_path = paths['index']
    means_path = paths['means']
    with index_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['roi_id', 'shape', 'cube_name', 'pixel_count', 'x_min', 'y_min', 'x_max', 'y_max',
                         'centroid_x', 'centroid_y', 'coordinate_origin', 'pixel_spectra_file', 'roi_spectra_file'])
        writer.writerows(compact_cube_rows(
            ([r['id'], r['shape'], cube_path, r['pixel_count'], *r['bounds_xyxy'],
              *r['centroid_xy'], 'zero-based', path.name, means_path.name] for r in summaries), 2))
    with means_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['roi_id', 'cube_name', 'wavelength_nm', 'mean', 'std', 'pixel_count'])
        writer.writerows(compact_cube_rows(
            ([r['id'], cube_path, wl, mean, std, r['pixel_count']] for r in summaries
             for wl, mean, std in zip(wavelengths, r['mean'], r['std'])), 1))
    metadata = dict(cube_path=cube_path, cube_shape=list(cube.shape), coordinate_origin='zero-based',
                    overlap_policy='Pixels are unique; roi_ids lists every containing ROI.',
                    statistics='Per-band finite values; population standard deviation.',
                    rois=[{k: v for k, v in r.items() if k not in ('coordinates_xy', 'mean', 'std')} for r in summaries])
    paths['geometry'].write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return summaries
