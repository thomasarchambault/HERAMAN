# Rôle : Charger et échantillonner les spectres de scores affichés avec les cartes de classification.

"""Bounded spectral references for an overlay; no model-confidence inference."""
from __future__ import annotations

import numpy as np
from shared.svm_decision import file_digest
from shared.svm_calibration import fat_membership_labels
from shared.probability_colors import probability_rgb, FAT_CLASS_RGB


def spectrum_normalization_context(folder, wavelengths):
    """Use result provenance, never the currently selected GUI normalization."""
    import json
    from pathlib import Path
    folder = Path(folder)
    metadata = {}
    for name in ('classification_source.json','class_probabilities_metadata.json'):
        path = folder/name
        if path.is_file():
            try:
                metadata.update(json.loads(path.read_text(encoding='utf-8')))
            except (OSError,ValueError):
                pass
    mode = metadata.get('normalization_mode',metadata.get('normalization'))
    if mode not in ('none','minmax','snv','snv_no_ratio','snv_three_ratios','log') or wavelengths is None:
        return None
    selected = metadata.get('selected_wavelengths')
    if selected is None and (folder/'spectra_data.npz').is_file():
        with np.load(folder/'spectra_data.npz',allow_pickle=True) as data:
            if 'wavelengths' in data:
                selected = data['wavelengths']
    axis = np.asarray(wavelengths)
    selected = axis if selected is None else np.asarray(selected)
    indices = [int(np.argmin(np.abs(axis-value))) for value in selected]
    if not indices or not np.allclose(axis[indices],selected,atol=1.,rtol=0):
        return None
    return dict(mode=mode,indices=np.asarray(indices,dtype=int))


def load_platt_probabilities(path, abundance_path, names, source_shape, display_shape):
    """Validate exact pairing; never mistake one-hot maps or raw scores for P(fat)."""
    import json
    import cv2
    from pathlib import Path
    path = Path(path)
    metadata = json.loads(path.with_name('class_probabilities_metadata.json').read_text(encoding='utf-8'))
    if metadata.get('version') != 1 or metadata.get('method') != 'Platt' or metadata.get('class_names') != list(names):
        raise ValueError('Incompatible Platt metadata/classes')
    fat_membership_labels(np.empty(0), metadata['meat_max'], metadata['fat_min'])
    if metadata.get('abundance_sha256') != file_digest(abundance_path) or metadata.get('probability_sha256') != file_digest(path):
        raise ValueError('Probabilities belong to a different classification')
    names_lower = [str(n).strip().lower() for n in names]
    if 'fat' not in names_lower:
        raise ValueError('No fat class in the probability map')
    values = np.load(path, allow_pickle=False)
    if values.shape != (*source_shape, len(names)):
        raise ValueError('Probability dimensions do not match')
    finite = np.isfinite(values).all(axis=2)
    if np.any(finite & ((values < 0).any(axis=2) | (values > 1).any(axis=2))):
        raise ValueError('Invalid class probability')
    if not np.allclose(values[finite].sum(axis=1), 1, atol=1e-5):
        raise ValueError('Class probabilities do not sum to one')
    native = np.where(finite, values[..., names_lower.index('fat')], np.nan).astype(np.float32)
    display = native if tuple(source_shape) == tuple(display_shape) else cv2.resize(
        native, (display_shape[1], display_shape[0]), interpolation=cv2.INTER_NEAREST)
    return dict(native=native, display=display, metadata=metadata)


def platt_overlay(base_bgr, p_fat, mode, meat_max=.25, fat_min=.75, opacity=.45, decision_labels=None):
    """Original image outside evaluated pixels; no thresholding of the heatmap."""
    labels = fat_membership_labels(p_fat, meat_max, fat_min) if decision_labels is None else decision_labels
    valid = labels >= 0
    color = np.zeros_like(base_bgr)
    if mode == 'fat_probability':
        color = probability_rgb(p_fat)[..., ::-1]  # RGB -> BGR
    else:
        for code, bgr in enumerate(FAT_CLASS_RGB[:, ::-1]):
            color[labels == code] = bgr
    out = np.asarray(base_bgr, np.float32).copy()
    out[valid] = (1-opacity)*out[valid] + opacity*color[valid]
    return np.clip(out, 0, 255).astype(np.uint8)


def load_decision_scores(path, abundance_path, names, source_shape, display_shape):
    """Accept only score maps paired with the current abundance file."""
    import cv2
    from pathlib import Path
    abundance_path = Path(abundance_path).with_suffix('.raw')
    with np.load(path, allow_pickle=False) as saved:
        if int(saved['version']) != 1:
            raise ValueError('Unsupported score format')
        if str(saved['abundance_sha256']) != file_digest(abundance_path):
            raise ValueError('Scores belong to a different classification')
        if saved['class_names'].tolist() != list(names):
            raise ValueError('Score class names do not match')
        result = {'svm_type': str(saved['svm_type'])}
        for key in ('gap', 'favorite', 'runner'):
            value = saved[key]
            if value.shape != tuple(source_shape):
                raise ValueError('Score dimensions do not match')
            if key != 'gap':
                if not np.issubdtype(value.dtype, np.integer) or np.any((value < -1) | (value >= len(names))):
                    raise ValueError('Invalid score class index')
            elif np.any(np.isfinite(value) & (value < 0)):
                raise ValueError('Invalid negative score gap')
            if tuple(source_shape) != tuple(display_shape):
                value = cv2.resize(value.astype(np.float32),
                                   (display_shape[1], display_shape[0]), interpolation=cv2.INTER_NEAREST)
            result[key] = value.astype(np.float32 if key == 'gap' else np.int32)
        finite = np.isfinite(result['gap'])
        if np.any(finite & ((result['favorite'] < 0) | (result['runner'] < 0)
                           | (result['favorite'] == result['runner']))):
            raise ValueError('Incomplete score ranking')
    return result


def pale_ambiguity(image, mask, strength=.58):
    """Lighten marked pixels while preserving enough of their class colour."""
    out = image.astype(np.float32, copy=True)
    out[mask] = out[mask] * (1.0 - strength) + 255.0 * strength
    return np.clip(out, 0, 255).astype(np.uint8)


def ambiguity_map(gap, threshold):
    """Red: at/below threshold; green: more clearly separated; grey: unscored."""
    values = np.asarray(gap, dtype=np.float32)
    out = np.full((*values.shape, 3), (75, 75, 75), dtype=np.uint8)
    valid = np.isfinite(values)
    scale = max(float(threshold), 1e-6)
    # 0 is red, threshold is amber, twice threshold or more is green.
    ratio = np.clip(values[valid] / scale, 0.0, 2.0)
    red = np.clip(255.0 * (1.0 - ratio), 0, 255)
    green = np.where(ratio <= 1.0, 180.0 * ratio + 75.0, 255.0)
    blue = np.where(ratio <= 1.0, 45.0, 45.0 * (2.0 - ratio))
    out[valid] = np.column_stack([blue, green, red]).astype(np.uint8)
    return out


def class_references(cube, abundances, limit=10000, normalization_mode='none', band_indices=None):
    """Return unique positive winning classes and sampled native-scale spectra.

    Ties, non-finite maps and all-zero pixels remain unassigned. Statistics
    describe the current cube's predicted groups, independently of display filters.
    """
    if cube.shape[:2] != abundances.shape[:2] or abundances.shape[2] == 0:
        raise ValueError("Cube and class map must have matching spatial dimensions.")
    maps = np.asarray(abundances).reshape(-1, abundances.shape[2])
    labels = np.full(len(maps), -1, dtype=np.int32)
    for start in range(0, len(maps), 65536):
        block = maps[start:start + 65536]
        winners = np.argmax(block, axis=1)
        maximum = np.max(block, axis=1)
        valid = (np.isfinite(block).all(axis=1) & (maximum > 0)
                 & ((block == maximum[:, None]).sum(axis=1) == 1))
        labels[start:start + len(block)] = np.where(valid, winners, -1)
    spectra = np.asarray(cube).reshape(-1, cube.shape[2])
    refs = {}
    for k in range(maps.shape[1]):
        ids = np.flatnonzero(labels == k)
        total = len(ids)
        if total > limit:
            ids = ids[np.random.default_rng(42 + k).choice(total, limit, replace=False)]
        sample = np.asarray(spectra[ids], dtype=np.float64)
        sample = sample[np.isfinite(sample).all(axis=1)]
        if band_indices is not None:
            sample = sample[:,band_indices]
        if len(sample) and normalization_mode != 'none':
            from shared.spectrum_normalization import normalize_spectra
            sample = normalize_spectra(sample, normalization_mode)
        if len(sample):
            refs[k] = dict(mean=sample.mean(axis=0), std=sample.std(axis=0),
                           count=len(sample), total=total)
    return labels.reshape(cube.shape[:2]), refs
