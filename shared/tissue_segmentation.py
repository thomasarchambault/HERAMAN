"""Intensity-only specimen masks, matching the standalone fat-vs-all experiment."""
import cv2
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.ndimage import binary_erosion, binary_fill_holes, label


def biological_dataset_cache_path(source, cache_dir, classes):
    """Compute the subset identity without creating a folder or a dataset."""
    source = Path(source).resolve()
    info = source.stat()
    identity = {'source': str(source), 'size': info.st_size, 'mtime_ns': info.st_mtime_ns,
                'classes': sorted({str(name).strip().casefold() for name in classes}), 'version': 1}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    return Path(cache_dir)/f'biological_{key}.npz'


def cached_biological_dataset(source, cache_dir, classes):
    """Stable subset identity allows the tissue SVM to be reused across cubes."""
    subset = biological_dataset_cache_path(source, cache_dir, classes)
    folder = subset.parent
    folder.mkdir(parents=True, exist_ok=True)
    reused = subset.is_file()
    if reused:
        with np.load(subset, allow_pickle=True) as bundle:
            names = bundle['y'].astype(str)
        count, labels = len(names), sorted(np.unique(names).tolist())
    else:
        temporary = subset.with_suffix('.tmp.npz')
        count, labels = biological_dataset(source, temporary, classes)
        temporary.replace(subset)
    return subset, count, labels, reused


def segment_cube(cube, wavelengths, args):
    band = segmentation_band_index(wavelengths, args.segmentation_band_nm)
    image = np.asarray(cube[:, :, band], dtype=np.float32)
    return intensity_tissue_mask(image, np.isfinite(image), args)


def draw_dashed_boundary(image, mask, color=(255, 255, 0), dash_px=8, gap_px=5):
    """Draw equal arc-length dashes on closed contours, without modifying inputs."""
    result = np.asarray(image).copy()
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.shape != result.shape[:2]:
        mask = cv2.resize(mask, (result.shape[1], result.shape[0]), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        points = contour.reshape(-1, 2)
        if len(points) < 2:
            continue
        distance = 0.
        for index in range(len(points)):
            first, second = points[index], points[(index+1) % len(points)]
            if distance % (dash_px+gap_px) < dash_px:
                cv2.line(result, tuple(first), tuple(second), color, 1, cv2.LINE_AA)
            distance += float(np.linalg.norm(second-first))
    return result


def biological_class_mask(y, biological_classes):
    """Validate class scope identically for preview checks and actual training."""
    selected = {str(name).strip().casefold() for name in biological_classes}
    nonbiological = {"background", "petri", "black ink", "yellow ink", "shade", "shadow"}
    if not selected or selected & nonbiological:
        raise ValueError("Select biological class names only; background/Petri/inks cannot be biological.")
    labels = np.array([str(name).strip().casefold() for name in y])
    unknown = set(labels) - selected - nonbiological
    if unknown:
        raise ValueError(f"Specify whether these classes are biological: {sorted(unknown)}. Update the biological classes field.")
    keep = np.isin(labels, list(selected))
    if np.unique(labels[keep]).size < 2:
        raise ValueError("Tissue classification requires at least two annotated biological classes.")
    return keep


def biological_dataset(source_path, destination, biological_classes):
    """Write a separate training subset; never edit the source annotations."""
    if Path(source_path).resolve() == Path(destination).resolve():
        raise ValueError("The biological subset must have a different path from the original dataset.")
    with np.load(source_path, allow_pickle=True) as bundle:
        payload = {key: bundle[key] for key in bundle.files}
    keep = biological_class_mask(payload['y'], biological_classes)
    for key in ('X', 'X_smooth', 'y', 'rows', 'cols', 'row_rel', 'col_rel', 'sample_cube_index', 'roi_id'):
        if key in payload and np.asarray(payload[key]).shape[0] == len(keep):
            payload[key] = payload[key][keep]
    np.savez_compressed(destination, **payload)
    return int(keep.sum()), sorted(np.unique(payload['y']).astype(str).tolist())

def erode_tissue_mask(mask, radius_px):
    """Circular inward erosion in native cube pixels; zero disables erosion."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("Le masque doit être 2D.")
    if not isinstance(radius_px, (int, np.integer)) or isinstance(radius_px, (bool, np.bool_)) or radius_px < 0:
        raise ValueError("Le rayon d'érosion doit être un entier >= 0.")
    if radius_px == 0 or not mask.any():
        return mask.copy()
    if min(mask.shape) <= 2 * radius_px:
        return np.zeros_like(mask)
    y, x = np.ogrid[-radius_px:radius_px+1, -radius_px:radius_px+1]
    return binary_erosion(mask, structure=x*x+y*y <= radius_px*radius_px, border_value=0)


def otsu_candidates(image, valid, log_intensity):
    """Return the quantized Otsu partition and the threshold in native units."""
    values = image[valid]
    if not values.size:
        raise ValueError("No finite intensities on this band.")
    working = np.log1p(np.maximum(values, 0)) if log_intensity else values
    low, high = float(working.min()), float(working.max())
    if high <= low:
        raise ValueError("Image constante : segmentation Otsu impossible ; régler un seuil manuel.")
    scaled = np.clip((working-low)/(high-low)*65535, 0, 65535).astype(np.uint16)
    otsu, _ = cv2.threshold(scaled.reshape(-1, 1), 0, 65535, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    threshold = low + otsu/65535*(high-low)
    if log_intensity:
        threshold = float(np.expm1(threshold))
    # Use exactly the quantized partition used by Otsu. Converting its
    # threshold back to float can round just below a constant background.
    candidates = np.zeros(image.shape, dtype=bool)
    candidates[valid] = scaled > otsu
    return candidates, threshold


def intensity_tissue_mask(mean, valid, args):
    """Choose largest component not touching image edges, fill holes, then erode.

    If all components touch the image frame, fall back to the largest one.
    Filling precedes erosion so dark internal areas cannot create erosion holes.
    """
    mean = np.asarray(mean)
    valid = np.asarray(valid, dtype=bool)
    if mean.ndim != 2 or valid.shape != mean.shape or not valid.any():
        raise ValueError("Image d'intensité ou masque de validité invalide.")
    if args.segmentation_method == "threshold":
        threshold = args.segmentation_threshold
        candidates = valid & (mean > threshold)
    else:
        candidates, threshold = otsu_candidates(mean, valid, args.segmentation_log_intensity)
    components, count = label(candidates, structure=np.ones((3, 3), dtype=bool))
    if not count:
        raise ValueError(f"Aucun échantillon au-dessus du seuil {threshold:.4g}.")
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    frame_ids = np.unique(np.concatenate((components[0], components[-1], components[:, 0], components[:, -1])))
    interior_sizes = sizes.copy()
    interior_sizes[frame_ids] = 0
    fallback = not interior_sizes.any()
    chosen = int(np.argmax(sizes if fallback else interior_sizes))
    envelope = binary_fill_holes(components == chosen)
    # Do not intersect valid before erosion: an isolated NaN must not erode its neighbours.
    eroded = erode_tissue_mask(envelope, args.tissue_mask_erosion_px)
    return envelope, eroded & valid, threshold, fallback


def segmentation_band_index(cube_wavelengths, band_nm):
    if band_nm is None:
        return None
    wavelengths = np.asarray(cube_wavelengths, dtype=np.float64)
    if (wavelengths.ndim != 1 or not wavelengths.size or
            not np.isfinite(wavelengths).all() or not np.isfinite(band_nm)):
        raise ValueError("Axe spectral ou longueur d'onde de segmentation invalide.")
    # argmin selects the first band on an exact tie; no interpolation.
    return int(np.argmin(np.abs(wavelengths - band_nm)))
