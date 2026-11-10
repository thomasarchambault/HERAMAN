"""Image-space helpers for interactive ROI drawing; never modify source images."""
import cv2
import numpy as np


def wand_lab_image(image_bgr):
    """Smoothed CIE Lab display colours, independent of annotation overlays."""
    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("The wand requires a uint8 BGR reference image.")
    smooth = cv2.GaussianBlur(image, (3, 3), .7)
    return cv2.cvtColor(smooth.astype(np.float32) / 255., cv2.COLOR_BGR2LAB)


def wand_reference(lab, point):
    x, y = point
    return np.median(lab[max(0, y-1):y+2, max(0, x-1):x+2], axis=(0, 1))


def wand_region(lab, point, reference, radius, tolerance, adaptive=False):
    """Return (x0, y0, local mask), bounded by a disk with connectivity 8.

    Fixed mode uses the original seed colour. Adaptive mode follows local seed
    colours with reduced lightness weight and an original-chroma guard. The
    guard reduces colour drift; it is not a biological tissue classifier.
    """
    height, width = lab.shape[:2]
    x, y = map(int, point)
    if radius < 1 or tolerance < 0 or not (0 <= x < width and 0 <= y < height):
        raise ValueError("Invalid wand radius, tolerance or image coordinate.")
    x0, x1 = max(0, x-radius), min(width, x+radius+1)
    y0, y1 = max(0, y-radius), min(height, y+radius+1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    patch = lab[y0:y1, x0:x1]
    delta = patch-np.asarray(reference)
    if adaptive:
        # Local colour can follow shading, while the stroke's original chroma
        # prevents successive samples from drifting into a different colour.
        local_reference = wand_reference(lab, point)
        local_delta = patch-local_reference
        # On monochrome references brightness is the only useful information.
        colour = np.hypot(reference[1], reference[2]) > 4
        weights = np.array([.35 if colour else 1., 1., 1.])
        distance = np.sum((local_delta*weights)**2, axis=2)
        guard = np.sum(delta[..., 1:]**2, axis=2) <= tolerance**2 if colour else np.sum(delta**2, axis=2) <= (2*tolerance)**2
    else:
        distance = np.sum(delta**2, axis=2)
        guard = True
    candidate = ((distance <= tolerance**2) & guard & ((xx-x)**2+(yy-y)**2 <= radius**2)).astype(np.uint8)
    seed = (x-x0, y-y0)
    if candidate[seed[1], seed[0]]:
        cv2.floodFill(candidate, None, seed, 2, loDiff=0, upDiff=0, flags=8 | cv2.FLOODFILL_FIXED_RANGE)
    return x0, y0, candidate == 2


def paint_stroke(mask, previous, current, radius):
    """Interpolate a disk brush between mouse events so fast strokes have no gaps."""
    cv2.line(mask, previous, current, 1, 2*radius+1)
    cv2.circle(mask, current, radius, 1, -1)


def erase_roi_masks(rois, eraser_mask):
    """Copy-on-write subtraction, preserving other ROIs and dropping empty ones."""
    yy, xx = np.nonzero(eraser_mask)
    if not len(xx):
        return list(rois), False
    region = np.s_[yy.min():yy.max()+1, xx.min():xx.max()+1]
    local = np.asarray(eraser_mask[region], dtype=bool)
    result, changed = [], False
    for roi in rois:
        if np.any(roi[region] & local):
            updated = roi.copy()
            updated[region] &= ~local
            changed = True
            if updated.any():
                result.append(updated)
        else:
            result.append(roi)
    return result, changed


def exclusion_zone(class_mask, margin=3):
    """Class mask plus an outward elliptical margin, in native cube pixels."""
    mask = np.asarray(class_mask, dtype=bool)
    radius = int(margin)
    if mask.ndim != 2 or radius < 0:
        raise ValueError("Expected a 2D class mask and a non-negative margin.")
    if radius == 0:
        return mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


fat_exclusion_zone = exclusion_zone  # Compatibility with earlier callers.


def saved_fat_mask(dataset, cube_id, shape):
    """Recover saved fat pixels of this cube only; never infer missing geometry.

    Older/subsampled datasets can only supply the coordinates actually stored.
    Full masks from the current selection are combined separately by the caller.
    """
    result = np.zeros(shape, bool)
    for name, mask in saved_class_masks(dataset, cube_id, shape).items():
        if name.casefold() == "fat":
            result |= mask
    return result


def saved_class_masks(dataset, cube_id, shape):
    """Recover each class's stored coordinates for this acquisition only."""
    from shared.dataset_provenance import cube_ids_from_dataset
    if not all(key in dataset.files for key in ("y", "rows", "cols", "sample_cube_index")):
        return {}
    ids = cube_ids_from_dataset(dataset)
    matches = np.flatnonzero(ids == cube_id)
    if not len(matches):
        return {}
    labels = np.asarray(dataset["y"]).reshape(-1)
    rows = np.asarray(dataset["rows"]).reshape(-1)
    cols = np.asarray(dataset["cols"]).reshape(-1)
    indices = np.asarray(dataset["sample_cube_index"]).reshape(-1)
    if not (labels.size == rows.size == cols.size == indices.size):
        raise ValueError("Dataset labels and spatial metadata have different lengths.")
    valid = np.isin(indices, matches) & (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
    names = np.array([str(label).strip() for label in labels])
    result = {}
    for name in np.unique(names[valid]):
        mask = np.zeros(shape, bool)
        keep = valid & (names == name)
        mask[rows[keep].astype(int), cols[keep].astype(int)] = True
        result[str(name)] = mask
    return result
