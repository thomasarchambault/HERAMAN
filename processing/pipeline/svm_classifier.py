# Rôle : Entraîner, charger et appliquer les classificateurs SVM aux cubes hyperspectraux.

"""
svm_classifier.py

Import-safe, mix_classifier/csvm_classifier-compatible module exposing:

    unmix_from_cube(cube, lib_path, out_dir, ...)

This version uses RobustScaler + PCA99 with selectable Linear/RBF SVM classifiers.
Hyperparameters are selected with cube-grouped folds when provenance coverage permits.

It creates the GUI artifacts used by the current application:
- abundances.hdr
- mixture_side_by_side.png
- overlay_mixture.png
- mask_<class>.png
- spectra_data.npz
- endmember_names.json

Notes:
- "abundances" are not physical NNLS abundances here.
  They are SVM-derived confidence maps with shape (H, W, C).
- New models use Platt probabilities learned from cube-held-out scores.
  All classes use calibrated argmax when enabled. Probabilities are kept in class_probabilities.hdr/.raw.
  Legacy models retain raw
  SVM predictions until explicitly retrained.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List, Sequence
import hashlib
import json
import shutil
import time
import sys
import re

import numpy as np
import cv2
from shared.image_io import read_image, write_image
import joblib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from shared.dataset_provenance import cube_ids_from_dataset, sample_cube_ids
from shared.paths import portable_project_path
from shared.svm_decision import decision_gap, file_digest
from shared.svm_transforms import BandWeighter
from shared.svm_calibration import (
    CALIBRATION_VERSION, fit_platt, platt_probabilities, calibration_metrics, predict_encoded, calibration_decision_scores,
    DEFAULT_FAT_THRESHOLD, DEFAULT_MEAT_THRESHOLD, fat_selective_predictions,
)


# ============================================================
# DEFAULT PATHS
# ============================================================

DEFAULT_NPZ_DATASET = Path("roi_outputs") / "roi_dataset.npz"
DEFAULT_MODEL_PATH = Path("svm_models") / "roi_dataset_svm.joblib"


def _atomic_joblib_dump(payload: Dict[str, Any], destination: Path | str) -> None:
    """Replace a model only after a complete write, so cancellation preserves it."""
    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        joblib.dump(payload, str(temporary))
        temporary.replace(destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


# ============================================================
# SVM CONFIG
# ============================================================

# Best configuration from svm_benchmark.py (07_linear_PCA99_C1)
# RobustScaler -> PCA(99% variance) -> LinearSVC(C=1)
_REFERENCE_PIPELINE = None
SVM_TYPE = "linear"
SUPPORTED_SVM_TYPES = {"linear", "rbf"}

LINEAR_C = 1.0
RBF_C = 10.0
RBF_GAMMA = "scale"

GROUP_CV_MAX_SPLITS = 4
GROUP_CV_RANDOM_STATE = 42

MAX_ITER = 5000
RANDOM_STATE = 42
SVM_TOL = 0.001

SCALER_MODE = "robust"
USE_PCA = True
PCA_VARIANCE = 0.99
PCA_SVD_SOLVER = "full"
DEFAULT_NORMALIZATION_MODE = "none"
SUPPORTED_NORMALIZATION_MODES = {
    "none",
    "minmax",
    "mean_center",
    "snv",
    "snv_no_ratio",
    "snv_three_ratios",
    "log",
    "snv_offset",
    "snv_mp",
}


# A model may use all dataset wavelengths or an explicit profile-specific subset.
USE_SELECTED_BANDS = False
SELECTED_BAND_INDICES = None
SELECTED_WAVELENGTHS_NM = None

DEFAULT_WAVELENGTHS_NM = np.arange(420, 721, 10, dtype=np.float32)
WAVELENGTH_MATCH_TOL_NM = 1.0

# The winning PCA benchmark used no manual band weighting.
# Weight arguments remain accepted for backward compatibility but are ignored.
USE_BAND_WEIGHTS = False
BAND_WEIGHTS_BY_WAVELENGTH = {}
DEFAULT_BAND_WEIGHT = 1.0


# ============================================================
# GUI / VISUAL CONFIG
# ============================================================

OVERLAY_ALPHA = 0.75
FIG_DPI = 120

MAKE_HARD_LABELS = True
HARD_MIN_ABUNDANCE = 0.45
UNKNOWN_LABEL = 9999

PALETTE_RGB = {
    "background": (0, 0, 0),
    "fat":        (255, 204, 153),
    "muscle":     (128, 0, 0),
    "glandular":  (153, 51, 102),
    "stroma":     (255, 0, 255),
    "yellow ink": (255, 255, 0),
    "black ink":  (80, 80, 80),
    "petri":      (255, 255, 255),
    "meat":       (153, 51, 0),
}

CLASS_MIN_ABUNDANCE = {
    "background": 0.,
    "fat":        0.,
    "glandular":  0.,
    "stroma":     0.,
    "yellow ink": 0.,
    "black ink":  0.,
    "petri":      0.,
}


# ============================================================
# SMALL UTILITIES
# ============================================================

def flush_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def to_uint8(img: np.ndarray, lo: float = 1, hi: float = 99) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    m, M = np.nanpercentile(img, [lo, hi])

    if not np.isfinite(m):
        m = float(np.nanmin(img))
    if not np.isfinite(M):
        M = float(np.nanmax(img))
    if M <= m:
        M = m + 1e-6

    out = np.clip((img - m) / (M - m), 0, 1)
    return (out * 255).astype(np.uint8)


def load_library(lib_path: str | Path) -> dict:
    with open(lib_path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_palette(names: Sequence[str]) -> np.ndarray:
    """
    Build the BGR palette expected by OpenCV from RGB definitions.
    """
    import colorsys

    names = list(names)
    L = len(names)
    P = np.zeros((L, 3), np.float32)
    golden = 0.61803398875

    for i, n in enumerate(names):
        key = (n or "").lower().strip()

        if key in PALETTE_RGB:
            rgb = np.asarray(PALETTE_RGB[key], dtype=np.float32)
            P[i] = rgb[::-1]
            continue

        # Python's hash() is randomized between processes. A cryptographic
        # digest keeps fallback colors stable across application processes.
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        h0 = int.from_bytes(digest[:4], "big") / float(2**32)
        h = (h0 + i * golden) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
        P[i] = np.array([b * 255.0, g * 255.0, r * 255.0], np.float32)

    return P


def save_label_figure_with_legend(
    img_bgr: np.ndarray,
    names: List[str],
    out_path: str | Path,
    title: str = "Overlay"
) -> None:
    def bgr2rgb(bgr):
        b, g, r = bgr
        return (r / 255.0, g / 255.0, b / 255.0)

    pal = make_palette(names)
    patches = []

    for i, n in enumerate(names):
        patches.append(
            Patch(
                facecolor=bgr2rgb(tuple(pal[i])),
                edgecolor="k",
                label=n
            )
        )

    plt.figure(figsize=(10, 10), dpi=FIG_DPI)
    plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.xlabel("x (px)")
    plt.ylabel("y (px)")
    plt.legend(handles=patches, loc="lower right", title="Classes")
    plt.tight_layout()
    plt.savefig(out_path, dpi=FIG_DPI)
    plt.close()


# ============================================================
# NORMALIZATION
# ============================================================

def snv_rows(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    mu = np.nanmean(X, axis=1, keepdims=True)
    X0 = X - mu
    sd = np.nanstd(X0, axis=1, keepdims=True)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return X0 / sd


def snv_rows_mean_preserving(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    sd = np.nanstd(X, axis=1, keepdims=True)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return X / sd


def _clean_normalization_mode(mode: str | None) -> str:
    cleaned = str(mode or DEFAULT_NORMALIZATION_MODE).lower().strip().replace("`", "")
    if cleaned not in SUPPORTED_NORMALIZATION_MODES:
        raise ValueError(
            f"Unsupported normalization mode {cleaned!r}. "
            f"Expected one of: {sorted(SUPPORTED_NORMALIZATION_MODES)}"
        )
    return cleaned


def _clean_svm_type(svm_type: str | None) -> str:
    cleaned = str(svm_type or SVM_TYPE).lower().strip().replace("`", "")
    if cleaned not in SUPPORTED_SVM_TYPES:
        raise ValueError(
            f"Unsupported SVM type {cleaned!r}. "
            f"Expected one of: {sorted(SUPPORTED_SVM_TYPES)}"
        )
    return cleaned


def prepare_svm_features(X, wavelengths, mode="none", *, include_ratio=True):
    """Normalize spectral bands, then append the selected raw reflectance ratios.

    The ratio uses the corrected, unnormalized reflectances, as in the benchmark.
    Spectral plots and the library still use normalize_rows without this feature.
    """
    mode = _clean_normalization_mode(mode)
    X = np.ascontiguousarray(X, dtype=np.float32)
    if mode in {"none", "snv_no_ratio", "log"} or not include_ratio:
        return normalize_rows(X, mode)
    wavelengths = np.asarray(wavelengths, dtype=float)
    from shared.svm_transforms import svm_ratio_pairs
    ratios = []
    for numerator, denominator in svm_ratio_pairs(mode):
        indices = []
        for band in (numerator, denominator):
            matches = np.flatnonzero(np.isclose(wavelengths, band, rtol=0, atol=1e-6))
            if len(matches) != 1:
                raise ValueError(f"Normalized SVM requires the {band:g} nm band for R{numerator:g}/R{denominator:g}.")
            indices.append(int(matches[0]))
        ratios.append(np.maximum(X[:,indices[0]],np.float32(1e-6))/np.maximum(X[:,indices[1]],np.float32(1e-6)))
    return np.column_stack([normalize_rows(X,mode),*ratios])


def svm_ratio_feature(mode):
    if _clean_normalization_mode(mode) == "snv_three_ratios":
        return "raw_620_580_560_540_700_620_v1"
    return "raw_620_580_v1" if _clean_normalization_mode(mode) not in {"none", "snv_no_ratio", "log"} else None


def normalize_rows(X: np.ndarray, mode: str = DEFAULT_NORMALIZATION_MODE) -> np.ndarray:
    mode = _clean_normalization_mode(mode)
    if mode in {'none', 'minmax', 'snv', 'snv_no_ratio', 'snv_three_ratios', 'log'}:
        from shared.spectrum_normalization import normalize_spectra
        return normalize_spectra(X, mode)
    X = np.asarray(X, dtype=np.float32)

    if mode == "none":
        return X

    if mode == "minmax":
        row_min = np.nanmin(X, axis=1, keepdims=True)
        row_max = np.nanmax(X, axis=1, keepdims=True)
        span = row_max - row_min
        span = np.where(span > 1e-12, span, 1.0)
        return (X - row_min) / span

    if mode == "mean_center":
        return X - np.nanmean(X, axis=1, keepdims=True)

    if mode in {"snv", "snv_no_ratio"}:
        return snv_rows(X)

    if mode == "snv_mp":
        return snv_rows_mean_preserving(X)

    # Preserve the legacy library behavior for "SNV + offset": the SNV shape
    # receives an intensity offset derived from the relative row mean.
    mu = np.nanmean(X, axis=1, keepdims=True)
    snv = snv_rows(X)
    mean_values = mu[:, 0]
    mean_min = float(np.nanmin(mean_values))
    mean_max = float(np.nanmax(mean_values))
    mean_span = mean_max - mean_min
    if not np.isfinite(mean_span) or mean_span <= 1e-12:
        offset = np.zeros((X.shape[0], 1), dtype=np.float32)
    else:
        offset = ((mean_values - mean_min) / mean_span).astype(np.float32)[:, None] * 10.0
    return snv + offset


# ============================================================
# BAND SELECTION / WEIGHTING
# ============================================================

def get_wavelengths_for_n_bands(n_bands: int, wavelengths_from_file=None):
    if wavelengths_from_file is not None:
        wavelengths_from_file = np.asarray(wavelengths_from_file, dtype=np.float32)
        if len(wavelengths_from_file) == n_bands:
            return wavelengths_from_file

    if DEFAULT_WAVELENGTHS_NM is not None:
        default_wl = np.asarray(DEFAULT_WAVELENGTHS_NM, dtype=np.float32)
        if len(default_wl) == n_bands:
            return default_wl

    return None


def resolve_band_indices(n_bands: int, wavelengths=None):
    if not USE_SELECTED_BANDS:
        band_indices = np.arange(n_bands, dtype=int)
        selected_wavelengths = wavelengths.copy() if wavelengths is not None else None
        return band_indices, selected_wavelengths

    if SELECTED_BAND_INDICES is not None:
        band_indices = np.asarray(SELECTED_BAND_INDICES, dtype=int)

        if band_indices.ndim != 1 or len(band_indices) == 0:
            raise ValueError("SELECTED_BAND_INDICES must be a non-empty 1D list.")

        if np.any(band_indices < 0) or np.any(band_indices >= n_bands):
            raise ValueError(
                f"SELECTED_BAND_INDICES contains invalid indices for {n_bands} bands: "
                f"{band_indices.tolist()}"
            )

        selected_wavelengths = wavelengths[band_indices] if wavelengths is not None else None
        return band_indices, selected_wavelengths

    if SELECTED_WAVELENGTHS_NM is not None:
        if wavelengths is None:
            raise ValueError(
                "SELECTED_WAVELENGTHS_NM is defined, but no wavelengths axis is available."
            )

        wavelengths = np.asarray(wavelengths, dtype=np.float32)
        requested = np.asarray(SELECTED_WAVELENGTHS_NM, dtype=np.float32)

        band_indices = []
        matched_wavelengths = []

        for wl in requested:
            idx = int(np.argmin(np.abs(wavelengths - wl)))
            diff = float(abs(wavelengths[idx] - wl))

            if diff > WAVELENGTH_MATCH_TOL_NM:
                raise ValueError(
                    f"Could not match {wl} nm. Closest band: "
                    f"{float(wavelengths[idx])} nm, diff={diff:.3f} nm."
                )

            band_indices.append(idx)
            matched_wavelengths.append(float(wavelengths[idx]))

        band_indices = np.asarray(band_indices, dtype=int)
        selected_wavelengths = np.asarray(matched_wavelengths, dtype=np.float32)

        if len(np.unique(band_indices)) != len(band_indices):
            raise ValueError(
                "Some selected wavelengths point to the same band. "
                f"Obtained indices: {band_indices.tolist()}"
            )

        return band_indices, selected_wavelengths

    raise ValueError(
        "USE_SELECTED_BANDS=True, but SELECTED_BAND_INDICES and "
        "SELECTED_WAVELENGTHS_NM are both None."
    )


def build_band_weights(selected_wavelengths, n_bands: int, default_weight=1.0):
    if selected_wavelengths is None:
        return np.ones(n_bands, dtype=np.float32) * float(default_weight)

    selected_wavelengths = np.asarray(selected_wavelengths, dtype=np.float32)
    weights = np.ones(len(selected_wavelengths), dtype=np.float32) * float(default_weight)

    for i, wl in enumerate(selected_wavelengths):
        wl_int = int(round(float(wl)))
        weights[i] = float(BAND_WEIGHTS_BY_WAVELENGTH.get(wl_int, default_weight))

    return weights


def print_band_selection_summary(prefix, n_bands_original, band_indices, selected_wavelengths, band_weights=None):
    flush_print(f"{prefix} Original bands: {n_bands_original}")
    flush_print(f"{prefix} Used bands    : {len(band_indices)}")
    flush_print(f"{prefix} Indices       : {np.asarray(band_indices).tolist()}")

    if selected_wavelengths is not None:
        flush_print(f"{prefix} Wavelengths nm: {np.asarray(selected_wavelengths).tolist()}")
    else:
        flush_print(f"{prefix} Wavelengths nm: not available")

    if band_weights is not None:
        flush_print(f"{prefix} Band weights  : {np.asarray(band_weights).tolist()}")


# ============================================================
# DATASET / MODEL
# ============================================================

def _parse_wavelengths_from_folder(folder: Path, n_bands: int | None = None) -> Optional[np.ndarray]:
    """
    Infer wavelengths from files named like corr_490nm.npy/png/tif in a folder.
    The cube builder in postprocessing_pipeline stacks wavelengths in sorted numerical order,
    so sorted unique filename wavelengths reproduce the cube spectral axis.
    """
    folder = Path(folder)
    if not folder.exists() or not folder.is_dir():
        return None

    allowed_ext = {".npy", ".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}
    found: set[int] = set()

    for path in folder.iterdir():
        if not path.is_file() or path.suffix.lower() not in allowed_ext:
            continue
        m = re.search(r"(\d{3,4})\s*nm", path.name.lower())
        if m:
            found.add(int(m.group(1)))

    if not found:
        return None

    wl = np.asarray(sorted(found), dtype=np.float32)
    if n_bands is not None and len(wl) != int(n_bands):
        return None
    return wl


def _infer_wavelengths_near_path(path: str | Path, n_bands: int | None = None) -> Optional[np.ndarray]:
    """
    Try to infer wavelengths from the corrected/sweep folder around a cube path.
    Works with:
      corrected/<sweep>_corrected/cube/<name>_cube.npy
      corrected/<sweep>_corrected/corr_490nm.npy
    """
    if Path(path).suffix.lower() in ('.hdr', '.raw', '.img', '.dat', '.bin'):
        from shared.envi import load_spectral_cube
        try:
            cube, axis = load_spectral_cube(path)
            count = cube.shape[2]
            mapping = getattr(cube, '_mmap', None)
            if mapping is not None: mapping.close()
            return axis if n_bands is None or len(axis) == n_bands else None
        except (OSError, ValueError):
            return None

    p = Path(path)
    candidates: list[Path] = []

    from shared.corrected_layout import nearby_image_folders
    candidates = nearby_image_folders(p)

    for folder in candidates:
        wl = _parse_wavelengths_from_folder(folder, n_bands=n_bands)
        if wl is not None:
            return wl

    return None


def _default_wavelengths_if_valid(n_bands: int) -> Optional[np.ndarray]:
    default = np.asarray(DEFAULT_WAVELENGTHS_NM, dtype=np.float32)
    if len(default) == int(n_bands):
        return default.copy()
    return None


def _infer_cube_wavelengths(n_bands: int, out_dir: Path) -> Optional[np.ndarray]:
    """Read the spectral axis from the ENVI source recorded with classification results."""
    source = Path(out_dir)/'classification_source.json'
    if source.is_file():
        try:
            metadata = json.loads(source.read_text(encoding='utf-8'))
            from shared.paths import resolve_project_path
            return _infer_wavelengths_near_path(resolve_project_path(metadata['cube_path']) if metadata.get('path_base')=='project' else metadata['cube_path'], n_bands)
        except (OSError, ValueError, KeyError):
            pass
    return None


def _load_dataset(npz_path: Path):
    with np.load(str(npz_path), allow_pickle=True) as D:
        if ("X" not in D.files) or ("y" not in D.files):
            raise RuntimeError(f"{npz_path} must contain keys 'X' and 'y'.")

        X = np.asarray(D["X"], np.float32)
        y = np.asarray(D["y"]).astype(str)
        wavelengths = None

        if "wavelengths" in D.files:
            wavelengths = np.asarray(D["wavelengths"], np.float32)
        elif "meta" in D.files:
            try:
                meta = D["meta"].item()
                if isinstance(meta, dict) and ("wavelengths" in meta):
                    wavelengths = np.asarray(meta["wavelengths"], np.float32)
            except Exception:
                wavelengths = None

        # Backward-compatible inference for older ROI datasets.
        if wavelengths is None or len(wavelengths) != X.shape[1]:
            wavelengths = _default_wavelengths_if_valid(X.shape[1])

        if (wavelengths is None or len(wavelengths) != X.shape[1]) and "cube_paths" in D.files:
            try:
                for cp in np.asarray(D["cube_paths"], dtype=object).tolist():
                    wl = _infer_wavelengths_near_path(cp, n_bands=X.shape[1])
                    if wl is not None:
                        wavelengths = wl
                        break
            except Exception:
                pass

    return X, y, wavelengths


def _load_dataset_bundle(npz_path: Path | str) -> Dict[str, Any]:
    """Load spectra plus stable cube/ROI provenance used by grouped training."""
    npz_path = Path(npz_path)
    X, y, wavelengths = _load_dataset(npz_path)
    n_samples = int(X.shape[0])

    with np.load(str(npz_path), allow_pickle=True) as dataset:
        cube_ids = cube_ids_from_dataset(dataset)
        if "sample_cube_index" in dataset.files:
            cube_index = np.asarray(dataset["sample_cube_index"], dtype=np.int32).reshape(-1)
        else:
            cube_index = np.full(n_samples, -1, dtype=np.int32)
        if "roi_id" in dataset.files:
            roi_ids = np.asarray(dataset["roi_id"], dtype=np.int32).reshape(-1)
        else:
            roi_ids = np.full(n_samples, -1, dtype=np.int32)

    if cube_index.size != n_samples:
        cube_index = np.full(n_samples, -1, dtype=np.int32)
    if roi_ids.size != n_samples:
        roi_ids = np.full(n_samples, -1, dtype=np.int32)

    groups = sample_cube_ids(cube_index, cube_ids)
    return {
        "X": X,
        "y": np.asarray(y).astype(str),
        "wavelengths": wavelengths,
        "cube_ids": cube_ids,
        "sample_cube_index": cube_index,
        "sample_cube_id": groups,
        "roi_id": roi_ids,
    }


def _cube_class_sample_weights(y: Sequence[str], groups: Sequence[str]) -> np.ndarray:
    """Give every represented cube/class cell the same total training weight."""
    y = np.asarray(y).astype(str)
    groups = np.asarray(groups, dtype=object)
    weights = np.ones(y.size, dtype=np.float64)
    for group, label in {(str(g), str(c)) for g, c in zip(groups, y)}:
        mask = (groups == group) & (y == label)
        count = int(mask.sum())
        if count:
            weights[mask] = 1.0 / float(count)
    mean_weight = float(np.mean(weights)) if weights.size else 1.0
    if mean_weight > 0:
        weights /= mean_weight
    return weights


def _default_svm_params() -> Dict[str, Any]:
    return {
        "linear_c": float(LINEAR_C),
        "rbf_c": float(RBF_C),
        "rbf_gamma": RBF_GAMMA,
    }


def _grouped_svm_candidates(svm_type: str) -> list[Dict[str, Any]]:
    _clean_svm_type(svm_type)
    return [_default_svm_params()]


def _valid_grouped_splits(y: np.ndarray, groups: np.ndarray):
    unique_groups = np.unique(groups)
    if unique_groups.size < 3:
        return None, "at least 3 cubes are required"

    class_group_counts = {
        label: len(np.unique(groups[y == label]))
        for label in np.unique(y)
    }
    min_class_groups = min(class_group_counts.values(), default=0)
    if min_class_groups < 2:
        return None, "every class must occur in at least 2 cubes"

    n_splits = min(int(GROUP_CV_MAX_SPLITS), int(unique_groups.size), int(min_class_groups))
    if n_splits < 2:
        return None, "not enough cube coverage for grouped folds"

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=GROUP_CV_RANDOM_STATE,
    )
    splits = list(splitter.split(np.zeros(y.size), y, groups))
    expected_classes = set(np.unique(y).tolist())
    if any(set(np.unique(y[train_idx]).tolist()) != expected_classes for train_idx, _ in splits):
        return None, "a grouped training fold is missing at least one class"
    return splits, None


def _calibration_splits(y, groups):
    """Grouped OOF scores, allowing two cubes for inner/final calibration."""
    if any(str(group).startswith("unknown_cube_") for group in groups):
        raise ValueError("Platt requires known cube provenance for every annotation.")
    splits, reason = _valid_grouped_splits(y, groups)
    if splits is None and len(np.unique(groups)) == 2:
        splits = list(StratifiedGroupKFold(2, shuffle=True, random_state=GROUP_CV_RANDOM_STATE).split(y, y, groups))
        if any(set(y[train]) != set(y) for train, _ in splits):
            splits = None
    if splits is None:
        raise ValueError("Platt grouped calibration unavailable: " + str(reason)
                         + ". Each training fold must contain every class.")
    return splits


def _fit_grouped_fold(X, y, groups, wavelengths, normalization_mode, svm_type, params):
    weights = (build_band_weights(wavelengths, X.shape[1], DEFAULT_BAND_WEIGHT)
               if USE_BAND_WEIGHTS else np.ones(X.shape[1], np.float32))
    if svm_ratio_feature(normalization_mode):
        weights = np.r_[weights, np.ones(3 if normalization_mode == "snv_three_ratios" else 1,dtype=np.float32)]
    model = _make_svm_model(weights, svm_type=svm_type, **params)
    model.fit(prepare_svm_features(X, wavelengths, normalization_mode), y,
              svm__sample_weight=_cube_class_sample_weights(y, groups))
    return model


def _grouped_oof_scores(X, y, groups, wavelengths, normalization_mode, svm_type, params):
    if normalization_mode == "snv_offset":
        raise ValueError("Platt requires per-spectrum preprocessing; choose SNV or SNV without ratio instead of SNV offset.")
    n_classes = len(np.unique(y))
    scores = np.full((len(y),) if n_classes == 2 else (len(y), n_classes), np.nan)
    for number, (train, valid) in enumerate(_calibration_splits(y, groups), 1):
        flush_print(f"[SVM][PLATT] Inner fold {number}: fitting on {len(train):,} pixels, holding out {len(valid):,}")
        model = _fit_grouped_fold(X[train], y[train], groups[train], wavelengths, normalization_mode, svm_type, params)
        scores[valid] = model.decision_function(prepare_svm_features(X[valid], wavelengths, normalization_mode))
    if not np.isfinite(scores).all():
        raise ValueError("Incomplete/nonfinite grouped calibration scores.")
    return scores


def _select_grouped_svm_params(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    roi_ids: np.ndarray,
    *,
    normalization_mode: str,
    svm_type: str,
    wavelengths=None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluate fixed SVM + Platt with nested cube-grouped validation."""
    params_default = _default_svm_params()
    if _clean_svm_type(svm_type) == "rbf" and len(y) >= 100_000:
        flush_print(
            f"[SVM][GROUP-CV][WARN] RBF validation will evaluate all {len(y):,} "
            "labeled pixels in each fold and may be slow. No parameter search is performed."
        )
    splits, reason = _valid_grouped_splits(y, groups)
    if splits is None:
        return params_default, {
            "status": "fallback",
            "reason": reason,
            "n_cubes": int(len(np.unique(groups))),
            "calibration": {"status": "unavailable", "reason": reason, "method": "sigmoid"},
        }

    X = np.asarray(X, dtype=np.float32)
    candidates = _grouped_svm_candidates(svm_type)
    candidate_results = []
    best_params = params_default
    best_score = -np.inf

    for candidate_number, candidate in enumerate(candidates, start=1):
        fold_scores = []
        cv_class_names = [str(label) for label in np.unique(y)]
        encoder = LabelEncoder().fit(y)
        encoded = encoder.transform(y)
        n_classes = len(cv_class_names)
        oof_scores = np.full((len(y),) if n_classes == 2 else (len(y), n_classes), np.nan)
        oof_probabilities = np.full((len(y), n_classes), np.nan)
        raw_confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
        per_cube_reports = []
        calibration_error = None
        aggregate_confusion = np.zeros(
            (len(cv_class_names), len(cv_class_names)), dtype=np.int64
        )
        failed = None
        flush_print(
            f"[SVM][GROUP-CV] Fixed configuration: {candidate}"
        )
        for fold_number, (train_idx, valid_idx) in enumerate(splits, start=1):
            try:
                flush_print(f"[SVM][GROUP-CV] Fold {fold_number}/{len(splits)}: SVM + Platt; held-out cubes: {', '.join(map(str, np.unique(groups[valid_idx])))}")
                try:
                    inner_scores = _grouped_oof_scores(X[train_idx], encoded[train_idx], groups[train_idx],
                        wavelengths, normalization_mode, svm_type, candidate)
                    calibrator = fit_platt(inner_scores, encoded[train_idx], n_classes)
                except ValueError as exc:
                    calibrator = None
                    calibration_error = str(exc)
                    flush_print(f"[SVM][PLATT][WARN] Nested validation unavailable: {exc}")
                model = _fit_grouped_fold(X[train_idx], encoded[train_idx], groups[train_idx],
                    wavelengths, normalization_mode, svm_type, candidate)
                X_valid = prepare_svm_features(X[valid_idx], wavelengths, normalization_mode)
                oof_scores[valid_idx] = model.decision_function(X_valid)
                raw_pred = encoder.inverse_transform(model.predict(X_valid))
                raw_confusion += confusion_matrix(y[valid_idx], raw_pred, labels=cv_class_names)
                if calibrator is None:
                    continue
                oof_probabilities[valid_idx] = platt_probabilities(calibrator, oof_scores[valid_idx])
                pred = encoder.inverse_transform(np.argmax(oof_probabilities[valid_idx], axis=1))
                aggregate_confusion += confusion_matrix(
                    y[valid_idx].astype(str), pred, labels=cv_class_names
                )

                per_cube = []
                for cube_id in np.unique(groups[valid_idx]):
                    mask = groups[valid_idx] == cube_id
                    truth = y[valid_idx][mask]
                    guess = pred[mask]
                    labels_present = np.unique(truth)
                    per_cube.append(
                        f1_score(
                            truth,
                            guess,
                            labels=labels_present,
                            average="macro",
                            zero_division=0,
                        )
                    )
                    per_cube_reports.append(dict(cube_id=str(cube_id), fold=fold_number,
                        macro_f1=float(per_cube[-1]),
                        confusion_matrix=confusion_matrix(truth, guess, labels=cv_class_names).tolist(),
                        **calibration_metrics(encoded[valid_idx][mask], oof_probabilities[valid_idx][mask], cv_class_names)))
                fold_scores.append(float(np.mean(per_cube)))
                flush_print(
                    f"[SVM][GROUP-CV] Fold {fold_number}/{len(splits)} complete: "
                    f"macro-F1={fold_scores[-1]:.4f}"
                )
            except Exception as exc:
                failed = str(exc)
                break

        complete = failed is None and np.isfinite(oof_scores).all()
        calibrated = complete and np.isfinite(oof_probabilities).all()
        final_calibrator = (fit_platt(oof_scores, encoded, n_classes)
                            if complete and normalization_mode != "snv_offset" else None)
        mean_score = float(np.mean(fold_scores)) if calibrated else -np.inf
        calibration_report = dict(status="evaluated" if calibrated else "unavailable", method="sigmoid",
            version=CALIBRATION_VERSION, reason=calibration_error or failed,
            protocol="nested cube-grouped validation; outer cubes excluded from SVM and calibrator",
            roi_only=True, weighted=False, per_cube=per_cube_reports if calibrated else [])
        if calibrated:
            calibration_report.update(calibration_metrics(encoded, oof_probabilities, cv_class_names))
            calibration_report["per_cube_mean_brier"] = float(np.mean([c["brier"] for c in per_cube_reports]))
            calibration_report["per_cube_mean_log_loss"] = float(np.mean([c["log_loss"] for c in per_cube_reports]))
        candidate_results.append({
            "params": candidate,
            "fold_macro_f1": fold_scores,
            "mean_macro_f1": mean_score if np.isfinite(mean_score) else None,
            "class_names": cv_class_names,
            "confusion_matrix": aggregate_confusion.tolist() if calibrated else None,
            "error": failed,
            "raw_confusion_matrix": raw_confusion.tolist() if complete else None,
            "calibration": calibration_report,
            "final_calibrator": final_calibrator,
        })
        if mean_score > best_score:
            best_score = mean_score
            best_params = candidate

    if not np.isfinite(best_score):
        return params_default, {
            "status": "fallback",
            "reason": "nested Platt validation unavailable; see calibration diagnostics",
            "n_cubes": int(len(np.unique(groups))),
            "candidates": candidate_results,
            "calibration": candidate_results[0].get("calibration") if candidate_results else None,
            "final_calibrator": candidate_results[0].get("final_calibrator") if candidate_results else None,
        }

    best_result = max(
        (item for item in candidate_results if item["mean_macro_f1"] is not None),
        key=lambda item: item["mean_macro_f1"],
    )
    return best_params, {
        "status": "evaluated_fixed_parameters",
        "n_cubes": int(len(np.unique(groups))),
        "n_splits": int(len(splits)),
        "metric": "mean per-cube macro_f1",
        "best_score": float(best_score),
        "fixed_params": best_params,
        "class_names": best_result["class_names"],
        "confusion_matrix": best_result["confusion_matrix"],
        "prediction_rule": "Platt probability argmax",
        "raw_confusion_matrix": best_result["raw_confusion_matrix"],
        "calibration": best_result["calibration"],
        "final_calibrator": best_result["final_calibrator"],
        "candidates": candidate_results,
    }


def _persist_dataset_wavelengths(npz_path: Path, wavelengths: Sequence[float]) -> None:
    """Ajoute une fois l'axe spectral à un ancien dataset, avec sauvegarde."""
    npz_path = Path(npz_path)
    wavelengths = np.asarray(wavelengths, dtype=np.float32).reshape(-1)

    with np.load(str(npz_path), allow_pickle=True) as D:
        payload = {key: D[key] for key in D.files}

    if "X" not in payload or payload["X"].shape[1] != wavelengths.size:
        raise ValueError("Dataset incompatible avec l'axe spectral à sauvegarder.")

    existing = payload.get("wavelengths")
    if existing is not None:
        existing = np.asarray(existing, dtype=np.float32).reshape(-1)
        if existing.size == wavelengths.size and np.allclose(existing, wavelengths, atol=0.5, rtol=0.0):
            return

    backup_path = npz_path.with_name(npz_path.stem + "_backup_before_wavelengths.npz")
    if not backup_path.exists():
        shutil.copy2(npz_path, backup_path)

    payload["wavelengths"] = wavelengths
    temporary_path = npz_path.with_name(npz_path.stem + "_wavelengths_tmp.npz")
    np.savez_compressed(str(temporary_path), **payload)
    temporary_path.replace(npz_path)


def _match_wavelength_indices(
    dataset_wavelengths: Sequence[float],
    cube_wavelengths: Sequence[float],
    requested_wavelengths: Optional[Sequence[float]] = None,
    tol_nm: float = WAVELENGTH_MATCH_TOL_NM,
):
    """
    Option B: use all wavelengths common to dataset and cube.
    Returns separate indices because dataset and cube indices may differ.
    """
    ds_wl = np.asarray(dataset_wavelengths, dtype=np.float32)
    cube_wl = np.asarray(cube_wavelengths, dtype=np.float32)

    if ds_wl.ndim != 1 or cube_wl.ndim != 1:
        raise ValueError("Wavelength axes must be 1D.")

    if requested_wavelengths is None:
        requested = ds_wl
    else:
        requested = np.asarray(requested_wavelengths, dtype=np.float32)

    ds_idx: list[int] = []
    cube_idx: list[int] = []
    matched: list[float] = []

    for wl in requested:
        i_ds = int(np.argmin(np.abs(ds_wl - float(wl))))
        i_cube = int(np.argmin(np.abs(cube_wl - float(wl))))
        d_ds = float(abs(ds_wl[i_ds] - float(wl)))
        d_cube = float(abs(cube_wl[i_cube] - float(wl)))

        if d_ds <= tol_nm and d_cube <= tol_nm:
            # Avoid duplicates if requested wavelengths are redundant.
            if i_ds not in ds_idx and i_cube not in cube_idx:
                ds_idx.append(i_ds)
                cube_idx.append(i_cube)
                matched.append(float(ds_wl[i_ds]))

    if requested_wavelengths is not None and len(matched) != len(requested):
        missing = [float(wl) for wl in requested if not np.any(np.isclose(matched, wl, atol=tol_nm, rtol=0.0))]
        raise ValueError(
            "The cube does not contain every wavelength required by this SVM profile. "
            f"Missing: {missing} nm; cube wavelengths={cube_wl.tolist()}"
        )

    if not matched:
        raise ValueError(
            "No common wavelengths found between dataset and cube.\n"
            f"dataset wavelengths={ds_wl.tolist()}\n"
            f"cube wavelengths={cube_wl.tolist()}"
        )

    return (
        np.asarray(ds_idx, dtype=int),
        np.asarray(cube_idx, dtype=int),
        np.asarray(matched, dtype=np.float32),
    )


def _make_wavelength_model_path(
    model_path: Path,
    matched_wavelengths: Sequence[float],
    normalization_mode: str,
    svm_type: str = SVM_TYPE,
    svm_params: Optional[Dict[str, Any]] = None,
) -> Path:
    """Separate preprocessing variants instead of overwriting another normalization."""
    model_path = Path(model_path)
    stem = model_path.stem
    suffix = model_path.suffix or ".joblib"
    svm_type = _clean_svm_type(svm_type)
    stem = re.sub(r"_(?:linear|rbf)(?:_b\d+(?:_[0-9a-f]{16})?)?(?:_norm_[a-z_]+)?$", "", stem)
    wavelengths = np.asarray(matched_wavelengths, dtype=np.float32).reshape(-1)
    normalization_mode = _clean_normalization_mode(normalization_mode)
    band_key = hashlib.sha256(wavelengths.astype("<f4").tobytes()).hexdigest()[:16]
    return model_path.with_name(f"{stem}_{svm_type}_b{wavelengths.size}_{band_key}_norm_{normalization_mode}{suffix}")


def _model_load_candidates(model_path):
    """New canonical path first, then legacy siblings; compatibility is metadata-based."""
    model_path = Path(model_path)
    family = re.sub(r"_(?:linear|rbf)(?:_b\d+(?:_[0-9a-f]{16})?)?(?:_norm_[a-z_]+)?$", "", model_path.stem)
    siblings = sorted(model_path.parent.glob(f'{family}_*.joblib'), key=lambda p: p.stat().st_mtime, reverse=True)
    candidates = [model_path, model_path.with_name(family+'.joblib'), *siblings]
    requested = re.search(r'_(linear|rbf)_b(\d+)_norm_([a-z_]+)$', model_path.stem)
    result = []
    for candidate in dict.fromkeys(candidates):
        if not candidate.is_file() or '_backup_' in candidate.stem:
            continue
        tagged = re.search(r'_(linear|rbf)_b(\d+)_norm_([a-z_]+)$', candidate.stem)
        if requested and tagged and requested.groups() != tagged.groups():
            continue
        result.append(candidate)
    return result


def _make_svm_model(
    band_weights: np.ndarray,
    svm_type: str = SVM_TYPE,
    *,
    linear_c: float = LINEAR_C,
    rbf_c: float = RBF_C,
    rbf_gamma: str | float = RBF_GAMMA,
):
    """Build the shared preprocessing pipeline with the selected SVM kernel."""
    if _REFERENCE_PIPELINE is not None:
        from sklearn.base import clone
        model = clone(_REFERENCE_PIPELINE)
        if 'band_weights' in model.named_steps:
            model.set_params(band_weights__weights=band_weights)
        return model
    svm_type = _clean_svm_type(svm_type)
    steps = []

    scaler_mode = str(SCALER_MODE).lower().strip()
    if scaler_mode == "standard":
        steps.append(("scaler", StandardScaler()))
    elif scaler_mode == "robust":
        steps.append(("scaler", RobustScaler()))
    elif scaler_mode in ["none", "no", "false"]:
        pass
    else:
        raise ValueError(f"Unsupported SCALER_MODE={SCALER_MODE!r}")

    if USE_BAND_WEIGHTS:
        steps.append(("band_weights", BandWeighter(band_weights)))

    if USE_PCA:
        steps.append((
            "pca",
            PCA(n_components=float(PCA_VARIANCE), svd_solver=PCA_SVD_SOLVER),
        ))

    if svm_type == "linear":
        svm = LinearSVC(
            C=float(linear_c),
            class_weight="balanced",
            dual=False,
            loss="squared_hinge",
            max_iter=int(MAX_ITER),
            tol=float(SVM_TOL),
            random_state=RANDOM_STATE,
        )
    elif svm_type == "rbf":
        svm = SVC(
            C=float(rbf_c), gamma=rbf_gamma, kernel="rbf",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    else:
        raise ValueError(f"Unsupported svm_type={svm_type!r}. Use 'linear' or 'rbf'.")

    steps.append(("svm", svm))
    return Pipeline(steps)


def _dataset_signature(path: Path | str) -> Dict[str, int | str]:
    """Small file signature used to reject stale cached SVM models."""
    p = Path(path)
    stat = p.stat()
    return {
        "path": portable_project_path(p),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _training_profile_path(model_path: Path | str) -> Path:
    model_path = Path(model_path)
    return model_path.with_name(f"{model_path.stem}_training_profile.json")


def _grouped_training_key(
    svm_type: str,
    normalization_mode: str,
    selected_wavelengths: Sequence[float],
    selected_cube_ids: Sequence[str],
    calibration_version: Optional[int] = None,
) -> str:
    descriptor = {
        "version": 4,
        "platt_version": CALIBRATION_VERSION if calibration_version is None else calibration_version,
        "application_protocol": "heraman_generic_v1",
        "svm_type": _clean_svm_type(svm_type),
        "normalization": _clean_normalization_mode(normalization_mode),
        "ratio_feature": svm_ratio_feature(normalization_mode),
        "wavelengths": [round(float(value), 4) for value in selected_wavelengths],
        "use_pca": bool(USE_PCA),
        "pca_variance": float(PCA_VARIANCE),
        "pca_solver": str(PCA_SVD_SOLVER),
        "scaler": str(SCALER_MODE),
        "use_band_weights": bool(USE_BAND_WEIGHTS),
        "band_weights": BAND_WEIGHTS_BY_WAVELENGTH if USE_BAND_WEIGHTS else {},
        "default_band_weight": float(DEFAULT_BAND_WEIGHT),
        "class_weight": "balanced",
        "sample_weights": "cube/class balanced",
        "svm_tol": float(SVM_TOL),
        "linear_max_iter": int(MAX_ITER),
        "random_state": RANDOM_STATE,
        "group_cv_random_state": GROUP_CV_RANDOM_STATE,
        "group_cv_max_splits": GROUP_CV_MAX_SPLITS,
        "parameter_policy": "fixed",
        "fixed_params": _default_svm_params(),
        "pixel_selection": "all_labeled_pixels",
        "selected_cube_ids": sorted(str(value) for value in selected_cube_ids),
    }
    raw = json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _resolve_group_aware_training(
    *,
    bundle: Dict[str, Any],
    dataset_path: Path | str,
    model_family_path: Path | str,
    band_indices: Sequence[int],
    selected_wavelengths: Sequence[float],
    normalization_mode: str,
    svm_type: str,
    allow_validation: bool = True,
    force_validation: bool = False,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any], Dict[str, Any]]:
    """Use every labeled pixel and cache cube-grouped fixed-parameter validation."""
    X = np.asarray(bundle["X"], dtype=np.float32)
    y = np.asarray(bundle["y"]).astype(str)
    groups = np.asarray(bundle["sample_cube_id"], dtype=object)
    roi_ids = np.asarray(bundle["roi_id"], dtype=np.int64)
    band_indices = np.asarray(band_indices, dtype=int)

    training_idx = np.arange(X.shape[0], dtype=np.int64)
    training_weights = _cube_class_sample_weights(y[training_idx], groups[training_idx])

    signature = _dataset_signature(dataset_path)
    profile_path = _training_profile_path(model_family_path)
    profile_key = _grouped_training_key(
        svm_type, normalization_mode, selected_wavelengths, np.unique(groups).tolist(),
        # Fat-only calibration changes only multiclass models containing fat.
        # Binary and non-fat models retain their unchanged version-1 protocol.
        calibration_version=(CALIBRATION_VERSION if len(np.unique(y)) > 2 and
                             any(str(name).strip().casefold() == 'fat' for name in np.unique(y)) else 1),
    )
    profile = {
        "version": 1,
        "dataset_signature": signature,
        "configurations": {},
    }
    if profile_path.is_file():
        try:
            loaded = json.loads(profile_path.read_text(encoding="utf-8"))
            if loaded.get("dataset_signature") == signature:
                profile = loaded
        except Exception:
            pass

    cached = (profile.get("configurations", {}) or {}).get(profile_key)
    if (
        not force_validation
        and isinstance(cached, dict)
        and isinstance(cached.get("params"), dict)
    ):
        params = {**_default_svm_params(), **cached["params"]}
        report = dict(cached.get("report", {}))
        report["cache"] = "loaded"
    elif allow_validation:
        cv_idx = np.arange(X.shape[0], dtype=np.int64)
        params, report = _select_grouped_svm_params(
            X[cv_idx][:, band_indices],
            y[cv_idx],
            groups[cv_idx],
            roi_ids[cv_idx],
            normalization_mode=normalization_mode,
            svm_type=svm_type,
            wavelengths=selected_wavelengths,
        )
        report.update({
            "cache": "created",
            "n_available_samples": int(X.shape[0]),
            "n_cv_samples": int(cv_idx.size),
            "cube_ids": [str(value) for value in np.unique(groups).tolist()],
        })
        profile.setdefault("configurations", {})[profile_key] = {
            "svm_type": svm_type,
            "normalization_mode": normalization_mode,
            "selected_wavelengths": [float(value) for value in selected_wavelengths],
            "params": params,
            "report": report,
        }
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = profile_path.with_suffix(profile_path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(profile, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary_path.replace(profile_path)
    else:
        # Normal inference must not unexpectedly launch grouped validation.
        # Explicit training evaluates the same fixed parameters used here.
        params = _default_svm_params()
        report = {
            "status": "default",
            "reason": "grouped validation skipped during classification",
            "cache": "missing",
            "n_cubes": int(len(np.unique(groups))),
            "cube_ids": [str(value) for value in np.unique(groups).tolist()],
        }

    report["n_final_training_samples"] = int(training_idx.size)
    report["profile_path"] = portable_project_path(profile_path)
    report["training_config_key"] = profile_key
    status = str(report.get("status", "unknown"))
    if status in {"evaluated_fixed_parameters", "optimized"}:
        flush_print(
            f"[SVM][GROUP-CV] Fixed parameters evaluated on {report.get('n_cubes')} cubes, "
            f"{report.get('n_splits')} folds, macro-F1={float(report.get('best_score', 0.0)):.4f}, "
            f"params={params}"
        )
    else:
        flush_print(
            f"[SVM][GROUP-CV] Fallback to defaults: {report.get('reason', 'not available')}"
        )
    flush_print(
        f"[SVM][TRAIN] Using all {training_idx.size:,} labeled pixels "
        f"from {len(np.unique(groups))} cube(s); no ROI or cube/class pixel cap"
    )
    return training_idx, training_weights, params, report


def _log_model_summary(action: str, model_path: Path | str, payload: Dict[str, Any]) -> None:
    """Print one concise line describing the model that will actually be used."""
    model_path = Path(model_path)
    model = payload.get("model")
    named_steps = getattr(model, "named_steps", {}) or {}
    pca = named_steps.get("pca") if hasattr(named_steps, "get") else None

    pca_present = pca is not None
    metadata_use_pca = bool(payload.get("use_pca", False))

    if pca_present:
        n_components = getattr(pca, "n_components_", None)
        if n_components is None:
            n_components = payload.get("pca_n_components")

        explained = None
        evr = getattr(pca, "explained_variance_ratio_", None)
        if evr is not None:
            try:
                explained = float(np.sum(evr))
            except Exception:
                explained = None
        if explained is None:
            explained = payload.get("pca_explained_variance")

        target = payload.get("pca_variance")
        try:
            target_txt = f"{100.0 * float(target):.0f}%"
        except (TypeError, ValueError):
            target_txt = "unknown"

        pca_txt = f"PCA=YES target={target_txt}"
        if n_components is not None:
            pca_txt += f" components={int(n_components)}"
        if explained is not None:
            pca_txt += f" explained={100.0 * float(explained):.3f}%"
    else:
        pca_txt = "PCA=NO"

    if metadata_use_pca != pca_present:
        pca_txt += f" [METADATA MISMATCH use_pca={metadata_use_pca}]"

    n_bands = payload.get("n_bands_model")
    bands_txt = f" | bands={int(n_bands)}" if n_bands is not None else ""

    scaler = str(payload.get("scaler_mode", "unknown"))
    svm_type = str(payload.get("svm_type", "unknown"))
    c_value = (
        payload.get("linear_c")
        if svm_type.lower() == "linear"
        else payload.get("rbf_c")
    )
    svm_txt = f"SVM={svm_type}"
    if c_value is not None:
        try:
            svm_txt += f" C={float(c_value):g}"
        except (TypeError, ValueError):
            pass
    if svm_type.lower() == "rbf":
        svm_txt += f" gamma={payload.get('rbf_gamma', 'unknown')}"

    normalization = str(payload.get("normalization_mode", DEFAULT_NORMALIZATION_MODE))
    flush_print(
        f"[SVM][MODEL] {action}: {model_path.name} | {pca_txt} | "
        f"normalization={normalization} | scaler={scaler} | {svm_txt}{bands_txt}"
    )


def _train_or_load_svm_model(
    X: np.ndarray,
    y_str: np.ndarray,
    model_path: Path,
    normalization_mode: str,
    svm_type: str = SVM_TYPE,
    svm_params: Optional[Dict[str, Any]] = None,
    training_sample_weight: Optional[Sequence[float]] = None,
    grouped_training_report: Optional[Dict[str, Any]] = None,
    force_retrain: bool = False,
    forced_band_indices: Optional[Sequence[int]] = None,
    forced_selected_wavelengths: Optional[Sequence[float]] = None,
    dataset_wavelengths: Optional[Sequence[float]] = None,
    dataset_signature: Optional[Dict[str, Any]] = None,
    training_groups: Optional[Sequence[str]] = None,
    require_platt: bool = False,
    inspect_only: bool = False,
    calibration_enabled: Optional[bool] = None,
):
    """
    Returns a payload containing:
    - model
    - label_encoder
    - class_names
    - band_indices
    - selected_wavelengths
    - wavelengths
    - normalization metadata

    inspect_only returns a matching path or None, without fitting or saving.
    """

    svm_type = _clean_svm_type(svm_type)
    svm_params = {**_default_svm_params(), **(svm_params or {})}
    X = np.asarray(X, np.float32)
    y_str = np.asarray(y_str).astype(str)
    if training_sample_weight is not None:
        training_sample_weight = np.asarray(training_sample_weight, dtype=np.float64).reshape(-1)
        if training_sample_weight.size != X.shape[0]:
            raise ValueError("training_sample_weight must match the number of training spectra.")

    n_bands_original = X.shape[1]
    wavelengths = None
    if dataset_wavelengths is not None:
        tmp_wl = np.asarray(dataset_wavelengths, dtype=np.float32)
        if len(tmp_wl) == n_bands_original:
            wavelengths = tmp_wl
    if wavelengths is None:
        wavelengths = get_wavelengths_for_n_bands(n_bands_original, None)

    if forced_band_indices is not None:
        band_indices = np.asarray(forced_band_indices, dtype=int)

        if band_indices.ndim != 1 or len(band_indices) == 0:
            raise ValueError("forced_band_indices must be a non-empty 1D list.")

        if np.any(band_indices < 0) or np.any(band_indices >= n_bands_original):
            raise ValueError(
                f"forced_band_indices contains invalid indices for {n_bands_original} bands: "
                f"{band_indices.tolist()}"
            )

        if forced_selected_wavelengths is not None:
            selected_wavelengths = np.asarray(forced_selected_wavelengths, dtype=np.float32)
        elif wavelengths is not None:
            selected_wavelengths = np.asarray(wavelengths, dtype=np.float32)[band_indices]
        else:
            selected_wavelengths = None
    else:
        band_indices, selected_wavelengths = resolve_band_indices(
            n_bands_original,
            wavelengths=wavelengths
        )

    if not force_retrain:
        for load_path in _model_load_candidates(model_path):
            try:
                payload = joblib.load(str(load_path))
            except Exception as exc:
                flush_print(f"[SVM][CACHE] Cannot read {load_path.name}: {exc}")
                continue
            if not isinstance(payload, dict) or not {'model', 'class_names', 'band_indices'} <= payload.keys():
                continue

            saved_classes = [str(c) for c in payload.get("class_names", [])]
            current_classes = sorted([str(c) for c in np.unique(y_str)])

            if saved_classes and sorted(saved_classes) != current_classes:
                flush_print("[SVM][WARN] Saved SVM classes do not match current dataset.")
                flush_print(f"[SVM][WARN] saved={saved_classes}")
                flush_print(f"[SVM][WARN] current={current_classes}")
                flush_print("[SVM][WARN] Skipping this cached variant; checking other saved models.")
                payload = None

            if payload is not None:
                saved_band_indices = np.asarray(payload.get("band_indices", []), dtype=int)
                if saved_band_indices.size and not np.array_equal(saved_band_indices, band_indices):
                    flush_print("[SVM][WARN] Saved SVM model was trained with different bands.")
                    flush_print(f"[SVM][WARN] saved bands={saved_band_indices.tolist()}")
                    flush_print(f"[SVM][WARN] requested bands={band_indices.tolist()}")
                    flush_print("[SVM][WARN] Skipping this cached variant; checking other saved models.")
                    payload = None

            if payload is not None:
                saved_wavelengths = np.asarray(payload.get('selected_wavelengths', []), dtype=float)
                requested_wavelengths = np.asarray(selected_wavelengths, dtype=float)
                if saved_wavelengths.shape != requested_wavelengths.shape or not np.allclose(saved_wavelengths, requested_wavelengths, rtol=0, atol=1e-5):
                    flush_print('[SVM][CACHE] Wavelengths do not match this configuration.')
                    payload = None

            if payload is not None:
                saved_use_weights = bool(payload.get("use_band_weights", False))
                if saved_use_weights != bool(USE_BAND_WEIGHTS):
                    flush_print("[SVM][WARN] Saved SVM model was trained with a different band-weight setting.")
                    flush_print(f"[SVM][WARN] saved use_band_weights={saved_use_weights}")
                    flush_print(f"[SVM][WARN] requested use_band_weights={bool(USE_BAND_WEIGHTS)}")
                    flush_print("[SVM][WARN] Skipping this cached variant; checking other saved models.")
                    payload = None

            if payload is not None and USE_BAND_WEIGHTS:
                expected_weights = build_band_weights(
                    selected_wavelengths,
                    n_bands=len(band_indices),
                    default_weight=DEFAULT_BAND_WEIGHT
                )
                saved_weights = np.asarray(payload.get("band_weights", []), dtype=np.float32)
                if svm_ratio_feature(normalization_mode):
                    expected_weights = np.r_[expected_weights, np.ones(3 if normalization_mode == "snv_three_ratios" else 1,dtype=np.float32)]
                if saved_weights.size and (saved_weights.shape != expected_weights.shape or not np.allclose(saved_weights, expected_weights)):
                    flush_print("[SVM][WARN] Saved SVM model was trained with different band weights.")
                    flush_print(f"[SVM][WARN] saved weights={saved_weights.tolist()}")
                    flush_print(f"[SVM][WARN] requested weights={expected_weights.tolist()}")
                    flush_print("[SVM][WARN] Skipping this cached variant; checking other saved models.")
                    payload = None

            if payload is not None:
                saved_scaler = str(payload.get("scaler_mode", "")).lower().strip()
                if saved_scaler != str(SCALER_MODE).lower().strip():
                    flush_print("[SVM][WARN] Saved SVM model uses a different scaler.")
                    payload = None

            if payload is not None:
                saved_svm_type = str(payload.get("svm_type", "")).lower().strip()
                if saved_svm_type != svm_type:
                    flush_print("[SVM][WARN] Saved SVM model uses a different SVM type.")
                    payload = None

            if payload is not None:
                if svm_type == "linear":
                    saved_c = payload.get("linear_c")
                    compatible = saved_c is not None and np.isclose(
                        float(saved_c), float(svm_params["linear_c"])
                    )
                else:
                    saved_c = payload.get("rbf_c")
                    saved_gamma = payload.get("rbf_gamma")
                    requested_gamma = svm_params["rbf_gamma"]
                    compatible = saved_c is not None and np.isclose(
                        float(saved_c), float(svm_params["rbf_c"])
                    ) and str(saved_gamma) == str(requested_gamma)
                if not compatible:
                    flush_print("[SVM][WARN] Saved SVM model uses different hyperparameters.")
                    payload = None

            if payload is not None:
                saved_use_pca = bool(payload.get("use_pca", False))
                saved_pca_variance = payload.get("pca_variance", None)
                pca_matches = saved_use_pca == bool(USE_PCA)
                if USE_PCA:
                    try:
                        pca_matches = pca_matches and np.isclose(
                            float(saved_pca_variance), float(PCA_VARIANCE)
                        )
                    except (TypeError, ValueError):
                        pca_matches = False
                if not pca_matches:
                    actual_model = payload.get("model")
                    actual_steps = getattr(actual_model, "named_steps", {}) or {}
                    actual_has_pca = "pca" in actual_steps
                    flush_print(
                        "[SVM][WARN] Saved SVM model uses a different PCA configuration "
                        f"(metadata use_pca={saved_use_pca}, pipeline PCA={actual_has_pca}, "
                        f"saved target={saved_pca_variance}, required target={PCA_VARIANCE})."
                    )
                    payload = None

            if payload is not None:
                saved_norm = str(payload.get("normalization_mode", "none")).lower().strip()
                expected_norm = str(normalization_mode or "none").lower().strip()
                if saved_norm != expected_norm:
                    flush_print("[SVM][WARN] Saved SVM model uses a different spectral normalization.")
                    payload = None

            if payload is not None and payload.get("ratio_feature") != svm_ratio_feature(normalization_mode):
                flush_print("[SVM][WARN] Ratio feature schema changed; skipping this cached variant.")
                payload = None

            if payload is not None and payload.get("application_protocol") != "heraman_generic_v1":
                payload = None
            if payload is not None:
                saved_calibrator = payload.get('platt_calibration') or {}
                if (svm_type == 'linear' and saved_calibrator.get('target') == 'fat'
                        and saved_calibrator.get('score_kind') != 'fat_vs_best_nonfat_v1'):
                    flush_print('[SVM][WARN] Multiclass fat calibration uses the obsolete standalone OVR score; recalibration required.')
                    payload = None

            if payload is not None and grouped_training_report is not None:
                saved_grouped = payload.get("grouped_training") or {}
                expected_key = grouped_training_report.get("training_config_key")
                if (expected_key and saved_grouped.get("training_config_key") != expected_key
                        and (payload.get("platt_calibration") or grouped_training_report.get("cache") != "missing")):
                    flush_print("[SVM][WARN] Saved SVM model predates the current grouped training configuration.")
                    payload = None

            if payload is not None and dataset_signature is not None:
                saved_signature = payload.get("dataset_signature")
                if saved_signature != dataset_signature:
                    flush_print("[SVM][WARN] Saved SVM model belongs to an older dataset version.")
                    flush_print(f"[SVM][WARN] saved signature={saved_signature}")
                    flush_print(f"[SVM][WARN] current signature={dataset_signature}")
                    flush_print("[SVM][WARN] Skipping this cached variant; checking other saved models.")
                    payload = None

            if payload is not None and calibration_enabled is not None and bool(payload.get("platt_calibration")) != calibration_enabled:
                payload = None
            if payload is not None and require_platt and not payload.get("platt_calibration"):
                flush_print("[SVM][PLATT] Saved model is uncalibrated; checking other saved variants.")
                payload = None

            if payload is not None:
                if inspect_only:
                    return {"path": str(load_path), "compatible": True}
                if load_path != model_path:
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    _atomic_joblib_dump(payload, model_path)
                    flush_print(f"[SVM][CACHE] Reused compatible model {load_path.name}; copied to {model_path.name} without fitting.")
                _log_model_summary("Loaded", model_path, payload)
                return payload

        flush_print("[SVM][TRAIN] No compatible saved SVM model found.")
    else:
        flush_print("[SVM][TRAIN] force_retrain=True, training new SVM model.")

    if inspect_only:
        return None

    flush_print("[SVM][TRAIN] Training new SVM model...")
    t0 = time.time()

    X = X[:, band_indices].astype(np.float32)

    if USE_BAND_WEIGHTS:
        band_weights = build_band_weights(
            selected_wavelengths,
            n_bands=X.shape[1],
            default_weight=DEFAULT_BAND_WEIGHT
        )
    else:
        band_weights = np.ones(X.shape[1], dtype=np.float32)

    print_band_selection_summary(
        "[SVM][TRAIN][BANDS]",
        n_bands_original,
        band_indices,
        selected_wavelengths,
        band_weights=band_weights
    )

    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y_str)
    class_names = label_encoder.classes_

    Xn = prepare_svm_features(X, selected_wavelengths, normalization_mode)
    if svm_ratio_feature(normalization_mode):
        band_weights = np.r_[band_weights, np.ones(3 if normalization_mode == "snv_three_ratios" else 1,dtype=np.float32)]

    valid = np.all(np.isfinite(Xn), axis=1)
    Xn = Xn[valid]
    y_enc = y_enc[valid]
    fit_weights = training_sample_weight[valid] if training_sample_weight is not None else None

    flush_print(f"[SVM][TRAIN][DEBUG] USE_BAND_WEIGHTS = {USE_BAND_WEIGHTS}")
    flush_print(f"[SVM][TRAIN][DEBUG] band_weights = {band_weights.tolist()}")

    model = _make_svm_model(
        band_weights,
        svm_type=svm_type,
        **svm_params,
    )

    flush_print(f"[SVM][TRAIN] X shape: {Xn.shape}")
    flush_print(f"[SVM][TRAIN] Classes: {list(class_names)}")
    flush_print(f"[SVM][TRAIN] Model: {model}")

    fit_kwargs = {"svm__sample_weight": fit_weights} if fit_weights is not None else {}
    model.fit(Xn, y_enc, **fit_kwargs)

    enabled = bool(require_platt) if calibration_enabled is None else calibration_enabled
    calibration = (grouped_training_report or {}).get("final_calibrator") if enabled else None
    calibration_reason = None
    if enabled and calibration is None and training_groups is not None:
        try:
            groups = np.asarray(training_groups)[valid]
            scores = _grouped_oof_scores(X[valid], y_enc, groups, selected_wavelengths,
                                         normalization_mode, svm_type, svm_params)
            calibration = fit_platt(scores, y_enc, len(class_names))
        except ValueError as exc:
            calibration_reason = str(exc)
    if calibration:
        flush_print("[SVM][PLATT] Multiclass SVM preserved; only P(fat) calibrated from cube-held-out scores. Other classes retain raw SVM ranking."
                    if calibration.get('target') == 'fat' else
                    "[SVM][PLATT] Final SVM calibrated from unweighted cube-held-out scores; predictions use probability argmax.")
    else:
        calibration_reason = calibration_reason or ("Cube provenance was not supplied." if enabled else "Platt scaling disabled for this model.")
        if require_platt:
            raise ValueError("Model not saved: Platt calibration could not be fitted. " + calibration_reason)
        flush_print(f"[SVM][PLATT][WARN] Model remains UNCALIBRATED: {calibration_reason}")

    if not enabled:
        # Do not present post-calibration validation metrics for a raw SVM.
        grouped_training_report = dict(grouped_training_report or {})
        grouped_training_report.update(final_calibrator=None, calibration={"status": "disabled"},
                                       prediction_rule="Raw SVM", best_score=None, confusion_matrix=None)
        try:
            from sklearn.metrics import confusion_matrix, f1_score
            if training_groups is None:
                raise ValueError("Cube provenance unavailable.")
            raw_scores = _grouped_oof_scores(X[valid], y_enc, np.asarray(training_groups)[valid],
                                             selected_wavelengths, normalization_mode, svm_type, svm_params)
            raw_pred = (raw_scores > 0).astype(int) if raw_scores.ndim == 1 else raw_scores.argmax(axis=1)
            grouped_training_report.update(
                best_score=float(f1_score(y_enc, raw_pred, average="macro")),
                confusion_matrix=confusion_matrix(y_enc, raw_pred, labels=np.arange(len(class_names))).tolist(),
                status="evaluated", reason=None)
        except ValueError as exc:
            grouped_training_report.update(status="unavailable", reason=str(exc))

    pca_components = None
    pca_explained_variance = None
    if USE_PCA and "pca" in model.named_steps:
        fitted_pca = model.named_steps["pca"]
        pca_components = int(getattr(fitted_pca, "n_components_", 0))
        evr = getattr(fitted_pca, "explained_variance_ratio_", None)
        if evr is not None:
            pca_explained_variance = float(np.sum(evr))
        flush_print(
            f"[SVM][TRAIN][PCA] {Xn.shape[1]} bands -> {pca_components} components "
            f"({100.0 * pca_explained_variance:.3f}% variance)"
        )

    payload = {
        "model": model,
        "label_encoder": label_encoder,
        "class_names": np.array(class_names, dtype=object),
        "normalization_mode": normalization_mode,
        "use_selected_bands": bool(USE_SELECTED_BANDS),
        "ratio_feature": svm_ratio_feature(normalization_mode),
        "n_features_model": int(Xn.shape[1]),
        "band_indices": np.asarray(band_indices, dtype=int),
        "selected_wavelengths": selected_wavelengths,
        "wavelengths": wavelengths,
        "n_bands_original": int(n_bands_original),
        "n_bands_model": int(X.shape[1]),
        "scaler_mode": SCALER_MODE,
        "use_pca": bool(USE_PCA),
        "pca_variance": float(PCA_VARIANCE) if USE_PCA else None,
        "pca_svd_solver": PCA_SVD_SOLVER if USE_PCA else None,
        "pca_n_components": pca_components,
        "pca_explained_variance": pca_explained_variance,
        "linear_c": float(svm_params["linear_c"]),
        "rbf_c": float(svm_params["rbf_c"]),
        "rbf_gamma": svm_params["rbf_gamma"],
        "svm_tol": float(SVM_TOL),
        "use_band_weights": bool(USE_BAND_WEIGHTS),
        "band_weights": band_weights,
        "band_weights_by_wavelength": BAND_WEIGHTS_BY_WAVELENGTH,
        "svm_type": svm_type,
        "grouped_training": grouped_training_report,
        "platt_calibration": calibration,
        "platt_enabled": bool(calibration),
        "application_protocol": "heraman_generic_v1",
        "calibration_unavailable_reason": calibration_reason,
        "dataset_signature": dataset_signature,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_joblib_dump(payload, model_path)

    flush_print(f"[SVM][TRAIN] Saved model: {model_path}")
    _log_model_summary("Trained", model_path, payload)
    flush_print(f"[SVM][TRAIN] Done in {time.time() - t0:.2f} s")

    return payload


def train_svm_from_dataset(
    dataset_path: Path | str,
    model_path: Path | str,
    *,
    normalization_mode: str = "none",
    svm_type: str = SVM_TYPE,
    force_retrain: bool = True,
    selected_wavelengths_nm: Optional[Sequence[float]] = None,
    use_band_weights: bool = False,
    band_weights_by_wavelength: Optional[Dict[int, float]] = None,
    default_band_weight: float = 1.0,
    invalidate_related_models: bool = True,
    optimize_all_kernels: bool = False,
    refresh_validation: bool = False,
    included_cube_ids: Optional[Sequence[str]] = None,
    require_platt: bool = False,
) -> Dict[str, Any]:
    """Train the base SVM associated with one ROI dataset.

    The base model uses either every dataset wavelength or the explicitly requested
    wavelengths. Saved kernel/normalization variants are kept independently;
    dataset signatures and preprocessing metadata reject stale variants on use.
    """

    dataset_path = Path(dataset_path)
    model_path = Path(model_path)
    model_family_path = model_path

    normalization_mode = _clean_normalization_mode(normalization_mode)
    svm_type = _clean_svm_type(svm_type)
    flush_print(f"[SVM][TRAIN] Effective normalization: {normalization_mode}")
    flush_print(f"[SVM][TRAIN] Effective SVM type: {svm_type}")

    band_weights_by_wavelength = {
        int(key): float(value) for key, value in (band_weights_by_wavelength or {}).items()
    }
    if any(not np.isfinite(value) or value <= 0 for value in band_weights_by_wavelength.values()):
        raise ValueError("Every wavelength weight must be finite and greater than zero.")
    use_band_weights = bool(use_band_weights and band_weights_by_wavelength)
    global USE_BAND_WEIGHTS, BAND_WEIGHTS_BY_WAVELENGTH, DEFAULT_BAND_WEIGHT
    old_use_weights = USE_BAND_WEIGHTS
    old_weights = dict(BAND_WEIGHTS_BY_WAVELENGTH)
    old_default_weight = DEFAULT_BAND_WEIGHT
    USE_BAND_WEIGHTS = use_band_weights
    BAND_WEIGHTS_BY_WAVELENGTH = band_weights_by_wavelength
    DEFAULT_BAND_WEIGHT = float(default_band_weight)
    if USE_BAND_WEIGHTS:
        flush_print(f"[SVM][TRAIN] Manual wavelength weights: {BAND_WEIGHTS_BY_WAVELENGTH}")

    if not dataset_path.is_file():
        raise FileNotFoundError(f"ROI dataset not found: {dataset_path}")

    bundle = _load_dataset_bundle(dataset_path)
    if included_cube_ids is not None:
        requested_cube_ids = [str(value) for value in included_cube_ids]
        if not requested_cube_ids:
            raise ValueError("Select at least one cube for SVM training.")
        available_cube_ids = [str(value) for value in np.asarray(bundle["cube_ids"], dtype=object)]
        unknown_cube_ids = sorted(set(requested_cube_ids) - set(available_cube_ids))
        if unknown_cube_ids:
            raise ValueError(
                "Requested training cube(s) are not in the dataset: "
                + ", ".join(unknown_cube_ids)
            )
        selected_cube_ids = set(requested_cube_ids)
        selected_mask = np.isin(
            np.asarray(bundle["sample_cube_id"], dtype=str), list(selected_cube_ids),
        )
        if not np.any(selected_mask):
            raise ValueError("The selected cubes contain no labeled pixels.")
        bundle = dict(bundle)
        for key in ("X", "y", "sample_cube_index", "sample_cube_id", "roi_id"):
            bundle[key] = np.asarray(bundle[key])[selected_mask]
        bundle["cube_ids"] = np.asarray(
            [cube_id for cube_id in available_cube_ids if cube_id in selected_cube_ids],
            dtype=object,
        )
        flush_print(
            "[SVM][TRAIN] Cube selection: " + ", ".join(bundle["cube_ids"].tolist())
        )

    X = np.asarray(bundle["X"], dtype=np.float32)
    y = np.asarray(bundle["y"]).astype(str)
    wavelengths = bundle["wavelengths"]
    classes = np.unique(np.asarray(y).astype(str))
    if classes.size < 2:
        raise ValueError(
            f"At least two classes are required to train an SVM. "
            f"Dataset {dataset_path} contains: {classes.tolist()}"
        )

    if wavelengths is None or len(wavelengths) != X.shape[1]:
        raise ValueError(
            "Could not determine the wavelength axis of the ROI dataset. "
            "Keep the original cube_paths metadata or save a wavelengths array in the NPZ file."
        )

    wavelengths = np.asarray(wavelengths, dtype=np.float32)

    if selected_wavelengths_nm is None:
        band_indices = np.arange(X.shape[1], dtype=int)
        selected_wavelengths = wavelengths.copy()
    else:
        requested = np.asarray(selected_wavelengths_nm, dtype=np.float32)
        band_indices = []
        selected = []
        for wl in requested:
            idx = int(np.argmin(np.abs(wavelengths - wl)))
            diff = float(abs(wavelengths[idx] - wl))
            if diff > WAVELENGTH_MATCH_TOL_NM:
                raise ValueError(
                    f"Dataset {dataset_path.name} does not contain {float(wl):g} nm. "
                    f"Closest wavelength is {float(wavelengths[idx]):g} nm."
                )
            band_indices.append(idx)
            selected.append(float(wavelengths[idx]))
        band_indices = np.asarray(band_indices, dtype=int)
        selected_wavelengths = np.asarray(selected, dtype=np.float32)

    flush_print(f"[SVM][TRAIN] Fitting bands ({len(selected_wavelengths)}): "
                + ", ".join(f"{value:g}" for value in selected_wavelengths) + " nm")
    prepare_svm_features(np.ones((1,len(selected_wavelengths)),np.float32),selected_wavelengths,normalization_mode)
    training_idx, training_weights, svm_params, grouped_report = _resolve_group_aware_training(
        bundle=bundle,
        dataset_path=dataset_path,
        model_family_path=model_family_path,
        band_indices=band_indices,
        selected_wavelengths=selected_wavelengths,
        normalization_mode=normalization_mode,
        svm_type=svm_type,
        force_validation=bool(refresh_validation),
        allow_validation=bool(require_platt),
    )

    grouped_reports = {svm_type: grouped_report}
    if optimize_all_kernels:
        for other_svm_type in sorted(SUPPORTED_SVM_TYPES - {svm_type}):
            _, _, _, other_report = _resolve_group_aware_training(
                bundle=bundle,
                dataset_path=dataset_path,
                model_family_path=model_family_path,
                band_indices=band_indices,
                selected_wavelengths=selected_wavelengths,
                normalization_mode=normalization_mode,
                svm_type=other_svm_type,
            )
            grouped_reports[other_svm_type] = other_report

    model_path = _make_wavelength_model_path(
        model_family_path, selected_wavelengths, normalization_mode,
        svm_type=svm_type, svm_params=svm_params,
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Preserve other preprocessing variants. Their signatures are checked on use;
    # deleting by a broad prefix could remove SNV-without-ratio when fitting SNV.

    signature = _dataset_signature(dataset_path)

    try:
        payload = _train_or_load_svm_model(
            X=X[training_idx],
            y_str=y[training_idx],
            model_path=model_path,
            normalization_mode=normalization_mode,
            svm_type=svm_type,
            svm_params=svm_params,
            training_sample_weight=training_weights,
            grouped_training_report=grouped_report,
            force_retrain=force_retrain,
            forced_band_indices=band_indices,
            forced_selected_wavelengths=selected_wavelengths,
            dataset_wavelengths=wavelengths,
            dataset_signature=signature,
            training_groups=np.asarray(bundle["sample_cube_id"])[training_idx],
            require_platt=require_platt,
            calibration_enabled=bool(require_platt),
        )
    finally:
        USE_BAND_WEIGHTS = old_use_weights
        BAND_WEIGHTS_BY_WAVELENGTH = old_weights
        DEFAULT_BAND_WEIGHT = old_default_weight

    payload["dataset_path"] = signature["path"]
    payload["dataset_size_bytes"] = int(signature["size_bytes"])
    payload["dataset_mtime_ns"] = int(signature["mtime_ns"])
    payload["dataset_signature"] = signature
    payload["n_available_samples"] = int(X.shape[0])
    payload["n_training_samples"] = int(training_idx.size)
    payload["dataset_classes"] = np.asarray(classes, dtype=object)
    payload["dataset_cube_ids"] = np.asarray(bundle["cube_ids"], dtype=object)
    payload["all_kernel_grouped_training"] = grouped_reports
    payload["model_family_path"] = portable_project_path(model_family_path)
    _atomic_joblib_dump(payload, model_path)

    flush_print(f"[SVM][TRAIN] Dataset: {dataset_path}")
    flush_print(f"[SVM][TRAIN] Base model: {model_path}")
    flush_print(f"SVM_MODEL_PATH {model_path}")
    return payload


# ============================================================
# SCORE TO CONFIDENCE
# ============================================================

def _softmax_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)

    if scores.ndim == 1:
        # Binary SVM decision score.
        scores = np.stack([-scores, scores], axis=1)

    scores = scores - np.nanmax(scores, axis=1, keepdims=True)
    exp = np.exp(scores)
    prob = exp / (np.nansum(exp, axis=1, keepdims=True) + 1e-12)

    return prob.astype(np.float32)


def _predict_confidence_in_chunks(model, X: np.ndarray, n_classes: int, chunk_size: int = 250_000, return_scores=False, calibrator=None):
    """
    Returns:
        prob: (N, C)

    Calibrated models return Platt probabilities. Legacy models return one-hot
    raw predictions. Optional gaps/favourites remain raw SVM score diagnostics.
    """
    N = X.shape[0]
    probs = []
    gap_parts, favorite_parts, runner_parts = [], [], []

    for i0 in range(0, N, chunk_size):
        i1 = min(i0 + chunk_size, N)
        flush_print(f"[SVM][INFER] Pixels {i0:,} -> {i1:,} / {N:,}")

        Xb = X[i0:i1]

        scores = model.decision_function(Xb) if calibrator or return_scores else None
        if calibrator:
            pb = platt_probabilities(calibrator, calibration_decision_scores(model, Xb, calibrator, scores)).astype(np.float32)
        else:
            pred = model.predict(Xb).astype(np.int32)
            pb = np.zeros((len(pred), n_classes), dtype=np.float32)
            pb[np.arange(len(pred)), pred] = 1.0
        if return_scores:
            gap, favorite, runner = decision_gap(scores, n_classes)
            gap_parts.append(gap)
            favorite_parts.append(favorite)
            runner_parts.append(runner)

        probs.append(pb)

    result = np.vstack(probs).astype(np.float32) if probs else np.zeros((0, int(n_classes)), dtype=np.float32)
    if return_scores:
        return (result,
                np.concatenate(gap_parts) if gap_parts else np.empty(0, dtype=np.float32),
                np.concatenate(favorite_parts) if favorite_parts else np.empty(0, dtype=np.int32),
                np.concatenate(runner_parts) if runner_parts else np.empty(0, dtype=np.int32))
    return result


# ============================================================
# MAIN PUBLIC API
# ============================================================

def unmix_from_cube(
    cube: np.ndarray,
    lib_path: Path | str,
    out_dir: Path | str,
    bands_sel: Optional[List[int]] = None,
    band_mode: str = "all",
    selected_wavelengths_nm: Optional[Sequence[float]] = None,
    use_band_weights: bool = False,
    band_weights_by_wavelength: Optional[Dict[int, float]] = None,
    default_band_weight: float = 1.0,
    fisher_top_k: int | None = None,
    custom_bands: Optional[Sequence[int]] = None,
    add_shade: bool = False,
    shade_mode: str = "constant",
    shade_class: str = "shadow",
    unmix_method: str = "svm",
    sum_to_one_penalty: float = 50.0,
    use_superpixels: bool = False,
    fov_mm: float = 70.0,
    target_um: float = 500.0,
    compactness: float = 20.0,
    sigma: float = 1.0,
    enforce_conn: bool = False,
    overlay_alpha: float = OVERLAY_ALPHA,
    make_hard_labels: bool = MAKE_HARD_LABELS,
    hard_min_abundance: float = HARD_MIN_ABUNDANCE,
    normalization_mode: str = "none",
    svm_type: str = SVM_TYPE,
    force_retrain: bool = False,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    dataset_path: Path | str = DEFAULT_NPZ_DATASET,
    chunk_size: int = 250_000,
    spectra_source_cube: Optional[np.ndarray] = None,
    prediction_only: bool = False,
    preclassified_labels: Optional[np.ndarray] = None,
    preclassified_class_names: Optional[Sequence[str]] = None,
    preclassified_probabilities: Optional[np.ndarray] = None,
    bypass_class_names: Optional[Sequence[str]] = None,
    bypass_erode_px: int = 0,
    preclassified_fixed_mask: Optional[np.ndarray] = None,
    return_decision_margin: bool = False,
    cube_wavelengths_nm: Optional[Sequence[float]] = None,
    prediction_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compatible with csvm_classifier.unmix_from_cube(...).

    The SVM produces class confidence maps.
    These maps are saved as abundances.hdr for GUI compatibility.
    """
    svm_type = _clean_svm_type(svm_type)
    svm_c = LINEAR_C if svm_type == "linear" else RBF_C
    flush_print(
        f"[SVM][WORKER] File: {Path(__file__).resolve()} | "
        f"PCA={'ON' if USE_PCA else 'OFF'} target={int(round(PCA_VARIANCE * 100)) if USE_PCA else '-'}% | "
        f"scaler={SCALER_MODE} | SVM={svm_type} C={svm_c:g}"
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(model_path)
    dataset_path = Path(dataset_path)

    normalization_mode = _clean_normalization_mode(normalization_mode)
    flush_print(f"[SVM] Effective normalization: {normalization_mode}")

    if use_band_weights:
        flush_print("[SVM][PCA] Manual band weights disabled by benchmark configuration.")
    use_band_weights = False
    band_weights_by_wavelength = {}
    default_band_weight = 1.0

    cube = np.asarray(cube, np.float32)

    if cube.ndim != 3:
        raise ValueError(f"cube must be 3D (H, W, B). Got: {cube.shape}")

    H, W, N = cube.shape

    flush_print("=" * 70)
    flush_print("[SVM] unmix_from_cube")
    flush_print("=" * 70)
    flush_print(f"[SVM] Cube shape: {cube.shape}")
    flush_print(f"[SVM] Dataset path: {dataset_path}")
    flush_print(f"[SVM] Model path  : {model_path}")
    flush_print(f"[SVM] Out dir     : {out_dir}")

    # ------------------------------------------------------------
    # Base image
    # ------------------------------------------------------------

    cube_dir = out_dir / "cube"
    cube_files = list(cube_dir.glob("*_cube.*"))

    if cube_files:
        cube_name = cube_files[0].stem.replace("_cube", "")
    else:
        cube_name = out_dir.name.replace("_corrected", "")

    existing_base_path = out_dir / "base_image_bgr.png"
    legacy_mean_path = cube_dir / f"{cube_name}_mean.png"

    base_image_path = None

    if existing_base_path.exists():
        base_bgr = read_image(str(existing_base_path), cv2.IMREAD_COLOR)
        if base_bgr is None:
            raise FileNotFoundError(f"Unable to read existing base image: {existing_base_path}")
        base_image_path = existing_base_path
        flush_print(f"[SVM] Using existing base image: {existing_base_path}")

    elif legacy_mean_path.exists():
        base_bgr = read_image(str(legacy_mean_path), cv2.IMREAD_COLOR)
        if base_bgr is None:
            raise FileNotFoundError(f"Unable to read legacy mean image: {legacy_mean_path}")
        base_image_path = legacy_mean_path
        flush_print(f"[SVM] Using legacy mean image: {legacy_mean_path}")

    else:
        base_img = cube.mean(axis=2)
        base8 = to_uint8(base_img)
        base_bgr = cv2.cvtColor(base8, cv2.COLOR_GRAY2BGR)
        flush_print("[SVM] No base image found, using in-memory grayscale preview")

    # ------------------------------------------------------------
    # Band alignment from real wavelengths
    # ------------------------------------------------------------

    cube_wavelengths = (_infer_cube_wavelengths(N, out_dir) if cube_wavelengths_nm is None
                        else np.asarray(cube_wavelengths_nm, dtype=np.float32).reshape(-1))
    if cube_wavelengths is not None and (not np.isfinite(cube_wavelengths).all() or np.unique(cube_wavelengths).size != N):
        raise ValueError("Invalid cube wavelength axis.")
    if cube_wavelengths is None or len(cube_wavelengths) != N:
        raise ValueError(
            "Could not infer cube wavelengths safely.\n"
            "Because postprocessing_pipeline passes only the cube array to unmix_from_cube, "
            "svm_classifier infers wavelengths from corr_XXXnm files in out_dir.\n"
            f"out_dir={out_dir}\n"
            f"cube bands={N}\n"
            "Fix: keep the corr_XXXnm files beside the cube, or pass/save wavelengths explicitly."
        )

    flush_print(f"[SVM][CUBE] Wavelengths nm: {np.asarray(cube_wavelengths).tolist()}")

    # ------------------------------------------------------------
    # Load dataset and find common wavelengths
    # ------------------------------------------------------------

    dataset_bundle = _load_dataset_bundle(dataset_path)
    X_ds = np.asarray(dataset_bundle["X"], dtype=np.float32)
    y_ds = np.asarray(dataset_bundle["y"]).astype(str)
    wavelengths_ds = dataset_bundle["wavelengths"]

    if wavelengths_ds is None or len(wavelengths_ds) != X_ds.shape[1]:
        if X_ds.shape[1] != len(cube_wavelengths):
            raise ValueError(
                "Could not infer dataset wavelengths safely.\n"
                f"dataset_path={dataset_path}\n"
                f"dataset bands={X_ds.shape[1]}\n"
                f"cube bands={len(cube_wavelengths)}\n"
                "The missing axis cannot be migrated because the band counts differ."
            )

        # Migration automatique pour les anciens datasets sans axe spectral.
        # Le cube courant possède un axe vérifié depuis corr_XXXnm et le nombre
        # de bandes est identique, donc cette information est ajoutée au NPZ.
        wavelengths_ds = np.asarray(cube_wavelengths, dtype=np.float32).copy()
        flush_print(
            "[SVM][MIGRATION] Dataset without wavelengths: using the verified "
            "current cube axis and saving it to the dataset."
        )
        try:
            _persist_dataset_wavelengths(dataset_path, wavelengths_ds)
            flush_print(f"[SVM][MIGRATION] Dataset updated: {dataset_path}")
        except Exception as exc:
            # La classification peut continuer pour ce run même si le fichier
            # est en lecture seule; un avertissement explicite est conservé.
            flush_print(f"[SVM][WARN] Could not persist wavelengths: {exc}")

    flush_print(f"[SVM][DATASET] X shape: {X_ds.shape}")
    flush_print(f"[SVM][DATASET] Wavelengths nm: {np.asarray(wavelengths_ds).tolist()}")

    # band_mode='all' means Option B: all wavelengths common to dataset and cube.
    if band_mode == "all":
        requested_wl = None

    elif band_mode in ["custom", "selected", "manual"]:
        if selected_wavelengths_nm is not None:
            requested_wl = selected_wavelengths_nm
        elif bands_sel is not None:
            bands_sel_arr = np.asarray(bands_sel, dtype=int)
            if np.any(bands_sel_arr < 0) or np.any(bands_sel_arr >= len(cube_wavelengths)):
                raise ValueError(
                    f"Invalid bands_sel for cube with {N} bands: {bands_sel_arr.tolist()}"
                )
            requested_wl = np.asarray(cube_wavelengths, dtype=np.float32)[bands_sel_arr]
        else:
            raise ValueError(
                "band_mode='custom' requires either selected_wavelengths_nm=[...] or bands_sel=[...]."
            )
    else:
        raise ValueError(f"Unsupported band_mode={band_mode!r}. Use 'all' or 'custom'.")

    dataset_band_indices, cube_band_indices, matched_wavelengths = _match_wavelength_indices(
        dataset_wavelengths=wavelengths_ds,
        cube_wavelengths=cube_wavelengths,
        requested_wavelengths=requested_wl,
        tol_nm=WAVELENGTH_MATCH_TOL_NM,
    )
    # Select hyperparameters with cube-grouped folds using every labeled ROI pixel.
    training_idx, training_weights, svm_params, grouped_report = _resolve_group_aware_training(
        bundle=dataset_bundle,
        dataset_path=dataset_path,
        model_family_path=model_path,
        band_indices=dataset_band_indices,
        selected_wavelengths=matched_wavelengths,
        normalization_mode=normalization_mode,
        svm_type=svm_type,
        allow_validation=False,
    )

    # Use one model per actual common wavelength set and selected configuration.
    model_path = _make_wavelength_model_path(
        model_path,
        matched_wavelengths,
        normalization_mode,
        svm_type=svm_type,
        svm_params=svm_params,
    )
    flush_print(f"[SVM] Wavelength-specific model path: {model_path}")

    print_band_selection_summary(
        "[SVM][ALIGN][DATASET]",
        X_ds.shape[1],
        dataset_band_indices,
        matched_wavelengths,
        band_weights=None,
    )
    print_band_selection_summary(
        "[SVM][ALIGN][CUBE]",
        N,
        cube_band_indices,
        matched_wavelengths,
        band_weights=None,
    )

    # ------------------------------------------------------------
    # Train/load SVM model on dataset common wavelengths
    # ------------------------------------------------------------

    global DEFAULT_WAVELENGTHS_NM, USE_BAND_WEIGHTS, BAND_WEIGHTS_BY_WAVELENGTH, DEFAULT_BAND_WEIGHT

    old_default_wl = DEFAULT_WAVELENGTHS_NM
    old_use_band_weights = USE_BAND_WEIGHTS
    old_band_weights_by_wavelength = dict(BAND_WEIGHTS_BY_WAVELENGTH)
    old_default_band_weight = DEFAULT_BAND_WEIGHT

    DEFAULT_WAVELENGTHS_NM = np.asarray(wavelengths_ds, dtype=np.float32)
    USE_BAND_WEIGHTS = bool(use_band_weights)

    if band_weights_by_wavelength is not None:
        BAND_WEIGHTS_BY_WAVELENGTH = {
            int(k): float(v)
            for k, v in band_weights_by_wavelength.items()
        }

    DEFAULT_BAND_WEIGHT = float(default_band_weight)

    flush_print(f"[SVM][WEIGHTS] use_band_weights = {USE_BAND_WEIGHTS}")
    flush_print(f"[SVM][WEIGHTS] band_weights_by_wavelength = {BAND_WEIGHTS_BY_WAVELENGTH}")
    flush_print(f"[SVM][WEIGHTS] default_band_weight = {DEFAULT_BAND_WEIGHT}")

    try:
        payload = _train_or_load_svm_model(
            X=X_ds[training_idx],
            y_str=y_ds[training_idx],
            model_path=model_path,
            normalization_mode=normalization_mode,
            svm_type=svm_type,
            svm_params=svm_params,
            training_sample_weight=training_weights,
            grouped_training_report=grouped_report,
            force_retrain=force_retrain,
            forced_band_indices=dataset_band_indices,
            forced_selected_wavelengths=matched_wavelengths,
            dataset_wavelengths=wavelengths_ds,
            dataset_signature=_dataset_signature(dataset_path),
            training_groups=np.asarray(dataset_bundle["sample_cube_id"])[training_idx],
        )
    finally:
        DEFAULT_WAVELENGTHS_NM = old_default_wl
        USE_BAND_WEIGHTS = old_use_band_weights
        BAND_WEIGHTS_BY_WAVELENGTH = old_band_weights_by_wavelength
        DEFAULT_BAND_WEIGHT = old_default_band_weight

    model = payload["model"]
    class_names = [str(c) for c in payload["class_names"]]
    names = list(class_names)
    L = len(names)

    # IMPORTANT:
    # Training uses dataset_band_indices, inference uses cube_band_indices.
    # They can differ when the cube starts at 460 nm but the dataset starts at 420 nm.
    train_band_indices = np.asarray(payload["band_indices"], dtype=int)
    band_indices = np.asarray(cube_band_indices, dtype=int)
    selected_wavelengths = np.asarray(matched_wavelengths, dtype=np.float32)
    wavelengths = np.asarray(wavelengths_ds, dtype=np.float32)

    saved_wl = payload.get("selected_wavelengths", None)
    if saved_wl is not None and not np.allclose(np.asarray(saved_wl, dtype=np.float32), selected_wavelengths):
        raise RuntimeError(
            "Saved SVM model wavelength mismatch. Use force_retrain=True or delete the cached model.\n"
            f"saved={np.asarray(saved_wl, dtype=np.float32).tolist()}\n"
            f"requested={selected_wavelengths.tolist()}"
        )

    if np.any(band_indices < 0) or np.any(band_indices >= N):
        raise ValueError(
            f"Cube inference band_indices are invalid for cube with {N} bands: "
            f"{band_indices.tolist()}"
        )

    flush_print(f"[SVM][DEBUG] train dataset band_indices = {train_band_indices.tolist()}")
    flush_print(f"[SVM][DEBUG] infer cube band_indices    = {band_indices.tolist()}")

    print_band_selection_summary(
        "[SVM][INFER][BANDS]",
        N,
        band_indices,
        selected_wavelengths,
        band_weights=payload.get("band_weights", None)
    )

    # ------------------------------------------------------------
    # Cube inference
    # ------------------------------------------------------------

    cube_sel = cube[:, :, band_indices].astype(np.float32)
    Bsel = cube_sel.shape[2]

    X_flat = cube_sel.reshape(-1, Bsel)
    valid_mask = np.all(np.isfinite(X_flat), axis=1)
    if prediction_mask is not None:
        region = np.asarray(prediction_mask, dtype=bool)
        if region.shape != (H, W):
            raise ValueError("Prediction mask must match the cube's spatial shape.")
        valid_mask &= region.reshape(-1)

    flush_print(f"[SVM][INFER] Total pixels: {X_flat.shape[0]:,}")
    flush_print(f"[SVM][INFER] Valid pixels: {int(valid_mask.sum()):,}")

    # A cheap preliminary classifier may lock obvious non-biological regions
    # so an expensive final RBF only evaluates the remaining biological pixels.
    inference_mask = valid_mask.copy()
    fixed_class_flat = np.full(X_flat.shape[0], UNKNOWN_LABEL, dtype=np.int32)
    bypass_counts: Dict[str, int] = {}
    if (
        preclassified_labels is not None
        and preclassified_class_names is not None
        and preclassified_fixed_mask is not None
    ):
        preliminary = np.asarray(preclassified_labels, dtype=np.int32)
        fixed_requested = np.asarray(preclassified_fixed_mask, dtype=bool)
        if preliminary.shape != (H, W) or fixed_requested.shape != (H, W):
            raise ValueError(
                "preclassified labels/fixed mask shape mismatch: "
                f"labels={preliminary.shape}, mask={fixed_requested.shape}, expected={(H, W)}"
            )
        final_name_to_index = {
            str(name).strip().lower(): index for index, name in enumerate(names)
        }
        fixed_requested_flat = fixed_requested.reshape(-1) & valid_mask
        for preliminary_index, preliminary_name in enumerate(preclassified_class_names):
            final_index = final_name_to_index.get(str(preliminary_name).strip().lower())
            if final_index is None:
                continue
            selected = (
                (preliminary.reshape(-1) == preliminary_index)
                & fixed_requested_flat
            )
            inference_mask[selected] = False
            fixed_class_flat[selected] = int(final_index)

        fixed_total = int((fixed_class_flat != UNKNOWN_LABEL).sum())
        flush_print(
            f"[SVM][CASCADE] Reusing {fixed_total:,} full-grid labels; "
            f"{int(inference_mask.sum()):,}/{int(valid_mask.sum()):,} valid pixels "
            f"are evaluated by {svm_type.upper()}"
        )
    elif (
        preclassified_labels is not None
        and preclassified_class_names is not None
        and bypass_class_names
    ):
        preliminary = np.asarray(preclassified_labels, dtype=np.int32)
        if preliminary.shape != (H, W):
            raise ValueError(
                f"preclassified_labels shape mismatch: {preliminary.shape} != {(H, W)}"
            )
        final_name_to_index = {
            str(name).strip().lower(): index for index, name in enumerate(names)
        }
        bypass_set = {str(name).strip().lower() for name in bypass_class_names}
        erosion_px = max(0, int(bypass_erode_px))
        kernel = None
        if erosion_px > 0:
            size = 2 * erosion_px + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

        for preliminary_index, preliminary_name in enumerate(preclassified_class_names):
            normalized_name = str(preliminary_name).strip().lower()
            final_index = final_name_to_index.get(normalized_name)
            if normalized_name not in bypass_set or final_index is None:
                continue
            class_mask = (preliminary == preliminary_index).astype(np.uint8)
            if kernel is not None and np.any(class_mask):
                class_mask = cv2.erode(class_mask, kernel, iterations=1)
            selected = class_mask.reshape(-1).astype(bool) & valid_mask
            inference_mask[selected] = False
            fixed_class_flat[selected] = int(final_index)
            bypass_counts[str(names[final_index])] = int(selected.sum())

        bypass_total = int((fixed_class_flat != UNKNOWN_LABEL).sum())
        flush_print(
            f"[SVM][BYPASS] Preliminary Linear locked {bypass_total:,} "
            f"non-biological pixels {bypass_counts}; RBF evaluates "
            f"{int(inference_mask.sum()):,}/{int(valid_mask.sum()):,} valid pixels "
            f"on the native {W}x{H} grid (boundary erosion={erosion_px}px)"
        )

    X_valid = X_flat[inference_mask]
    X_valid = prepare_svm_features(X_valid, matched_wavelengths, normalization_mode)

    if prediction_only:
        pred_parts = []
        probability_parts = []
        margin_parts = []
        for i0 in range(0, X_valid.shape[0], chunk_size):
            i1 = min(i0 + chunk_size, X_valid.shape[0])
            flush_print(
                f"[SVM][INFER] Pixels {i0:,} -> {i1:,} / {X_valid.shape[0]:,}"
            )
            batch = X_valid[i0:i1]
            scores = None
            if payload.get("platt_calibration"):
                scores = np.asarray(model.decision_function(batch))
                batch_probability = platt_probabilities(payload["platt_calibration"], calibration_decision_scores(model, batch, payload['platt_calibration'], scores))
                probability_parts.append(batch_probability)
                pred_parts.append(batch_probability.argmax(axis=1).astype(np.int32))
            else:
                pred_parts.append(model.predict(batch).astype(np.int32))
            if return_decision_margin:
                if scores is None:
                    scores = np.asarray(model.decision_function(batch))
                if scores.ndim == 1:
                    margin = np.abs(scores)
                elif scores.shape[1] < 2:
                    margin = np.abs(scores[:, 0])
                else:
                    top_two = np.partition(scores, -2, axis=1)[:, -2:]
                    margin = top_two[:, 1] - top_two[:, 0]
                margin_parts.append(np.asarray(margin, dtype=np.float32))

        pred_valid = (
            np.concatenate(pred_parts)
            if pred_parts
            else np.empty((0,), dtype=np.int32)
        )
        labels_flat = np.full(X_flat.shape[0], UNKNOWN_LABEL, dtype=np.int32)
        labels_flat[inference_mask] = pred_valid
        fixed_mask = fixed_class_flat != UNKNOWN_LABEL
        labels_flat[fixed_mask] = fixed_class_flat[fixed_mask]

        decision_margin = None
        if return_decision_margin:
            margin_valid = (
                np.concatenate(margin_parts)
                if margin_parts
                else np.empty((0,), dtype=np.float32)
            )
            margin_flat = np.full(X_flat.shape[0], np.nan, dtype=np.float32)
            margin_flat[inference_mask] = margin_valid
            decision_margin = margin_flat.reshape(H, W)

        normalized_names = [name.lower().strip() for name in names]
        if "background" in normalized_names and prediction_mask is None:
            labels_flat[~valid_mask] = normalized_names.index("background")

        flush_print("[SVM] Prediction-only pass done.")
        result = {
            "names": names,
            "hard_labels": labels_flat.reshape(H, W),
            "svm_type": svm_type,
            "model_path": str(model_path),
        }
        if decision_margin is not None:
            result["decision_margin"] = decision_margin
        if payload.get("platt_calibration"):
            full_probability = np.full((X_flat.shape[0], L), np.nan, np.float32)
            if probability_parts:
                full_probability[inference_mask] = np.concatenate(probability_parts)
            result["class_probabilities"] = full_probability.reshape(H, W, L)
        return result

    prob_valid, gap_valid, favorite_valid, runner_valid = _predict_confidence_in_chunks(
        model,
        X_valid,
        n_classes=L,
        chunk_size=chunk_size,
        return_scores=True,
        calibrator=payload.get("platt_calibration"),
    )

    if prob_valid.shape[1] != L:
        raise RuntimeError(
            f"Probability shape mismatch: got {prob_valid.shape[1]} classes, expected {L}."
        )

    from shared.envi import write_envi
    probability_flat = None
    unsure_flat = np.zeros(X_flat.shape[0], bool)
    if payload.get("platt_calibration"):
        # Preserve the GUI's categorical overlays. Keep actual probabilities in
        # a separate map, with NaN for pixels that were not evaluated by this SVM.
        probability_flat = np.full((X_flat.shape[0], L), np.nan, dtype=np.float32)
        probability_flat[inference_mask] = prob_valid
        predicted = np.argmax(prob_valid, axis=1)
        prob_valid = np.eye(L, dtype=np.float32)[predicted]

    prob_flat = np.zeros((X_flat.shape[0], L), dtype=np.float32)
    prob_flat[inference_mask] = prob_valid
    fixed_mask = fixed_class_flat != UNKNOWN_LABEL
    fixed_rows = np.flatnonzero(fixed_mask)
    if fixed_rows.size:
        prob_flat[fixed_rows, fixed_class_flat[fixed_rows]] = 1.0

    if probability_flat is not None:
        if preclassified_probabilities is not None:
            preliminary_p = np.asarray(preclassified_probabilities)
            if preliminary_p.shape != (H, W, len(preclassified_class_names)):
                raise ValueError("Preclassified probability shape mismatch.")
            for column, name in enumerate(names):
                if name in preclassified_class_names:
                    source_column = list(preclassified_class_names).index(name)
                    probability_flat[fixed_rows, column] = preliminary_p.reshape(-1, preliminary_p.shape[-1])[fixed_rows, source_column]
        evaluated = np.isfinite(probability_flat).all(axis=1)
        assigned = np.full(X_flat.shape[0], UNKNOWN_LABEL, dtype=int)
        assigned[evaluated] = probability_flat[evaluated].argmax(axis=1)
        prob_flat[evaluated] = np.eye(L, dtype=np.float32)[assigned[evaluated]]
        write_envi(out_dir / "class_probabilities.hdr", probability_flat.reshape(H, W, L), band_names=names)
        write_envi(out_dir / "unsure_mask.hdr", unsure_flat.reshape(H, W), band_names=["unsure"])
        write_image(str(out_dir / "mask_unsure.png"), unsure_flat.reshape(H, W).astype(np.uint8)*255)
        flush_print("[SVM][PLATT] Generic classification: highest calibrated class probability.")

    # Invalid pixels become background if background exists; otherwise all zeros.
    if prediction_mask is None and "background" in [n.lower().strip() for n in names]:
        bg_idx = [n.lower().strip() for n in names].index("background")
        prob_flat[~valid_mask, bg_idx] = 1.0

    abund = prob_flat.reshape(H, W, L).astype(np.float32)

    from shared.envi import write_envi
    abund_path = write_envi(out_dir / "abundances.hdr", abund, band_names=names)
    if probability_flat is not None:
        (out_dir / "class_probabilities_metadata.json").write_text(json.dumps(dict(
            version=1, class_names=names, method="Platt", prediction_rule="argmax",
            normalization_mode=normalization_mode, selected_wavelengths=np.asarray(matched_wavelengths).tolist(),
            calibrated_classes=(payload.get('platt_calibration') or {}).get('calibrated_classes', names),
            other_columns=(payload.get('platt_calibration') or {}).get('other_columns'),
            abundance_sha256=file_digest(abund_path.with_suffix(".raw")),
            probability_sha256=file_digest(out_dir / "class_probabilities.raw")), indent=2), encoding="utf-8")
    # Save scores from this exact inference, including its normalization and
    # shading correction. Bypassed / invalid pixels deliberately have no score.
    gap_flat = np.full(X_flat.shape[0], np.nan, dtype=np.float32)
    favorite_flat = np.full(X_flat.shape[0], -1, dtype=np.int32)
    runner_flat = np.full(X_flat.shape[0], -1, dtype=np.int32)
    gap_flat[inference_mask] = gap_valid
    favorite_flat[inference_mask] = favorite_valid
    runner_flat[inference_mask] = runner_valid
    np.savez_compressed(
        out_dir / "decision_scores.npz", version=np.array(1),
        gap=gap_flat.reshape(H, W), favorite=favorite_flat.reshape(H, W),
        runner=runner_flat.reshape(H, W), class_names=np.asarray(names, dtype=str),
        svm_type=np.array(svm_type), model_path=np.array(str(model_path)),
        abundance_sha256=np.array(file_digest(abund_path.with_suffix(".raw"))),
    )

    # ------------------------------------------------------------
    # Visual mixture / overlay
    # ------------------------------------------------------------

    pal = make_palette(names).astype(np.float32)

    mix_bgr = np.tensordot(abund, pal, axes=([2], [0]))
    mix_bgr = np.clip(mix_bgr, 0, 255).astype(np.uint8)

    H_base, W_base = base_bgr.shape[:2]

    if mix_bgr.shape[:2] != (H_base, W_base):
        mix_bgr_rs = cv2.resize(
            mix_bgr,
            (W_base, H_base),
            interpolation=cv2.INTER_NEAREST
        )
    else:
        mix_bgr_rs = mix_bgr

    sep = np.full((H_base, 8, 3), 30, np.uint8)
    side_by_side = np.hstack([base_bgr, sep, mix_bgr_rs])

    mixture_side_by_side_path = out_dir / "mixture_side_by_side.png"
    write_image(str(mixture_side_by_side_path), side_by_side)

    oa = float(overlay_alpha)
    oa = min(1.0, max(0.0, oa))

    overlay = cv2.addWeighted(base_bgr, 1.0 - oa, mix_bgr_rs, oa, 0.0)
    assigned_mask = (abund.max(axis=2) > 0).astype(np.uint8)
    if assigned_mask.shape != base_bgr.shape[:2]:
        assigned_mask = cv2.resize(assigned_mask, (W_base, H_base), interpolation=cv2.INTER_NEAREST)
    overlay[assigned_mask == 0] = base_bgr[assigned_mask == 0]

    overlay_path = out_dir / "overlay_mixture.png"
    write_image(str(overlay_path), overlay)

    # ------------------------------------------------------------
    # Hard labels and masks
    # ------------------------------------------------------------

    hard_labels = None

    if make_hard_labels:
        hard_idx = np.argmax(abund, axis=2).astype(np.int32)
        hard_val = np.max(abund, axis=2).astype(np.float32)

        default_thr = float(hard_min_abundance)
        win_thr = np.full((H, W), default_thr, np.float32)

        for i, cname in enumerate(names):
            key = (cname or "").lower().strip()
            win_thr[hard_idx == i] = float(CLASS_MIN_ABUNDANCE.get(key, default_thr))

        unknown = (hard_val <= 0) | (hard_val < win_thr)
        hard_idx[unknown] = UNKNOWN_LABEL

        hard_labels = hard_idx

        for i, cname in enumerate(names):
            key = (cname or "").lower().strip()
            thr_i = float(CLASS_MIN_ABUNDANCE.get(key, default_thr))
            m = ((hard_labels == i) & (abund[:, :, i] >= thr_i)).astype(np.uint8) * 255
            safe_name = str(cname).replace("/", "_").replace("\\", "_")
            write_image(str(out_dir / f"mask_{safe_name}.png"), m)

    # ------------------------------------------------------------
    # Spectra export
    # ------------------------------------------------------------

    lib = {}

    try:
        lib = load_library(lib_path)
    except Exception:
        lib = {}

    def _get_med_std_from_entry(entry, n_bands_expected: int):
        if isinstance(entry, dict):
            if "median_snv" in entry:
                m = np.asarray(entry["median_snv"], np.float32)
                s = np.asarray(entry.get("std_snv", np.zeros_like(m)), np.float32)

            elif "median" in entry:
                m = np.asarray(entry["median"], np.float32)
                s = np.asarray(entry.get("std", np.zeros_like(m)), np.float32)

            elif "mean" in entry:
                m = np.asarray(entry["mean"], np.float32)
                s = np.asarray(entry.get("std", np.zeros_like(m)), np.float32)

            elif "spectra" in entry:
                S = np.asarray(entry["spectra"], np.float32)
                if S.ndim == 2 and S.shape[1] > 0:
                    m = np.nanmedian(S, axis=0).astype(np.float32)
                    s = np.nanstd(S, axis=0).astype(np.float32)
                else:
                    m = np.zeros((n_bands_expected,), np.float32)
                    s = np.zeros_like(m)

            else:
                m = np.zeros((n_bands_expected,), np.float32)
                s = np.zeros_like(m)

        else:
            v = np.asarray(entry, np.float32).ravel()
            m = v
            s = np.zeros_like(m, np.float32)

        return m, s

    lib_median = []
    lib_stdev = []

    for cname in names:
        entry = lib.get(cname, None)

        library_mode = ((lib.get('_meta') or {}).get('normalization')
                        or (entry.get('normalization') if isinstance(entry, dict) else None))
        if library_mode != normalization_mode:
            # The library is a preview, not SVM input. Derive consistent reference
            # spectra from raw ROIs in memory instead of rewriting it/retraining.
            spectra = X_ds[y_ds == cname][:, dataset_band_indices]
            spectra = normalize_rows(spectra, normalization_mode)
            spectra = spectra[np.isfinite(spectra).all(axis=1)]
            lib_median.append(np.median(spectra, axis=0).astype(np.float32)
                              if len(spectra) else np.full(Bsel, np.nan, np.float32))
            lib_stdev.append(np.std(spectra, axis=0).astype(np.float32)
                             if len(spectra) else np.full(Bsel, np.nan, np.float32))
            continue

        if entry is None:
            m = np.zeros((N,), np.float32)
            s = np.zeros((N,), np.float32)
        else:
            m, s = _get_med_std_from_entry(entry, N)

        m = np.asarray(m, np.float32).ravel()
        s = np.asarray(s, np.float32).ravel()

        if m.size >= int(train_band_indices.max()) + 1:
            m = m[train_band_indices]
        else:
            m = np.zeros((Bsel,), np.float32)

        if s.size >= int(train_band_indices.max()) + 1:
            s = s[train_band_indices]
        else:
            s = np.zeros((Bsel,), np.float32)

        m = normalize_rows(m[None, :], mode=normalization_mode).ravel()

        lib_median.append(m.astype(np.float32))
        lib_stdev.append(s.astype(np.float32))

    lib_median_arr = (
        np.stack(lib_median, axis=0).astype(np.float32)
        if lib_median else np.zeros((0, Bsel), np.float32)
    )

    lib_stdev_arr = (
        np.stack(lib_stdev, axis=0).astype(np.float32)
        if lib_stdev else np.zeros_like(lib_median_arr, np.float32)
    )

    # The classification may optionally run on a temporarily shading-corrected
    # cube while the GUI spectra remain tied to the original radiometric data.
    # This keeps the displayed/scientific intensities untouched by the
    # experimental shading correction.
    if spectra_source_cube is None:
        Xsel_for_mean = normalize_rows(X_flat, mode=normalization_mode)
    else:
        spectra_cube = np.asarray(spectra_source_cube, dtype=np.float32)
        if spectra_cube.shape != cube.shape:
            raise ValueError(
                f"spectra_source_cube shape mismatch: {spectra_cube.shape} != {cube.shape}"
            )
        spectra_sel = spectra_cube[:, :, band_indices].astype(np.float32, copy=False)
        Xsel_for_mean = normalize_rows(
            spectra_sel.reshape(-1, Bsel), mode=normalization_mode
        )

    labels_flat = (
        hard_labels.reshape(-1)
        if hard_labels is not None
        else np.argmax(abund, axis=2).reshape(-1)
    )

    img_mean = []

    for i, cname in enumerate(names):
        idx = np.where(labels_flat == i)[0]

        if idx.size > 0:
            img_mean.append(np.nanmean(Xsel_for_mean[idx], axis=0).astype(np.float32))
        else:
            img_mean.append(np.full((Bsel,), np.nan, np.float32))

    img_mean_arr = (
        np.stack(img_mean, axis=0).astype(np.float32)
        if img_mean else np.zeros((0, Bsel), np.float32)
    )

    bands_arr = np.asarray(band_indices, np.int32)
    dataset_bands_arr = np.asarray(train_band_indices, np.int32)
    wv = np.asarray(selected_wavelengths, dtype=np.float32)

    class_colors_bgr = pal.astype(np.uint8)

    spectra_npz_path = out_dir / "spectra_data.npz"

    np.savez(
        str(spectra_npz_path),
        wavelengths=wv,
        classes=np.array(names, dtype=object),
        class_names=np.array(names, dtype=object),
        bands_idx=bands_arr,
        bands=bands_arr,
        dataset_bands_idx=dataset_bands_arr,
        lib_median=lib_median_arr,
        lib_stdev=lib_stdev_arr,
        img_mean=img_mean_arr,
        class_colors_bgr=class_colors_bgr,
    )

    # ------------------------------------------------------------
    # Endmember names
    # ------------------------------------------------------------

    try:
        (out_dir / "endmember_names.json").write_text(
            json.dumps(names, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
    except Exception as e:
        flush_print(f"[WARN] Could not save endmember_names.json: {e}")

    flush_print("[SVM] Done.")
    flush_print(f"[SVM] Overlay: {overlay_path}")

    return {
        "names": names,
        "abundances_path": str(abund_path),
        "overlay_path": str(overlay_path),
        "hard_labels_path": None,
        "hard_labels_color_path": None,
        "abundance_map_paths": [],
        "abundance_combined_path": None,
        "abundances": abund,
        "hard_labels": hard_labels,
        "mixture_color_path": None,
        "base_image_path": str(base_image_path) if base_image_path is not None else None,
        "mixture_side_by_side_path": str(mixture_side_by_side_path),
        "spectra_data_path": str(spectra_npz_path),
        "endmember_names": names,
    }


# ============================================================
# COMMAND-LINE TRAINING ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train an SVM model associated with one ROI dataset."
    )
    parser.add_argument("--reference_model", help="Clone all pipeline settings from this saved model for retraining.")
    parser.add_argument("--train-only", action="store_true", help="Train the model and exit.")
    parser.add_argument("--dataset_path", default=str(DEFAULT_NPZ_DATASET))
    parser.add_argument("--model_path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--normalization_mode", default="none")
    parser.add_argument("--svm_type", choices=sorted(SUPPORTED_SVM_TYPES), default=SVM_TYPE)
    parser.add_argument("--force_retrain", action="store_true")
    parser.add_argument("--platt", action=argparse.BooleanOptionalAction, default=None, help="Require cube-held-out sigmoid calibration; otherwise save a raw SVM.")
    parser.add_argument("--optimize_all_kernels", action="store_true")
    parser.add_argument("--keep_related_models", action="store_true")
    parser.add_argument("--refresh_validation", action="store_true")
    parser.add_argument(
        "--band_weight", action="append", default=[], metavar="NM=WEIGHT",
        help="Manual wavelength weight, e.g. --band_weight 620=1.5. Repeat as needed.",
    )
    parser.add_argument(
        "--included_cube_id", action="append", default=None,
        help="Cube ID to include in training; repeat to select multiple cubes. Default: all cubes.",
    )
    parser.add_argument(
        "--selected_wavelengths_nm",
        nargs="*",
        type=float,
        default=None,
        help="Optional wavelength subset. By default all dataset bands are used.",
    )
    parser.add_argument("--no_band_weights", action="store_true", help="Legacy option; PCA benchmark already disables band weights")
    args = parser.parse_args()

    if not args.train_only:
        parser.error("This command-line entry point currently supports --train-only.")

    try:
        band_weights = {
            int(float(item.split("=", 1)[0])): float(item.split("=", 1)[1])
            for item in args.band_weight
        }
    except (IndexError, ValueError):
        parser.error("--band_weight must use NM=WEIGHT, for example 620=1.5")

    reference = None
    if args.reference_model:
        reference = joblib.load(args.reference_model)
        if args.platt is None:
            args.platt = bool(reference.get('platt_calibration'))
        _REFERENCE_PIPELINE = reference['model']
        estimator = _REFERENCE_PIPELINE.named_steps['svm']
        for field, setting in [('scaler_mode','SCALER_MODE'), ('use_pca','USE_PCA'),
                               ('pca_variance','PCA_VARIANCE')]:
            if field in reference:
                globals()[setting] = reference[field]
        LINEAR_C = float(reference.get('linear_c', LINEAR_C))
        RBF_C = float(reference.get('rbf_c', RBF_C))
        RBF_GAMMA = reference.get('rbf_gamma', RBF_GAMMA)
        MAX_ITER = int(getattr(estimator, 'max_iter', MAX_ITER))
        SVM_TOL = float(getattr(estimator, 'tol', SVM_TOL))
        RANDOM_STATE = getattr(estimator, 'random_state', RANDOM_STATE)
        if 'pca' in _REFERENCE_PIPELINE.named_steps:
            PCA_SVD_SOLVER = _REFERENCE_PIPELINE.named_steps['pca'].svd_solver
        band_weights = reference.get('band_weights_by_wavelength', {}) if reference.get('use_band_weights') else {}
        args.normalization_mode = reference.get('normalization_mode', 'none')
        args.svm_type = reference.get('svm_type', 'linear')
        if args.selected_wavelengths_nm is None:
            args.selected_wavelengths_nm = np.asarray(reference['selected_wavelengths'], dtype=float).tolist()

    if args.selected_wavelengths_nm is not None:
        flush_print(f"[SVM][TRAIN] Requested bands ({len(args.selected_wavelengths_nm)}): "
                    + ', '.join(f'{value:g}' for value in args.selected_wavelengths_nm) + ' nm')

    trained = train_svm_from_dataset(
        dataset_path=args.dataset_path,
        model_path=args.model_path,
        normalization_mode=args.normalization_mode,
        svm_type=args.svm_type,
        force_retrain=True,
        selected_wavelengths_nm=args.selected_wavelengths_nm,
        use_band_weights=bool(band_weights),
        band_weights_by_wavelength=band_weights,
        default_band_weight=DEFAULT_BAND_WEIGHT,
        invalidate_related_models=not args.keep_related_models,
        optimize_all_kernels=args.optimize_all_kernels,
        refresh_validation=args.refresh_validation,
        included_cube_ids=args.included_cube_id,
        require_platt=bool(args.platt),
    )

    if args.reference_model:
        if np.array_equal(np.asarray(trained['selected_wavelengths'], dtype=float),
                          np.asarray(reference['selected_wavelengths'], dtype=float)):
            _atomic_joblib_dump(trained, Path(args.reference_model))
            flush_print(f"[SVM][TRAIN] Updated selected variant: {args.reference_model}")
        else:
            flush_print(f"[SVM][TRAIN] New band configuration saved separately; reference preserved: {args.reference_model}")
