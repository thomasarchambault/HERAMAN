"""Per-spectrum normalization shared by SVM features and inspection plots."""
import numpy as np


def normalize_spectra(values, mode='none'):
    X = np.asarray(values, dtype=np.float32)
    if mode == 'none':
        return X
    if mode == 'log':
        # Corrected MARGIN reflectance is stored in percent. A fixed floor
        # keeps dark/negative corrected values finite at training and inference.
        return -np.log10(np.maximum(X / np.float32(100.), np.float32(1e-6)))
    if mode == 'minmax':
        low = np.nanmin(X,axis=1,keepdims=True)
        span = np.nanmax(X,axis=1,keepdims=True)-low
        return (X-low)/np.where(span>1e-12,span,1.)
    if mode in ('snv','snv_no_ratio','snv_three_ratios'):
        centered = X-np.nanmean(X,axis=1,keepdims=True)
        sd = np.nanstd(centered,axis=1,keepdims=True)
        return centered/np.where(sd>1e-12,sd,1.)
    raise ValueError(f'Unsupported inspection normalization: {mode}')


def normalize_display_block(block, mode):
    """Apply the transform across wavelengths for each pixel."""
    if mode == 'log_reflectance':
        result = np.full(block.shape, np.nan, dtype=np.float64)
        np.log10(block, out=result, where=(block > 0) & np.isfinite(block))
        return -result
    return normalize_spectra(block.reshape(-1, block.shape[-1]), mode).reshape(block.shape)


def log_reflectance_plot_limits(cube):
    """Estimate a common display range from spatially sampled full spectra."""
    h, w, bands = cube.shape
    count = min(h*w, max(1, 500_000 // bands))
    pixels = np.linspace(0, h*w-1, count, dtype=np.intp)
    spectra = cube[pixels//w, pixels%w, :]
    values = normalize_display_block(spectra, 'log_reflectance')
    values = values[np.isfinite(values)]
    if not values.size:
        return 0., 1.
    low, high = np.percentile(values, [1, 99])
    return min(0., low), max(0., high)
