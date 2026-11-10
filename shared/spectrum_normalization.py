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
