# Rôle : Définir les transformateurs sklearn sérialisables partagés par l’entraînement et le GUI.

"""Pickle-stable sklearn transformers shared by SVM training and the GUI."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# User-facing choices. Legacy transforms remain readable for saved models.
NORMALIZATION_OPTIONS = (
    ("None", "none"),
    ("Min-Max (per spectrum)", "minmax"),
    ("SNV + ratio R620/R580", "snv"),
    ("SNV + 3 ratios (620/580, 560/540, 700/620)", "snv_three_ratios"),
    ("SNV without ratio", "snv_no_ratio"),
    ("−log10(R)", "log"),
)

THREE_SNV_RATIOS = ((620.,580.),(560.,540.),(700.,620.))


def svm_ratio_pairs(mode):
    if mode == 'snv_three_ratios':
        return THREE_SNV_RATIOS
    return () if mode in ('none','snv_no_ratio','log') else ((620.,580.),)


class BandWeighter(BaseEstimator, TransformerMixin):
    """Multiply each post-scaled spectral feature by its fixed weight."""

    def __init__(self, weights):
        self.weights = np.asarray(weights, dtype=np.float32)

    def fit(self, X, y=None):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D. Got: {X.shape}")
        if X.shape[1] != len(self.weights):
            raise ValueError(
                f"X has {X.shape[1]} bands, but weights has {len(self.weights)} values."
            )
        return self

    def transform(self, X):
        return np.asarray(X, dtype=np.float32) * self.weights
