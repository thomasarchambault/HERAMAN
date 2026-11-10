# Rôle : Calculer les écarts de scores de décision SVM et les empreintes des fichiers associés.

"""Raw decision-score gaps; these are not calibrated probabilities."""
import hashlib
import numpy as np


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def decision_gap(scores, n_classes):
    """Top-two class-score gap and score-ranked favorite/runner-up.

    Binary sklearn decisions have a single signed score. Represent them as
    [-s, s], so the top-two gap is 2*abs(s). SVC multiclass OVR scores are
    vote-derived scores, not geometric distances. Predict ties may differ.
    """
    values = np.asarray(scores, dtype=float)
    if values.ndim == 1:
        if n_classes != 2:
            raise ValueError('One decision score requires a binary classifier.')
        values = np.column_stack([-values, values])
    if values.ndim != 2 or values.shape[1] != n_classes or n_classes < 2:
        raise ValueError('Expected one score per class.')
    order = np.argsort(values, axis=1, kind='stable')
    favorite, runner = order[:, -1], order[:, -2]
    rows = np.arange(len(values))
    gap = values[rows, favorite] - values[rows, runner]
    valid = np.isfinite(values).all(axis=1)
    return (np.where(valid, gap, np.nan).astype(np.float32),
            np.where(valid, favorite, -1).astype(np.int32),
            np.where(valid, runner, -1).astype(np.int32))


def ambiguity_mask(gap, threshold):
    values = np.asarray(gap)
    return np.isfinite(values) & (values <= float(threshold))
