"""Platt calibration of held-out SVM scores; no spectral preprocessing here.

Binary scores refer to classes_[1]. New fat-containing multiclass models
calibrate only fat; legacy multiclass OVR calibrators remain readable.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import log_loss

CALIBRATION_VERSION = 3
DEFAULT_FAT_THRESHOLD = .75
DEFAULT_MEAT_THRESHOLD = .25
DEFAULT_FAT_CLEANUP_MAX_PIXELS = 5


def small_fat_islands(mask, max_pixels=DEFAULT_FAT_CLEANUP_MAX_PIXELS):
    """Eight-connected islands, inclusive size limit, in native cube pixels."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or max_pixels <= 0:
        return np.zeros_like(mask)
    from scipy.ndimage import label
    components, _ = label(mask, structure=np.ones((3,3), dtype=np.uint8))
    sizes = np.bincount(components.ravel())
    remove = sizes <= max_pixels
    remove[0] = False
    return remove[components]


def spatial_fat_membership_labels(p, meat_max=DEFAULT_MEAT_THRESHOLD, fat_min=DEFAULT_FAT_THRESHOLD,
                                  max_pixels=DEFAULT_FAT_CLEANUP_MAX_PIXELS):
    labels = fat_membership_labels(p, meat_max, fat_min)
    labels[small_fat_islands(labels == 1, max_pixels)] = 2
    return labels


def fat_membership_labels(p, meat_max=DEFAULT_MEAT_THRESHOLD, fat_min=DEFAULT_FAT_THRESHOLD):
    """-1 excluded, 0 non-fat, 1 fat, 2 unsure; thresholds concern P(fat)."""
    if not (0 <= meat_max < fat_min <= 1):
        raise ValueError("Require 0 <= meat max < fat min <= 1.")
    p = np.asarray(p)
    valid = np.isfinite(p) & (p >= 0) & (p <= 1)
    labels = np.full(p.shape, -1, np.int8)
    labels[valid] = 2
    labels[valid & (p <= meat_max)] = 0
    labels[valid & (p >= fat_min)] = 1
    return labels


def fat_selective_predictions(probabilities, names, unknown=9999):
    """Reject intermediate P(fat), retaining original non-fat class identities."""
    probabilities = np.asarray(probabilities)
    labels = np.argmax(probabilities, axis=-1).astype(np.int32)
    valid = np.isfinite(probabilities).all(axis=-1)
    labels[~valid] = unknown
    unsure = np.zeros(labels.shape, bool)
    names = [str(name).strip().lower() for name in names]
    if 'fat' in names:
        index = names.index('fat')
        decision = fat_membership_labels(probabilities[..., index])
        unsure = (decision == 2) & valid
        labels[unsure | (decision == -1)] = unknown
        labels[(decision == 1) & valid] = index
        other_codes = np.array([i for i in range(len(names)) if i != index])
        if other_codes.size:
            # With >=4 classes, fat may win an argmax/tie even below 25%.
            # The low-P(fat) branch must choose among NON-fat classes only.
            nonfat = (decision == 0) & valid
            best_other = other_codes[np.argmax(probabilities[..., other_codes], axis=-1)]
            labels[nonfat] = best_other[nonfat]
    return labels, unsure


class _Scores(ClassifierMixin, BaseEstimator):
    def __init__(self, n_classes=2):
        self.n_classes = n_classes
        self.classes_ = np.arange(n_classes)
        self.n_features_in_ = 1 if n_classes == 2 else n_classes

    def fit(self, X, y=None):
        raise RuntimeError("Precomputed held-out scores must remain frozen.")

    def decision_function(self, X):
        values = np.asarray(X)
        return values[:, 0] if self.n_classes == 2 else values

    def predict(self, X):
        scores = self.decision_function(X)
        return (scores > 0).astype(int) if self.n_classes == 2 else np.argmax(scores, axis=1)


def fit_platt(scores, labels, n_classes):
    labels = np.asarray(labels, dtype=int)
    if not np.array_equal(np.unique(labels), np.arange(n_classes)):
        raise ValueError("Platt calibration requires held-out scores for every class.")
    values = np.asarray(scores, dtype=float).reshape(len(labels), -1)
    if values.shape[1] != (1 if n_classes == 2 else n_classes) or not np.isfinite(values).all():
        raise ValueError("Invalid held-out SVM decision scores.")
    fitted = CalibratedClassifierCV(
        FrozenEstimator(_Scores(n_classes)), method="sigmoid", ensemble=False,
    ).fit(values, labels)
    sigmoids = fitted.calibrated_classifiers_[0].calibrators
    return dict(version=CALIBRATION_VERSION, method="sigmoid", n_classes=int(n_classes),
                a=[float(s.a_) for s in sigmoids], b=[float(s.b_) for s in sigmoids],
                source="cube-grouped out-of-fold scores", weighted=False)


def platt_probabilities(calibrator, scores):
    if calibrator.get('target') == 'fat':
        values = np.asarray(scores, dtype=float)
        index = int(calibrator['fat_index'])
        p = expit(-(values[:, index] * calibrator['a'][0] + calibrator['b'][0]))
        # Other columns are routing mass, NOT calibrated class probabilities.
        # Keeping all residual mass on the best raw non-fat class preserves
        # its identity and makes argmax equivalent to the binary 0.5 rule.
        other = values.copy()
        other[:, index] = -np.inf
        winner = np.argmax(other, axis=1)
        result = np.zeros_like(values)
        result[np.arange(len(p)), winner] = 1-p
        result[:, index] = p
        return result
    n_classes = int(calibrator["n_classes"])
    values = np.asarray(scores, dtype=float)
    values = values.reshape(-1, 1) if n_classes == 2 else values
    probabilities = expit(-(values * np.asarray(calibrator["a"]) + np.asarray(calibrator["b"])))
    if n_classes == 2:
        return np.column_stack((1 - probabilities[:, 0], probabilities[:, 0]))
    total = probabilities.sum(axis=1, keepdims=True)
    return np.divide(probabilities, total, out=np.full_like(probabilities, 1/n_classes), where=total != 0)


def predict_encoded(payload, features):
    """Fat-only routing uses the 0.5 fat threshold; legacy models use argmax."""
    model, calibrator = payload["model"], payload.get("platt_calibration")
    if calibrator:
        return np.argmax(platt_probabilities(calibrator, calibration_decision_scores(model, features, calibrator)), axis=1)
    return model.predict(features)


def calibration_decision_scores(model, features, calibrator, raw_scores=None):
    """Keep raw non-fat ranking, with the identical fat score used during fitting."""
    scores = np.asarray(model.decision_function(features) if raw_scores is None else raw_scores)
    if calibrator.get('target') == 'fat' and (
            getattr(model.named_steps.get('svm'), 'kernel', None) == 'rbf'
            or calibrator.get('score_kind') == 'fat_vs_best_nonfat_v1'):
        from processing.pipeline.fat_calibration import fat_score
        scores = scores.copy()
        index = int(calibrator['fat_index'])
        scores[:,index] = fat_score(model, features, index, raw_scores=scores)
    return scores


def calibration_metrics(labels, probabilities, names, n_bins=10):
    """Compact JSON-safe reliability bins, including empty-bin counts."""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    curves = []
    for code, name in enumerate(names):
        truth, p = (labels == code).astype(float), probabilities[:, code]
        bins = np.minimum((p*n_bins).astype(int), n_bins-1)
        count = np.bincount(bins, minlength=n_bins)
        sum_p = np.bincount(bins, weights=p, minlength=n_bins)
        sum_y = np.bincount(bins, weights=truth, minlength=n_bins)
        present = count > 0
        curves.append(dict(class_name=str(name), counts=count.tolist(),
            predicted=np.divide(sum_p, count, out=np.zeros(n_bins), where=present)[present].tolist(),
            observed=np.divide(sum_y, count, out=np.zeros(n_bins), where=present)[present].tolist(),
            brier=float(np.mean((p-truth)**2))))
    # Binary convention: MSE for one class. Multiclass: sum of classwise MSEs.
    brier = curves[-1]["brier"] if len(names) == 2 else sum(c["brier"] for c in curves)
    return dict(brier=float(brier), log_loss=float(log_loss(labels, probabilities, labels=np.arange(len(names)))),
                brier_definition="binary MSE" if len(names) == 2 else "multiclass sum of squared errors",
                n_samples=int(len(labels)), bin_edges=np.linspace(0, 1, n_bins+1).tolist(), curves=curves)
