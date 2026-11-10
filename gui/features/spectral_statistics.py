# Rôle : Calculer les statistiques descriptives et tests non paramétriques des spectres ROI.

"""Notebook-derived descriptive spectral statistics, independent of classifier weights.

Kruskal H with rank eta_H²=(H-k+1)/(N-k); Dunn with pooled ties;
BH correction across ALL band/pair tests, not separately after band selection.
P-values are nominal: ROI medians can still be clustered within specimens.
"""
from itertools import combinations
import numpy as np
from scipy.stats import kruskal, rankdata, norm, mannwhitneyu
from sklearn.feature_selection import mutual_info_classif
from gui.features.svm_visualization import load_model_dataset


def bh_adjust(p):
    p = np.asarray(p, dtype=float)
    output = np.full(p.shape, np.nan)
    flat = p.ravel()
    ids = np.flatnonzero(np.isfinite(flat))
    order = ids[np.argsort(flat[ids], kind='stable')]
    if len(order):
        corrected = np.minimum.accumulate((flat[order]*len(order)/np.arange(1, len(order)+1))[::-1])[::-1]
        output.ravel()[order] = np.minimum(corrected, 1)
    return output


def prepare_observations(payload, path, mode='roi', exclude_artificial=True):
    x, y = load_model_dataset(payload, path)
    valid = np.isfinite(x).all(axis=1) & np.isin(y, np.asarray(payload['class_names']).astype(str))
    if exclude_artificial:
        valid &= ~np.isin(np.char.lower(y), ['background', 'petri', 'black ink', 'yellow ink'])
    with np.load(path, allow_pickle=True) as data:
        key = next((k for k in ['sample_cube_index', 'cube_idx', 'sample_id'] if k in data.files), None)
        cubes = np.asarray(data[key]) if key else None
        if cubes is not None and cubes.shape != y.shape:
            raise ValueError('Source identifiers must have one entry per pixel.')
        if mode == 'roi':
            if 'roi_id' not in data.files or cubes is None:
                raise ValueError('ROI medians require roi_id AND source identifiers. Regenerate the dataset with ROI provenance.')
            roi = np.asarray(data['roi_id'])
            if roi.shape != y.shape:
                raise ValueError('ROI identifiers must have one entry per pixel.')
            groups = {}
            for i in np.flatnonzero(valid):
                groups.setdefault((str(cubes[i]), str(y[i]), str(roi[i])), []).append(i)
            rows = [(np.median(x[ids], axis=0), key[1]) for key, ids in groups.items()]
            if not rows:
                raise ValueError('No finite ROI observations for the chosen classes.')
            values, labels = zip(*rows)
            x_used, y_used = np.asarray(values), np.asarray(labels)
        elif mode == 'pixels':
            x_used, y_used = x[valid], y[valid]
        else:
            raise ValueError('Unknown analysis unit.')
    # Fixed balanced sample, shared by every band; never average p-values.
    names, counts = np.unique(y_used, return_counts=True)
    if len(names) < 2 or counts.min() < 2:
        raise ValueError('At least two classes with two observations each are required. Try including non-tissue classes.')
    n = min(int(counts.min()), 1000)
    rng = np.random.default_rng(42)
    ids = np.concatenate([rng.choice(np.flatnonzero(y_used == name), n, replace=False) for name in names])
    return x_used[ids], y_used[ids], dict(mode=mode, per_class=n, available=counts,
        sources=len(np.unique(cubes[valid])) if cubes is not None else None,
        excluded=int((~valid).sum()))


def analyze_spectra(payload, path, mode='roi', exclude_artificial=True):
    x, y, metadata = prepare_observations(payload, path, mode, exclude_artificial)
    names = np.unique(y)
    pairs = list(combinations(range(len(names)), 2))
    n, bands = x.shape
    k = len(names)
    masks = [y == name for name in names]
    h, p, eta = np.zeros(bands), np.ones(bands), np.zeros(bands)
    dunn = np.ones((len(pairs), bands))
    delta = np.zeros_like(dunn)
    for band in range(bands):
        values = x[:, band]
        if np.all(values == values[0]):
            continue
        groups = [values[m] for m in masks]
        h[band], p[band] = kruskal(*groups)
        eta[band] = np.clip((h[band]-k+1)/(n-k), 0, 1)
        ranks = rankdata(values)
        _, ties = np.unique(values, return_counts=True)
        ties = ties.astype(float)
        variance = n*(n+1)/12 - np.sum(ties**3-ties)/(12*(n-1))
        for row, (a, b) in enumerate(pairs):
            se = np.sqrt(variance*(1/len(groups[a])+1/len(groups[b])))
            z = (ranks[masks[a]].mean()-ranks[masks[b]].mean())/se
            dunn[row, band] = 2*norm.sf(abs(z))
            u = mannwhitneyu(groups[a], groups[b], method='asymptotic').statistic
            delta[row, band] = 2*u/(len(groups[a])*len(groups[b]))-1
    # Spearman correlation with explicit constant-band handling.
    ranks = np.apply_along_axis(rankdata, 0, x)
    centered = ranks-ranks.mean(axis=0)
    lengths = np.linalg.norm(centered, axis=0)
    denominator = np.outer(lengths, lengths)
    corr = np.divide(centered.T@centered, denominator, out=np.zeros((bands,bands)), where=denominator>0)
    corr = np.clip(corr, -1, 1)
    mi = mutual_info_classif(x, y, random_state=42, n_neighbors=min(3,metadata['per_class']-1))
    mi[lengths == 0] = 0
    relevance = mi/mi.max() if mi.max()>0 else np.zeros(bands)
    selected = []
    for _ in range(min(8, bands)):
        candidates = [j for j in range(bands) if j not in selected and lengths[j]>0 and mi[j]>0]
        if not candidates:
            break
        best = max(candidates, key=lambda j: relevance[j]-(np.abs(corr[j,selected]).mean() if selected else 0))
        selected.append(best)
    # Undefined correlations must not look like evidence of independence.
    corr[denominator == 0] = np.nan
    # The chi-square/normal approximations are unsuitable for tiny groups.
    if metadata['per_class'] < 5:
        p[:] = np.nan
        dunn[:] = np.nan
    return dict(eta=eta, h=h, kw_q=bh_adjust(p), dunn_q=bh_adjust(dunn), delta=delta,
                names=names, pairs=[f'{names[a]} / {names[b]}' for a,b in pairs],
                wavelengths=np.asarray(payload['selected_wavelengths']), corr=corr, mi=mi,
                selected=selected, metadata=metadata)
