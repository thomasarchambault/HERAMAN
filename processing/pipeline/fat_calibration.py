"""Multiclass SVM training with a single fat-versus-rest Platt calibrator."""
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneGroupOut
from shared.svm_calibration import fit_platt, calibration_metrics, CALIBRATION_VERSION


def fat_code(names):
    return next((i for i, name in enumerate(names) if str(name).strip().casefold() == 'fat'), None)


def fat_score(model, features, code, raw_scores=None):
    classes = np.asarray(model.classes_)
    positions = np.flatnonzero(classes == code)
    if not len(positions):
        raise ValueError('Training partition contains no fat annotations.')
    scores = np.asarray(model.decision_function(features) if raw_scores is None else raw_scores)
    if scores.ndim == 1:
        return scores if classes[1] == code else -scores
    estimator = model.named_steps.get('svm')
    if getattr(estimator, 'kernel', None) == 'rbf':
        # SVC's OVR vote score scales with the number of competing classes.
        # Mean oriented pairwise margins remain comparable when a rare
        # non-fat class is absent from an inner training partition.
        previous = estimator.decision_function_shape
        try:
            estimator.decision_function_shape = 'ovo'
            pairs = np.asarray(model.decision_function(features))
        finally:
            estimator.decision_function_shape = previous
        margins, column = [], 0
        for first in range(len(classes)):
            for second in range(first+1,len(classes)):
                if classes[first] == code:
                    margins.append(pairs[:,column])
                elif classes[second] == code:
                    margins.append(-pairs[:,column])
                column += 1
        return np.mean(margins,axis=0)
    index = int(positions[0])
    # LinearSVC predicts by comparing OVR scores, not by thresholding fat's
    # score alone. The gap to its strongest rival preserves that decision.
    return scores[:, index] - np.delete(scores, index, axis=1).max(axis=1)


def grouped_scores(engine, X, y, groups, wavelengths, normalization, kernel, params, code):
    if normalization == 'snv_offset':
        raise ValueError('Fat Platt requires per-spectrum preprocessing.')
    if any(str(group).startswith('unknown_cube_') for group in groups):
        raise ValueError('Fat Platt requires known cube provenance.')
    if len(np.unique(groups)) < 2:
        raise ValueError('Fat Platt requires at least two independent training cubes.')
    scores = np.full(len(y), np.nan)
    for train, valid in LeaveOneGroupOut().split(X, y, groups):
        binary = y[train] == code
        if not binary.any() or binary.all():
            raise ValueError('Each SVM training partition must contain fat and non-fat; other individual classes may be absent.')
        model = engine._fit_grouped_fold(X[train],y[train],groups[train],wavelengths,normalization,kernel,params)
        scores[valid] = fat_score(model, engine.prepare_svm_features(X[valid], wavelengths, normalization), code)
    return scores


def fit_fat(scores, y, code, n_classes, kernel='linear'):
    result = fit_platt(scores, (np.asarray(y) == code).astype(int), 2)
    result.update(target='fat', fat_index=int(code), n_classes=int(n_classes),
                  calibrated_classes=['fat'], other_columns='uncalibrated routing mass to best raw non-fat class',
                  score_kind='mean_pairwise_v1' if kernel == 'rbf' else 'fat_vs_best_nonfat_v1')
    return result


def select_fat(engine, X, y, groups, wavelengths, normalization, kernel):
    """Leave one cube out externally and internally; no pixel/cube leakage."""
    params = engine._default_svm_params()
    encoder = LabelEncoder().fit(y)
    names = encoder.classes_.tolist()
    encoded = encoder.transform(y)
    code = fat_code(names)
    probabilities = np.full(len(y),np.nan)
    raw_scores = np.full(len(y),np.nan)
    per_cube, fold_scores = [], []
    raw_confusion = np.zeros((len(names),len(names)),dtype=np.int64)
    error = None
    try:
        if len(np.unique(groups)) < 3:
            raise ValueError('Nested fat Platt validation requires at least three independent cubes.')
        for number,(train,valid) in enumerate(LeaveOneGroupOut().split(X,y,groups),1):
            engine.flush_print(f'[SVM][GROUP-CV] Fat-only Platt fold {number}: held-out cubes: {np.unique(groups[valid]).tolist()}')
            inner = grouped_scores(engine,X[train],encoded[train],groups[train],wavelengths,normalization,kernel,params,code)
            calibrator = fit_platt(inner,(encoded[train]==code).astype(int),2)
            model = engine._fit_grouped_fold(X[train],encoded[train],groups[train],wavelengths,normalization,kernel,params)
            features = engine.prepare_svm_features(X[valid],wavelengths,normalization)
            scores = fat_score(model,features,code)
            raw_scores[valid] = scores
            probabilities[valid] = engine.platt_probabilities(calibrator,scores)[:,1]
            truth = (encoded[valid]==code).astype(int)
            prediction = (probabilities[valid]>=.5).astype(int)
            fold_scores.append(float(f1_score(truth,prediction,average='macro',zero_division=0)))
            raw_confusion += confusion_matrix(encoded[valid],model.predict(features),labels=np.arange(len(names)))
            per_cube.append(dict(cube_id=str(groups[valid][0]),fold=number,macro_f1=fold_scores[-1],
                **calibration_metrics(truth,np.column_stack((1-probabilities[valid],probabilities[valid])),['non-fat','fat'])))
    except ValueError as exc:
        error = str(exc)
    final = None
    try:
        if not np.isfinite(raw_scores).all():
            raw_scores = grouped_scores(engine,X,encoded,groups,wavelengths,normalization,kernel,params,code)
        final = fit_fat(raw_scores,encoded,code,len(names),kernel)
    except ValueError as exc:
        error = error or str(exc)
    evaluated = error is None and np.isfinite(probabilities).all()
    calibration = dict(status='evaluated' if evaluated else 'unavailable',reason=error,
        target='fat',method='sigmoid',version=CALIBRATION_VERSION,roi_only=True,weighted=False,
        protocol='nested leave-one-cube-out; multiclass SVM, fat-only Platt',per_cube=per_cube if evaluated else [])
    report = dict(status='evaluated_fixed_parameters' if evaluated else 'fallback',reason=error,
        n_cubes=len(np.unique(groups)),n_splits=len(np.unique(groups)),fixed_params=params,
        metric='mean per-cube fat/non-fat macro_f1',class_names=['non-fat','fat'],
        prediction_rule='Platt fat probability threshold 0.5',calibration=calibration,final_calibrator=final,
        raw_class_names=names,raw_confusion_matrix=raw_confusion.tolist() if evaluated else None)
    if evaluated:
        truth = (encoded==code).astype(int)
        calibration.update(calibration_metrics(truth,np.column_stack((1-probabilities,probabilities)),['non-fat','fat']))
        calibration['per_cube_mean_brier'] = float(np.mean([r['brier'] for r in per_cube]))
        calibration['per_cube_mean_log_loss'] = float(np.mean([r['log_loss'] for r in per_cube]))
        report.update(best_score=float(np.mean(fold_scores)),
            confusion_matrix=confusion_matrix(truth, probabilities>=.5, labels=[0,1]).tolist())
        calibration['curves'] = [curve for curve in calibration['curves'] if curve['class_name'] == 'fat']
        for cube in per_cube:
            cube['curves'] = [curve for curve in cube['curves'] if curve['class_name'] == 'fat']
    return params,report
