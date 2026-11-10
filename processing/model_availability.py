"""Read-only compatibility check, run outside the GUI process."""
from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

from processing.pipeline import svm_classifier as svm


RESULT_PREFIX = "MODEL_AVAILABILITY "


def inspect_selection(selection):
    """Use the actual inference loader in inspection mode: never fit/copy/save."""
    dataset = Path(selection["dataset"])
    family = Path(selection["model"])
    if not dataset.is_file() or not Path(selection["library"]).is_file():
        raise ValueError("Select a project with an existing ROI dataset and library.")
    mode = selection["kernel"]
    kernels = ["linear", "rbf"] if mode == "hybrid" else [mode]
    requested_dataset = dataset
    missing_subset = not dataset.is_file()
    bundle = svm._load_dataset_bundle(requested_dataset if missing_subset else dataset)
    wavelengths = bundle["wavelengths"]
    if wavelengths is None:
        raise ValueError("The ROI dataset needs an explicit spectral axis before model compatibility can be verified independently of a cube.")
    requested = selection.get("bands") or None
    indices, _, matched = svm._match_wavelength_indices(wavelengths, wavelengths, requested,
                                                       tol_nm=svm.WAVELENGTH_MATCH_TOL_NM)
    svm.prepare_svm_features(np.ones((1,len(matched)),np.float32),matched,selection['normalization'])
    if missing_subset:
        return {"state": "training", "models": [], "missing": kernels, "bands": matched.tolist(),
                "reason": "No biological training subset exists for this dataset/class selection."}
    found, missing, reasons = [], [], []
    for kernel in kernels:
        _, _, params, report = svm._resolve_group_aware_training(
            bundle=bundle, dataset_path=dataset, model_family_path=family,
            band_indices=indices, selected_wavelengths=matched,
            normalization_mode=selection["normalization"], svm_type=kernel,
            allow_validation=False)
        canonical = svm._make_wavelength_model_path(family, matched, selection["normalization"], kernel)
        diagnostics = io.StringIO()
        with contextlib.redirect_stdout(diagnostics):
            result = svm._train_or_load_svm_model(
                X=bundle["X"], y_str=bundle["y"], model_path=canonical,
                normalization_mode=selection["normalization"], svm_type=kernel,
                svm_params=params, grouped_training_report=report,
                forced_band_indices=indices, forced_selected_wavelengths=matched,
                dataset_wavelengths=wavelengths, dataset_signature=svm._dataset_signature(dataset),
                inspect_only=True)
        if result:
            found.append({"kernel": kernel, "path": result["path"]})
        else:
            missing.append(kernel)
            warning = next((line for line in diagnostics.getvalue().splitlines() if line.startswith(("[SVM][WARN]", "[SVM][CACHE]"))), "No saved model found for these options.")
            reasons.append(kernel.upper() + ": " + warning.replace("[SVM][WARN] ", "").replace("[SVM][CACHE] ", ""))
    return {"state": "training" if missing else "ready", "models": found, "missing": missing,
            "bands": matched.tolist(), "reason": "\n".join(reasons)}


def main():
    try:
        # The loader's normal diagnostics are not a GUI status protocol.
        with contextlib.redirect_stdout(io.StringIO()):
            result = inspect_selection(json.loads(sys.argv[1]))
    except Exception as exc:
        result = {"state": "error", "models": [], "missing": [], "reason": str(exc)}
    print(RESULT_PREFIX + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
