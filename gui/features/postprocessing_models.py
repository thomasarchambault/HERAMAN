"""Live model availability and explicit confirmation before costly RBF fitting."""
from __future__ import annotations

import json
from pathlib import Path

from PyQt5.QtCore import QProcess, QTimer
from PyQt5.QtWidgets import QLabel, QMessageBox, QSizePolicy

from shared.paths import PROJECT_ROOT


class PostprocessingModelsMixin:
    def _init_pp_model_status(self, layout):
        self.pp_model_status = QLabel("Checking model compatibility…")
        self.pp_model_status.setWordWrap(True)
        self.pp_model_status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        layout.addWidget(self.pp_model_status)
        self._pp_model_probe = None
        self._pp_model_result = None
        self._pp_model_launch_pending = False
        self._pp_model_timer = QTimer(self)
        self._pp_model_timer.setSingleShot(True)
        self._pp_model_timer.timeout.connect(self._check_pp_model_status)
        for combo in (self.combo_pp_model, self.combo_pp_normalization,
                      self.combo_pp_classification, self.combo_pp_source):
            combo.currentIndexChanged.connect(self._queue_pp_model_check)
        self.cb_pp_tissue_only.toggled.connect(self._queue_pp_model_check)
        self.combo_pp_normalization.currentIndexChanged.connect(self._on_pp_normalization_changed)
        self._queue_pp_model_check()

    def _queue_pp_model_check(self, *_args):
        if not hasattr(self, "_pp_model_timer"):
            return
        self._pp_model_result = None
        self._pp_model_launch_pending = False
        self.btn_run_pp.setEnabled(True)
        self._cancel_pp_model_probe()
        self._set_pp_availability("checking", "Checking model compatibility…")
        self._pp_model_timer.start(400)

    def _cancel_pp_model_probe(self):
        previous = self._pp_model_probe
        self._pp_model_probe = None
        if previous is not None and previous.state() != QProcess.NotRunning:
            previous.kill()  # Asynchronous; a stale result cannot update the panel.

    def _pp_model_selection(self):
        paths = self._get_selected_postprocessing_profile_paths()
        bands = None
        profile = paths["model"].parent / f"{paths['profile']}_svm_bands.json"
        if profile.is_file():
            bands = [float(value) for value in json.loads(profile.read_text(encoding="utf-8")).get("selected_wavelengths_nm", [])]
        result = {key: str(paths[key]) for key in ("dataset", "library", "model")}
        result.update(source=str(self.combo_pp_source.currentData()),
                      kernel=str(self.combo_pp_classification.currentData() or "linear"),
                      normalization=str(self.combo_pp_normalization.currentData() or "none"),
                      tissue_only=self.cb_pp_tissue_only.isChecked(), bands=bands,
                      classes=[])
        return result

    def _set_pp_availability(self, state, text, tooltip=""):
        colors = {"ready": "#62d790", "training": "#ffc15e", "checking": "#a9cbe9", "error": "#ff9696"}
        self.pp_model_status.setText(text)
        self.pp_model_status.setStyleSheet(f"QLabel {{ color: {colors[state]}; padding: 7px; border: 1px solid {colors[state]}; border-radius: 4px; }}")
        self.pp_model_status.setToolTip(tooltip or text)

    def _check_pp_model_status(self, launch=False, bands_override=None):
        self._pp_model_timer.stop()
        self._cancel_pp_model_probe()
        self._pp_model_launch_pending = bool(launch)
        self.btn_run_pp.setEnabled(not launch)
        try:
            base_selection = self._pp_model_selection()
            selection = dict(base_selection)
            if bands_override is not None:
                selection['bands'] = list(bands_override)
            python = self._get_python64_path()
            if not Path(python).is_file():
                raise ValueError("64-bit Python interpreter not found.")
        except Exception as exc:
            self._pp_model_check_failed(str(exc))
            return
        self._set_pp_availability("checking", "Checking model compatibility…")
        process = QProcess(self)
        from gui.runtime import configure_qt_worker
        configure_qt_worker(process)
        self._pp_model_probe = process
        process.setWorkingDirectory(str(PROJECT_ROOT))
        process.setProgram(str(python))
        process.setArguments(["-B", "-m", "processing.model_availability", json.dumps(selection)])
        process.setProcessChannelMode(QProcess.MergedChannels)
        output = bytearray()

        def drain():
            output.extend(bytes(process.readAllStandardOutput()))

        def finished(code, _status):
            drain()
            if self._pp_model_probe is not process:
                process.deleteLater()
                return
            self._pp_model_probe = None
            try:
                lines = output.decode("utf-8", errors="replace").splitlines()
                line = next(line for line in reversed(lines) if line.startswith("MODEL_AVAILABILITY "))
                result = json.loads(line[len("MODEL_AVAILABILITY "):])
                if code != 0:
                    raise ValueError("Model check process failed.")
                if base_selection != self._pp_model_selection():
                    self._queue_pp_model_check()
                else:
                    self._apply_pp_model_status(result)
            except Exception as exc:
                self._pp_model_check_failed(f"Compatibility could not be verified: {exc}")
            process.deleteLater()

        process.readyReadStandardOutput.connect(drain)
        process.finished.connect(finished)

        def failed(error):
            if error == QProcess.FailedToStart and self._pp_model_probe is process:
                self._pp_model_probe = None
                self._pp_model_check_failed(process.errorString())
                process.deleteLater()

        process.errorOccurred.connect(failed)
        process.start()

    def _pp_model_check_failed(self, reason):
        pending = self._pp_model_launch_pending
        self._pp_model_launch_pending = False
        self.btn_run_pp.setEnabled(True)
        self._set_pp_availability("error", "Cannot verify model: " + reason)
        if pending:
            QMessageBox.warning(self, "Model compatibility", reason)

    def _apply_pp_model_status(self, result):
        self._pp_model_result = result
        if result["state"] == "error":
            self._pp_model_check_failed(result["reason"])
            return
        models, missing = result["models"], result["missing"]
        details = "\n".join(f"{item['kernel'].upper()}: {item['path']}" for item in models)
        if missing:
            names = " + ".join(kernel.upper() for kernel in missing)
            text = f"Training required: {names}."
            if models:
                text += " Reuse: " + ", ".join(item["kernel"].upper() for item in models) + "."
            if result.get("reason"):
                text += " " + result["reason"]
            if "rbf" in missing:
                text += " RBF training can take a very long time."
        else:
            text = "Compatible model found — reuse " + " + ".join(item["kernel"].upper() for item in models) + ". No training."
        bands = result.get('bands') or []
        if bands:
            text += f" {len(bands)} bands ({min(bands):g}–{max(bands):g} nm)."
        self._set_pp_availability(result["state"], text, details + "\n" + result.get("reason", ""))
        launch = self._pp_model_launch_pending
        self._pp_model_launch_pending = False
        self.btn_run_pp.setEnabled(True)
        if launch:
            if not self._confirm_pp_cube_bands(result):
                return
            if "rbf" in missing:
                answer = QMessageBox.warning(self, "RBF training required",
                    "No compatible RBF model exists for these selections.\n\n"
                    "Training will use all selected ROI spectra. On a large dataset, RBF training "
                    "can take tens of minutes, several hours or longer, and use substantial memory.\n\n"
                    "Continue and train the missing model(s)?",
                    QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel)
                if answer != QMessageBox.Yes:
                    return
            try:
                self._start_postprocessing_external()
            except Exception as exc:
                message = f"Classification launch failed: {exc}"
                self.pp_process_status.setText(message)
                self.log_output.appendPlainText(message)
                self.btn_run_pp.setEnabled(True)
                QMessageBox.warning(self, "Classification launch", message)

    def _confirm_pp_cube_bands(self, result):
        """Only a launch may propose a different spectral schema; never do it silently."""
        if self.combo_pp_source.currentData() != 'cube' or not result.get('bands'):
            return True
        import numpy as np
        from shared.envi import load_spectral_cube
        try:
            cube, axis = load_spectral_cube(str(self.combo_pp_cube.currentData() or ''))
            if isinstance(cube, np.memmap):
                cube._mmap.close()
        except Exception as exc:
            QMessageBox.warning(self, 'Cube spectral bands', str(exc))
            return False
        common = [float(band) for band in result['bands'] if np.any(np.isclose(axis, band, atol=1., rtol=0.))]
        absent = [float(band) for band in result['bands'] if band not in common]
        if not absent:
            return True
        if not common:
            QMessageBox.warning(self, 'Cube spectral bands', 'No bands common to the cube and the selected SVM configuration.')
            return False
        answer = QMessageBox.question(self, 'Use the common spectral bands?',
            'This cube is missing required model bands (nm): ' + ', '.join(f'{b:g}' for b in absent) +
            '\n\nUse these common bands for this classification (nm): ' + ', '.join(f'{b:g}' for b in common) +
            '?\n\nA compatible model will be reused if available; otherwise a new model will be trained. '
            'The dataset band profile will remain unchanged.',
            QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel)
        if answer == QMessageBox.Yes:
            self._check_pp_model_status(launch=True, bands_override=common)
        return False

    def _run_postprocessing_external(self):
        running = getattr(self, "_proc_pp", None)
        if isinstance(running, QProcess) and running.state() != QProcess.NotRunning:
            QMessageBox.information(self, "Post-processing", "Post-processing is already running.")
            return
        # Fresh read-only check immediately before launch, including filesystem changes.
        self._check_pp_model_status(launch=True)
