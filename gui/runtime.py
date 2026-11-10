# Rôle : Charger la configuration de l’interpréteur et vérifier l’architecture Python du GUI.

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from shared.paths import CONFIG_DIR, PROJECT_ROOT


@dataclass(frozen=True)
class RuntimeSettings:
    python64: Path
    expected_gui_bits: int = 64


def current_python_bits() -> int:
    return struct.calcsize("P") * 8


def _project_path(value: str | Path) -> Path:
    path = Path(str(value).strip()).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _checked_python64(path: Path) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Python interpreter not found: {path}")
    if path == Path(sys.executable).resolve():
        bits = current_python_bits()
    else:
        result = subprocess.run(
            [str(path), "-I", "-c", "import struct; print(struct.calcsize('P') * 8)"],
            capture_output=True, text=True, timeout=10, check=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        bits = int(result.stdout.strip())
    if bits != 64:
        raise RuntimeError(f"HERAMAN workers require Python 64-bit; {path} is {bits}-bit.")
    return path


def load_runtime_settings() -> RuntimeSettings:
    path = CONFIG_DIR / "runtime_64.json"
    data = json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else {}
    override = os.environ.get("MSI_PYTHON64", "").strip()
    configured = data.get("python64")
    if override:
        # Explicit environment overrides must not silently launch a different Python.
        python = _checked_python64(_project_path(override))
    elif configured and _project_path(configured).is_file():
        python = _checked_python64(_project_path(configured))
    else:
        # The GUI already runs with its installed dependencies. Reuse that exact
        # environment, including when a copied configuration references a missing venv.
        active = Path(sys.executable)
        if active.name.lower() == "pythonw.exe" and active.with_name("python.exe").is_file():
            active = active.with_name("python.exe")
        python = _checked_python64(active)
    return RuntimeSettings(python64=python, expected_gui_bits=int(data.get("expected_gui_bits", 64)))


def require_gui_64bit(settings: RuntimeSettings) -> None:
    actual = current_python_bits()
    if actual != settings.expected_gui_bits:
        raise RuntimeError(
            f"This GUI must run with Python {settings.expected_gui_bits}-bit; "
            f"the current interpreter is {actual}-bit."
        )


def configure_qt_worker(process):
    """Match QProcess output to the GUI's explicit UTF-8 decoder."""
    from PyQt5.QtCore import QProcessEnvironment
    from shared.process_io import worker_environment
    environment = QProcessEnvironment()
    for name, value in worker_environment().items():
        environment.insert(name, value)
    process.setProcessEnvironment(environment)
