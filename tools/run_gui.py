"""Launch HERAMAN using the same interpreter selection as its workers."""
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from gui.runtime import load_runtime_settings


def configured_python64() -> Path:
    return load_runtime_settings().python64


def main() -> int:
    os.chdir(PROJECT_ROOT)
    python = configured_python64()
    if Path(sys.executable).resolve() == python:
        from gui.main import main as run_application
        return int(run_application())
    return int(subprocess.call([str(python), "-m", "gui.main"], cwd=str(PROJECT_ROOT)))


if __name__ == "__main__":
    raise SystemExit(main())
