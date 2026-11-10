# Rôle : Lancer le worker de sélection ROI du pipeline 64 bits.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from processing.pipeline_runner import run_pipeline_script


if __name__ == "__main__":
    raise SystemExit(run_pipeline_script("roi_selection_pipeline.py"))
