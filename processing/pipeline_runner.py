# Rôle : Exécuter les scripts actifs du pipeline dans un contexte de chemins et d’architecture contrôlé.

from __future__ import annotations

import runpy
import sys
from pathlib import Path

from shared.paths import PROCESSING_PIPELINE_DIR
from processing.runtime_guard import require_64bit
from shared.process_io import configure_utf8_stdio


def run_pipeline_script(*candidate_names: str) -> int:
    configure_utf8_stdio()
    require_64bit()
    pipeline_path = str(PROCESSING_PIPELINE_DIR)
    if pipeline_path not in sys.path:
        sys.path.insert(0, pipeline_path)

    for name in candidate_names:
        target = PROCESSING_PIPELINE_DIR / name
        if target.is_file():
            sys.argv[0] = str(target)
            runpy.run_path(str(target), run_name="__main__")
            return 0

    expected = " ou ".join(str(PROCESSING_PIPELINE_DIR / name) for name in candidate_names)
    print(f"[PROCESSING][ERROR] Script introuvable : {expected}", file=sys.stderr)
    return 2
