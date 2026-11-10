# Rôle : Démarrer QApplication, valider Python 64 bits, appliquer le thème et ouvrir le GUI.

from __future__ import annotations

import os
import sys

from PyQt5.QtWidgets import QApplication, QMessageBox

from gui.runtime import load_runtime_settings, require_gui_64bit
from shared.paths import PROJECT_ROOT
from shared.process_io import configure_utf8_stdio


def load_stylesheet() -> str:
    style_path = PROJECT_ROOT / "styles" / "dark.qss"
    return style_path.read_text(encoding="utf-8") if style_path.is_file() else ""


def main() -> int:
    configure_utf8_stdio()
    os.chdir(PROJECT_ROOT)
    app = QApplication(sys.argv)

    try:
        settings = load_runtime_settings()
        require_gui_64bit(settings)
    except Exception as exc:
        QMessageBox.critical(None, "Python incompatible", str(exc))
        return 2

    # HERAMAN performs analysis only; no hardware drivers are imported.
    from gui.main_window import MainGUI

    app.setStyleSheet(load_stylesheet())
    gui = MainGUI()
    gui.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
