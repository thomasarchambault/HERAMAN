"""Validate HERAMAN's Python environment without changing datasets or models."""
import argparse
import importlib
import importlib.metadata
import os
import struct
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from shared.process_io import configure_utf8_stdio


def main():
    configure_utf8_stdio()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gui',action='store_true')
    args=parser.parse_args()
    if struct.calcsize('P')*8!=64 or not (3,10)<=sys.version_info[:2]<=(3,12):
        raise RuntimeError('HERAMAN requires Python 3.10, 3.11 or 3.12 in 64-bit mode for this environment.')
    modules={'numpy':'numpy','opencv-python':'cv2','scipy':'scipy','scikit-learn':'sklearn',
             'scikit-image':'skimage','matplotlib':'matplotlib','joblib':'joblib',
             'PyQt5':'PyQt5.QtWidgets','pyqtgraph':'pyqtgraph','tifffile':'tifffile'}
    requirements=Path(__file__).with_name('requirements-64bit.txt').read_text(encoding='utf-8-sig')
    expected=dict(line.strip().split('==') for line in requirements.splitlines() if '==' in line and not line.lstrip().startswith('#'))
    for package,module in modules.items():
        importlib.import_module(module)
        version=importlib.metadata.version(package)
        if version!=expected[package]:
            raise RuntimeError(f'{package}: found {version}, expected {expected[package]}. Rerun INSTALL HERAMAN.bat.')
        print(f'OK {package} {version}')
    from sklearn.frozen import FrozenEstimator
    from shared.envi import read_envi, write_envi
    from processing.pipeline import roi_selection_pipeline, svm_classifier, postprocessing_pipeline
    if args.gui:
        os.environ['QT_QPA_PLATFORM']='offscreen'
        from PyQt5.QtWidgets import QApplication,QWidget
        application=QApplication([])
        widget=QWidget();widget.close()
        print('OK Qt initialization (offscreen)')
    print(f'HERAMAN environment ready: {sys.executable}')
    return 0


if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception as exc:
        print(f'ENVIRONMENT ERROR: {exc}',file=sys.stderr)
        raise SystemExit(1)
