"""OpenCV image IO through Python paths, including accented Windows folders."""
from pathlib import Path
import cv2
import numpy as np


def read_image(path, flags=cv2.IMREAD_COLOR):
    try:
        return cv2.imdecode(np.frombuffer(Path(path).read_bytes(),dtype=np.uint8),flags)
    except (OSError,cv2.error):
        return None


def write_image(path, image):
    try:
        path=Path(path)
        ok,encoded=cv2.imencode(path.suffix,image)
        if not ok: return False
        path.write_bytes(encoded.tobytes())
        return True
    except (OSError,cv2.error):
        return False
