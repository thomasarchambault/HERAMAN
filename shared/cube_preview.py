"""Display-only RGB previews shared by exploration, ROI selection and classification."""
from pathlib import Path
import cv2
import numpy as np

RGB_WAVELENGTHS = (650., 550., 450.)


def stretch_display_band(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    low, high = np.percentile(finite, [2, 98]) if finite.size else (0., 1.)
    scaled = np.clip((values-low)/max(high-low, 1e-12), 0., 1.)
    return (np.nan_to_num(scaled, nan=0., posinf=1., neginf=0.)*255).astype(np.uint8)


def make_rgb_preview(cube, wavelengths):
    """Return native-size uint8 BGR and actual R/G/B wavelengths; never modify spectra."""
    axis = np.asarray(wavelengths, dtype=float)
    if cube.ndim != 3 or axis.shape != (cube.shape[2],) or not np.isfinite(axis).all():
        raise ValueError('An explicit wavelength axis matching the cube is required for RGB.')
    indices = [int(np.argmin(abs(axis-value))) for value in RGB_WAVELENGTHS]
    channels = [stretch_display_band(cube[:, :, index]) for index in indices]
    return np.stack(channels[::-1], axis=-1), axis[indices]


def save_preview_png(path, bgr):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode('.png', bgr)
    if not success:
        raise OSError(f'Could not encode RGB preview: {path}')
    path.write_bytes(encoded.tobytes())
    return path


def save_rgb_from_cube(cube, out_dir, wavelengths, filename='base_image_bgr.png'):
    image, _ = make_rgb_preview(cube, wavelengths)
    return str(save_preview_png(Path(out_dir)/filename, image))
