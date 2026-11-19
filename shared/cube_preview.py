"""Display-only previews for exploration, ROI selection and classification."""
from pathlib import Path
import cv2
import numpy as np

RGB_WAVELENGTHS = (650., 550., 450.)
NIREOS_RGB_WAVELENGTHS = (737.2, 557., 463.9)


def stretch_display_band(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    low, high = np.percentile(finite, [2, 98]) if finite.size else (0., 1.)
    return scale_display_band(values, low, high)


def scale_display_band(values, low, high):
    values = np.asarray(values, dtype=float)
    scaled = np.clip((values-low)/max(high-low, 1e-12), 0., 1.)
    return (np.nan_to_num(scaled, nan=0., posinf=1., neginf=0.)*255).astype(np.uint8)


def grayscale_limits(cube):
    """Use zero as black and a robust white level shared by all wavelengths."""
    sample = cube[::4, ::4, :]
    finite = sample[np.isfinite(sample)]
    if finite.size == 0:
        return 0., 1.
    # Calibrated against the default HyperLAB view of the reference HERA cube.
    high = float(np.percentile(finite, 99.85))
    if high <= 0.:
        high = float(finite.max())
    return 0., high if high > 0. else 1.


def make_rgb_preview(cube, wavelengths, *, targets=RGB_WAVELENGTHS, limits=None):
    """Return native-size uint8 BGR and actual R/G/B wavelengths; never modify spectra."""
    axis = np.asarray(wavelengths, dtype=float)
    if cube.ndim != 3 or axis.shape != (cube.shape[2],) or not np.isfinite(axis).all():
        raise ValueError('An explicit wavelength axis matching the cube is required for RGB.')
    indices = [int(np.argmin(abs(axis-value))) for value in targets]
    bands = [cube[:, :, index] for index in indices]
    if limits is None:
        channels = [stretch_display_band(band) for band in bands]
    else:
        channels = [scale_display_band(band, *limits) for band in bands]
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
