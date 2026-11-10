"""One linear probability colour scale shared by the GUI and standalone plots."""
import numpy as np
from matplotlib import colormaps

FAT_PROBABILITY_CMAP = 'turbo'

# Categorical overlay: non-fat, fat, unsure. Separate from the P(fat) heatmap.
FAT_CLASS_RGB = np.array([[215, 40, 40], [25, 185, 65], [245, 205, 25]], dtype=np.uint8)
FAT_CLASS_LEGEND = 'Platt: green = fat · red = non-fat · yellow = unsure.'


def probability_rgb(probabilities):
    p = np.clip(np.nan_to_num(np.asarray(probabilities, dtype=float)), 0, 1)
    return (255*colormaps[FAT_PROBABILITY_CMAP](p)[..., :3]).astype(np.uint8)
