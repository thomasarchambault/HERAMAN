import unittest
import numpy as np
from types import SimpleNamespace
from shared.tissue_segmentation import segmentation_band_index, segment_cube

class SegmentationBandTests(unittest.TestCase):
    def test_nearest_nonuniform_axis(self):
        axis = np.array([716.93, 722.19, 727.53], dtype=np.float32)
        self.assertEqual(segmentation_band_index(axis, 720), 1)
        self.assertEqual(segmentation_band_index(axis, float(axis[2])), 2)
        self.assertEqual(segmentation_band_index(axis, 400), 0)
        self.assertEqual(segmentation_band_index(axis, 1000), 2)

    def test_tie_uses_first_band(self):
        self.assertEqual(segmentation_band_index([700, 720], 710), 0)

    def test_invalid_axis_or_target(self):
        for axis, target in [([], 720), ([700, np.nan], 720), ([700], np.inf), ([[700]], 720)]:
            with self.subTest(axis=axis, target=target), self.assertRaises(ValueError):
                segmentation_band_index(axis, target)

    def test_mask_uses_nearest_band_values(self):
        cube = np.zeros((5, 5, 3), dtype=np.float32)
        cube[1:4, 1:4, 1] = 2
        args = SimpleNamespace(segmentation_band_nm=720, segmentation_method='threshold',
                               segmentation_threshold=1, tissue_mask_erosion_px=0)
        _, mask, _, _ = segment_cube(cube, [716.93, 722.19, 727.53], args)
        expected = np.zeros((5, 5), dtype=bool)
        expected[1:4, 1:4] = True
        np.testing.assert_array_equal(mask, expected)
