"""Regression tests for HERA headers with omitted wavelength units."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from shared.envi import read_envi


class HeraEnviTests(unittest.TestCase):
    def read_fixture(self, wavelengths, units=None, offset=80):
        with tempfile.TemporaryDirectory() as directory:
            header = Path(directory) / 'camera_raw.hdr'
            values = np.arange(12, dtype='<f4').reshape(2, 2, 3)
            binary = header.with_suffix('')
            with binary.open('wb') as stream:
                stream.write(bytes(offset))
                values.transpose(2, 0, 1).tofile(stream)
            text = ('ENVI\nsamples = 2\nlines = 2\nbands = 3\n'
                    'data type = 4\ninterleave = bsq\nbyte order = 0\n'
                    f'header offset = {offset}\nwavelength = {{{wavelengths}}}\n')
            if units is not None:
                text += f'wavelength units = {units}\n'
            header.write_text(text, encoding='utf-8')
            cube, wl, fields = read_envi(binary)
            np.testing.assert_array_equal(cube, values)
            self.assertEqual(header.read_text(encoding='utf-8'), text)
            # Copy so the fixture's mapped file can be removed on Windows.
            result = (wl.copy(), dict(fields))
            del cube
            return result

    def test_hera_missing_units_with_both_offsets(self):
        for offset in (0, 80):
            with self.subTest(offset=offset):
                wl, fields = self.read_fixture('400, 551.23, 1000', offset=offset)
                np.testing.assert_allclose(wl, [400, 551.23, 1000])
                self.assertEqual(fields['wavelength units'], 'Nanometers')
                self.assertIn('heraman wavelength units source', fields)

    def test_explicit_units(self):
        for units, axis in [('nm', '400, 550, 1000'),
                            ('um', '0.4, 0.55, 1.0')]:
            with self.subTest(units=units):
                wl, fields = self.read_fixture(axis, units)
                np.testing.assert_allclose(wl, [400, 550, 1000])
                self.assertNotIn('heraman wavelength units source', fields)

    def test_missing_units_outside_hera_range(self):
        for axis in ('0.4, 0.55, 1.0', '399, 550, 1000', '400, 550, 1001'):
            with self.subTest(axis=axis), self.assertRaisesRegex(ValueError, 'Explicit wavelength units'):
                self.read_fixture(axis)

    def test_unknown_explicit_unit_is_not_overridden(self):
        with self.assertRaisesRegex(ValueError, 'Explicit wavelength units'):
            self.read_fixture('400, 550, 1000', 'unknown')

    def test_invalid_axes_are_rejected(self):
        for axis in ('', '400, 550', '400, 400, 1000', '400, nan, 1000'):
            with self.subTest(axis=axis), self.assertRaisesRegex(ValueError, 'invalid ENVI wavelengths'):
                self.read_fixture(axis)


if __name__ == '__main__':
    unittest.main()
