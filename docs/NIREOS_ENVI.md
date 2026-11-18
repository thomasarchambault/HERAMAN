# NIREOS exports and ENVI fixtures

Sources consulted on November 10, 2026:

- [HERA software](https://nireos.com/hera-software/): the HERA Acquisition App saves in ENVI format; HyperLAB imports ENVI cubes and provides analysis and export tools.
- [NIREOS FAQ](https://nireos.com/faq/): ENVI export is described as a flat binary with an ASCII header, with MATLAB as another option; datasets are available from NIREOS on request.
- [HERA VNIR](https://nireos.com/product/hera-vnir/): acquisition, analysis, and standard export; reflectance calculation in HyperLAB.
- [ENVI header files](https://www.nv5geospatialsoftware.com/docs/ENVIHeaderFiles.html): specifications for dimensions, data types, interleave, offset, byte order, wavelengths, and units.
- [ENVI image files](https://www.nv5geospatialsoftware.com/docs/enviimagefiles.html): BSQ/BIL/BIP storage.

No complete, freely downloadable official manual was found during that search. A third-party copy of a manual was identified but was not used as a technical specification. At that stage, binary filenames, available fields, units, and the scientific meaning of values still needed to be confirmed against real NIREOS exports. The generated fixtures are standard ENVI files, not certified reproductions of NIREOS outputs. The reader does not assume a uniform HERA spectral grid.

## Native HERAMAN implementation

The reader uses the ENVI header and binary directly (read-only, without NPY conversion). Visualization, ROIs, and SVMs share the same H × W × bands representation. Classification images and masks are written as ENVI BIP with class names as band names, without assigning a spectral axis to these maps. ROI NPZ datasets and Joblib models remain training records separate from the cubes. Outputs reference the source `.hdr` in `classification_source.json` and are stored under `results/`.

## Local HERA acquisitions examined on November 12, 2026

The 30 pairs in `ENVI files/HERA Acquisitions/` contain a `.hdr` header and an extensionless binary (`_raw`, `_ref`, or `_nrm`). They use BSQ storage, float32 (`data type = 4`), little-endian byte order, dimensions of 640 × 512 or 1280 × 1024, and 78/156/234 bands. The offset is 0 or 80 bytes; all binary sizes match their headers.

Their nonuniform axes span 400 to 1000, and all omit `wavelength units`. The reader accepts these axes as nm when every value falls within that range. It fills in `wavelength units` and `heraman wavelength units source` only in the returned metadata. Explicit units take precedence; axes outside this range without units are rejected. No spectral values, source bytes, or headers on disk are modified. These observations describe those local acquisitions and do not certify all NIREOS exports or the scientific meaning of their values.
