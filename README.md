# HERAMAN

**A graphical interface for exploring, annotating, and classifying HERA hyperspectral images.**

HERAMAN supports the analysis of reconstructed and corrected spectral cubes, from inspecting images and spectra to generating classification maps. It brings visualization, region selection, and model training together in a Python application.

The project works directly with **ENVI** files and can analyze acquisitions without the camera or its drivers. Acquisition reconstruction and correction take place upstream, outside HERAMAN.

## What HERAMAN can do

- **Explore a cube**: display an RGB composite or a spectral band, navigate the image, and inspect pixel spectra.
- **Extract regions of interest (ROIs)**: select pixels, rectangles, polygons, or circles, compare their spectra, and export the data. Exported selections can be reloaded on their source cube.
- **Build a training dataset**: define classes and organize annotations into separate profiles.
- **Train and apply SVM models**: compare linear and RBF kernels, with optional probability calibration, then classify other compatible cubes.
- **Inspect results**: overlay class maps on the image, examine probabilities when available, and use intensity segmentation to restrict analysis to the sample.

## Quick start

HERAMAN targets **Windows x64**, with **64-bit Python 3.10 to 3.12**. An Internet connection is required to install dependencies.

1. Download or clone the repository into a writable folder.
2. Install Python x64 if needed, then run **`INSTALL HERAMAN.bat`**. The script creates the local `.venv-heraman` environment and installs the project dependencies.
3. Open the application with **`HERAMAN LAUNCHER.bat`**.
4. The application creates `ENVI files/` on first launch. Create an acquisition subfolder inside it, add your ENVI files, then refresh the cube list and select a cube in the **Visualisation** tab.

Keep each `.hdr` header together with its ENVI binary file. Data and trained models are not included in the repository.

For installation details, troubleshooting, or moving the project to another computer, see the [installation guide](INSTALLATION.md).

## Analysis workflow

**Visualisation** is the entry point for opening a cube, inspecting its contents, and extracting spectra. Acquisition folders organize the cubes to explore; selection tools let you save ROIs and their spectral statistics.

**ROI selection** builds a training dataset from regions annotated by class. Each profile keeps its own dataset, spectral library, and associated models.

**SVM Models** provides tools for inspecting and retraining models. Models can then be applied from **Visualisation**, with optional sample segmentation. The resulting maps are displayed over the image and saved separately from the source data.

To restore an exported selection, click **Load ROIs…** and choose **the cube folder**. The GUI searches that folder and `extracted spectra/` for exports, then opens the associated cube. If multiple selections are available, a list lets you choose which one to restore. See the [ROI export and reload guide](docs/ROI_EXPORTS.md).

## Data and outputs

Cubes are read in ENVI format with their spectral axis. **BSQ, BIL, and BIP** layouts are supported. Source values are preserved: contrast settings affect the display, while spectral normalization, when selected, is applied during processing.

The main output files are:

| Content | Default location | Formats |
| --- | --- | --- |
| Extracted spectra and exported selections | `extracted spectra/`, next to the cube | Pixel CSV with ROI geometry, or mean spectra CSV (one row per ROI); optional PNG image |
| Training datasets and ROI libraries | `roi_outputs/` | NPZ and JSON |
| SVM models | `svm_models/` | Joblib and associated diagnostic files |
| Classification maps and provenance | `results/` | ENVI, PNG previews, and JSON metadata |

Local data, results, models, and environments are excluded from Git tracking in the designated folders. Some exports retain the cube path in their metadata: review their contents before sharing them.

## Project layout

After downloading or cloning the repository, the main folders and entry points are:

```text
HERAMAN/
├── README.md            Project overview and quick start
├── INSTALLATION.md      Setup and troubleshooting guide
├── INSTALL HERAMAN.bat  Windows installer
├── HERAMAN LAUNCHER.bat Windows launcher
├── gui/                 Graphical interface and interactions
│   └── features/        View features and analysis tools
├── processing/          Processing launchers
│   └── pipeline/        ROI selection, training, and classification
├── shared/              ENVI I/O and shared utilities
├── config/              Default runtime configuration
├── styles/              Interface appearance
├── tools/               Installation, diagnostics, and synthetic data
└── docs/                Additional documentation
```

Installation creates `.venv-heraman/`. On first launch, the application creates `ENVI files/`, `roi_outputs/`, and `svm_models/`; these folders initially contain no acquisitions, annotations, or trained models. Classification creates outputs under `results/`, and exporting spectra creates `extracted spectra/` next to the source cube. The output table above describes their contents.

The [architecture overview](docs/ARCHITECTURE.md) explains how the supplied code folders interact and where your data and generated results are stored.

## Try it without acquisitions

After installation, generate the demonstration scenes from the project root:

```powershell
.\.venv-heraman\Scripts\python.exe tools\generate_synthetic_envi.py
```

The generator creates three scenes in `ENVI files/synthetic_data/`, including ENVI cubes and reference label maps. They let you explore visualization, annotation, and classification. Their signatures are artificial and do not represent measurements of real materials.

## Scope and limitations

HERAMAN is an analysis tool whose results depend on the data, annotations, and training protocol. Synthetic-data experiments and software tests do not constitute scientific validation of performance on new acquisitions.

Cubes must be prepared and corrected before import. ENVI fields `bbl`, `data ignore value`, and `reflectance scale factor` are not applied automatically. Reading uses memory mapping, but some processing steps may load large arrays into memory. The installation environment targets Windows; operation on other systems has not been validated.

Header details and HERA acquisition characteristics are covered in the [ENVI documentation](docs/NIREOS_ENVI.md). The spectral comparison methods are described in the [spectral statistics documentation](docs/spectral_statistics.md).

## Research context

HERAMAN was developed as part of a research project conducted at LumedLab, Polytechnique Montréal.

## License

Copyright © 2026 thomasarchambault.

HERAMAN is distributed under the **GNU General Public License, version 3 only** (`GPL-3.0-only`). You may use, modify, and redistribute the software under the terms of this license, including for commercial purposes. Redistribution must comply with its source-code and notice requirements. The software is provided without warranty, as detailed in [LICENSE](LICENSE).

See [NOTICE](NOTICE) for research attribution and [licensing information](docs/LICENSING.md) for an overview of permissions and obligations. Third-party components retain their own licenses.
