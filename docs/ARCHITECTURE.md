# HERAMAN architecture

HERAMAN separates the graphical interface, analysis processing, and shared utilities. Acquisitions and generated files are kept apart from the source code.

## General workflow

```text
ENVI acquisitions (.hdr + binary)
        │
        ▼
Graphical interface (gui/)
        ├── Exploration and spectral extraction
        ├── Annotation and ROI dataset creation
        └── Model selection and analysis settings
                    │
                    ▼
Processing launchers (processing/)
                    │
                    ▼
ROI selection, training, and classification (processing/pipeline/)
                    │
                    ▼
ROI datasets, models, and output maps
```

The interface and processing modules use `shared/` to read cubes, manage paths, transform spectra, and follow common conventions. Some interactive operations, such as spectral extraction, run directly in the GUI. ROI and SVM processing launched separately uses the same 64-bit Python environment as the application by default.

## Source and configuration folders

| Folder | Purpose |
| --- | --- |
| `gui/` | Application startup, main window assembly, display components, and dialogs. |
| `gui/features/` | Visualization tools, ROI selection and reloading, model management, spectral statistics, and classification display. These components are assembled in the main window. |
| `processing/` | Entry points for processing in separate processes, environment checks, and model availability checks. |
| `processing/pipeline/` | Scientific processing: training dataset creation, SVM training, optional segmentation, classification, and output generation. |
| `shared/` | Functions reused across the application: ENVI I/O, previews, normalization, calibration, segmentation, provenance, and ROI import/export. |
| `config/` | Runtime environment settings. The local cube catalogue, `envi_sources.json`, is generated and ignored by Git; the generic `runtime_64.json` configuration is supplied. |
| `styles/` | Qt stylesheet defining the interface appearance. |
| `tools/` | Environment installation, startup, diagnostics, and synthetic scene generation. Dependencies are defined in `requirements-64bit.txt`. |
| `tests/` | Local automated tests and workflow checks. This folder is excluded from Git and is not distributed with the repository. |
| `docs/` | Additional guides on architecture, ENVI data, ROI exports, and statistical methods. |

## Local working folders

These folders are not distributed as data with the repository. They appear during installation, when acquisitions are added, or when analyses run.

| Folder | Contents |
| --- | --- |
| `ENVI files/` | Acquisitions to analyze, organized into subfolders. `synthetic_data/` holds generated demonstrations. |
| `extracted spectra/` | Exported spectra, ROI positions, and geometry, next to the corresponding cube. This folder may therefore exist in several acquisition folders. |
| `roi_outputs/` | Training datasets by profile, spectral libraries, previews, and annotation provenance. |
| `svm_models/` | Trained models and associated evaluation files. |
| `results/` | Class maps, probabilities when available, masks, previews, and classification settings. |
| `.venv-heraman/` | Locally installed Python environment. It must be recreated on each computer. |
| `tmp/` | Temporary files used by some local checks and operations. |

## Code entry points

To start the application, `gui/main.py` initializes Qt and opens the window assembled by `gui/main_window.py`. Scripts at the project root provide the Windows entry points: `INSTALL HERAMAN.bat` installs the environment and `HERAMAN LAUNCHER.bat` starts the application.

`shared/paths.py` centralizes project folders. `shared/envi.py` handles ENVI reading and writing. The main processing stages are described in the [pipeline overview](../processing/pipeline/README.md).

## Source and output separation

Source cubes are read without modification. An ROI selection references their pixels; a training dataset collects annotated spectra; a model learns the classes defined in that dataset. Applying the model to a compatible cube produces maps stored separately under `results/`.

Metadata links outputs to their cube and processing settings. To retrieve the associated image and spectra, retain the source ENVI pairs and relevant companion files. Internal paths in several provenance files are relative to the project; external references and some exports may contain absolute paths.

Exclusions are defined in [`.gitignore`](../.gitignore). An export placed outside the excluded folders is not automatically protected: review staged files before publishing.
