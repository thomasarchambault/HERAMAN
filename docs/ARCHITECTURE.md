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

These folders are included when you download or clone the repository.

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
| `docs/` | Additional guides on architecture, ENVI data, ROI exports, and statistical methods. |

## Local working folders

The repository does not include acquisitions, annotations, or trained models. Installation creates the Python environment; the first application launch creates the empty acquisition, ROI dataset, and model folders. Other output folders appear when you run the corresponding operation.

| Folder | Contents |
| --- | --- |
| `ENVI files/` | Created on first launch. Add your acquisitions in subfolders, or run the generator to populate `synthetic_data/` with demonstrations. |
| `extracted spectra/` | One CSV per exported ROI selection, with spectra and geometry included; optional annotated PNGs. Located next to the corresponding cube. |
| `roi_outputs/` | Created on first launch; populated when you create training datasets, spectral libraries, and annotation previews. |
| `svm_models/` | Created on first launch; populated with trained models and associated evaluation files. |
| `results/` | Created during classification to store class maps, probabilities when available, masks, previews, and settings. |
| `.venv-heraman/` | Locally installed Python environment. It must be recreated on each computer. |

## Code entry points

To start the application, `gui/main.py` initializes Qt and opens the window assembled by `gui/main_window.py`. Scripts at the project root provide the Windows entry points: `INSTALL HERAMAN.bat` installs the environment and `HERAMAN LAUNCHER.bat` starts the application.

`shared/paths.py` centralizes project folders. `shared/envi.py` handles ENVI reading and writing. The main processing stages are described in the [pipeline overview](../processing/pipeline/README.md).

## Source and output separation

Source cubes are read without modification. An ROI selection references their pixels and is exported as one self-contained CSV. A training dataset collects annotated spectra in internal NPZ records with a JSON spectral library; a Joblib model learns the classes defined in that dataset. Applying the model to a compatible cube produces ENVI maps stored separately under `results/`. Input cubes and classification maps share the same image format; training records and models are handled by the application rather than offered as competing ROI export formats.

Metadata links outputs to their cube and processing settings. To retrieve the associated image and spectra, retain the source ENVI pairs and relevant companion files. Internal paths in several provenance files are relative to the project; external references and some exports may contain absolute paths.

Exclusions are defined in [`.gitignore`](../.gitignore). An export placed outside the excluded folders is not automatically protected: review staged files before publishing.
