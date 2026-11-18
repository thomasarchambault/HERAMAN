# Installing and moving HERAMAN

HERAMAN targets **Windows x64**, with **Python 3.10, 3.11, or 3.12 x64**. Installation has been tested locally with Python 3.10. HERA/Hamamatsu/VariSpec drivers and NIREOS software are not required to analyze ENVI exports.

## First installation on a computer

1. Install Python x64 within the supported range if needed, including the standard `venv` and `pip` components.
2. Place the HERAMAN folder in a writable location, then double-click **INSTALL HERAMAN.bat**.
3. The installer detects Python, creates **.venv-heraman**, installs the versions defined in `tools/requirements-64bit.txt`, and checks imports and Qt. Internet access is required to download dependencies. Windows trusted certificates are used for pip; TLS verification remains enabled.
4. Run **HERAMAN LAUNCHER.bat**. It selects the dedicated environment; ROI/SVM processes use the same Python interpreter.

The script does not download or install Python itself. No dependencies are installed into the global Python environment.

## Moving a project

Copy the HERAMAN folder **without virtual environments** such as `.venv-heraman`, `.venv64`, `.venv`, or `venv`. Recreate the environment on each computer with the installer; a virtual environment is not a portable package.

To retain your data and work, also copy the relevant folders:

- `ENVI files/`: one subfolder per acquisition group, including `synthetic_data/` for synthetic data and `acquisitions_reelles/` for measurements. Keep ENVI `.hdr` + binary pairs together and preserve the header's `data file` field. Older `data/` and `synthetic_data/` folders remain accessible if present.
- `roi_outputs/`: annotations, training datasets, and libraries.
- `svm_models/`: models and profiles; retain the same library versions when reloading them.
- `results/`: classifications and provenance.
- `config/envi_sources.json`: optional local catalogue.

New internal paths in the catalogue, classification provenance, and ROI datasets are relative to the HERAMAN folder. The folder can be renamed and moved. Files outside HERAMAN remain absolute references: copy them and select them again on the new computer. An older result containing an outdated absolute path may need to be regenerated from its ENVI source. External ENVI headers that use an absolute `data file` path must be updated if that location changes.

Large data files and models are ignored by Git: cloning the repository does not transfer them. To create the three demonstration scenes after cloning, run:

```powershell
.\.venv-heraman\Scripts\python.exe tools\generate_synthetic_envi.py
```

## Checking or repairing the installation

In PowerShell, from the HERAMAN folder:

```powershell
& '.\INSTALL HERAMAN.bat' -Check
& '.\INSTALL HERAMAN.bat' -Python 'D:\Python312\python.exe'
```

The Python path above is an example; replace it with the interpreter you want to use.

To recreate only the environment, close HERAMAN and run:

```powershell
& '.\INSTALL HERAMAN.bat' -Recreate
```

`-Recreate` removes only the verified `.venv-heraman` folder inside HERAMAN, then reinstalls dependencies. Data, annotations, and models are not deleted.

An explicit `MSI_PYTHON64` setting or `python64` in `config/runtime_64.json` can select another interpreter. Installation resets the configuration to automatic selection; if it replaces an explicit path, it keeps a `runtime_64.json.before-install.json` backup. An explicit environment variable retains priority over the launcher.

## Verification performed

Previous checks included installation in an isolated environment, dependency diagnostics, and Qt initialization; a small project was also copied to a renamed folder containing spaces and accented characters, the old copy removed, and the catalogue, model, and ENVI overlay reloaded. These checks establish relocation on the same machine; they did not use a second physical computer.
