@echo off
setlocal
cd /d "%~dp0"
if defined MSI_PYTHON64 (
    "%MSI_PYTHON64%" -m gui.main
) else if exist ".venv-heraman\Scripts\python.exe" (
    ".venv-heraman\Scripts\python.exe" -m gui.main
) else (
    python -m gui.main
)
if errorlevel 1 pause
