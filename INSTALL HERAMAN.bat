@echo off
setlocal
set "HERAMAN_POWERSHELL=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if exist "%SystemRoot%\Sysnative\WindowsPowerShell\v1.0\powershell.exe" set "HERAMAN_POWERSHELL=%SystemRoot%\Sysnative\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%HERAMAN_POWERSHELL%" (
    echo PowerShell est introuvable dans le dossier Windows.
    echo Faites verifier l'installation de Windows PowerShell sur ce PC.
    pause
    exit /b 1
)
"%HERAMAN_POWERSHELL%" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\install_heraman.ps1" %*
set "HERAMAN_SETUP_RESULT=%ERRORLEVEL%"
if not "%HERAMAN_SETUP_RESULT%"=="0" pause
if "%HERAMAN_SETUP_RESULT%"=="0" if "%~1"=="" pause
exit /b %HERAMAN_SETUP_RESULT%
