@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\install_heraman.ps1" %*
set "HERAMAN_SETUP_RESULT=%ERRORLEVEL%"
if not "%HERAMAN_SETUP_RESULT%"=="0" pause
if "%HERAMAN_SETUP_RESULT%"=="0" if "%~1"=="" pause
exit /b %HERAMAN_SETUP_RESULT%
