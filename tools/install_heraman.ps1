param([switch]$Check, [switch]$Recreate, [string]$Python)
$ErrorActionPreference = 'Stop'
$projectDir = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$environmentDir = Join-Path $projectDir '.venv-heraman'
$environmentPython = Join-Path $environmentDir 'Scripts\python.exe'
$probeCode = "import sys,struct,json; print(json.dumps(dict(path=sys.executable,bits=struct.calcsize('P')*8,version=list(sys.version_info[:2]))))"

function Get-PythonInfo([string]$Executable, [string[]]$PrefixArgs = @()) {
    try {
        $output = & $Executable @PrefixArgs -c $probeCode 2>$null
        if ($LASTEXITCODE -ne 0) { return $null }
        $info = ($output -join "`n") | ConvertFrom-Json
        if ($info.bits -eq 64 -and $info.version[0] -eq 3 -and $info.version[1] -ge 10 -and $info.version[1] -le 12) { return $info.path }
    } catch { return $null }
    return $null
}

function Find-Python {
    if ($Python) {
        $found = Get-PythonInfo $Python
        if (!$found) { throw 'The supplied -Python must be a working Python 3.10-3.12 x64 interpreter.' }
        return $found
    }
    $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($launcher) {
        foreach ($version in @('-3.12-64','-3.11-64','-3.10-64')) {
            $found = Get-PythonInfo $launcher.Source @($version)
            if ($found) { return $found }
        }
    }
    foreach ($name in @('python.exe','python3.exe')) {
        $command = Get-Command $name -ErrorAction SilentlyContinue
        if ($command -and $command.Source -notlike '*\WindowsApps\*') {
            $found = Get-PythonInfo $command.Source
            if ($found) { return $found }
        }
    }
    foreach ($base in @((Join-Path $env:LOCALAPPDATA 'Programs\Python'), $env:ProgramFiles)) {
        if (Test-Path -LiteralPath $base) {
            foreach ($folder in Get-ChildItem -LiteralPath $base -Directory -Filter 'Python*') {
                $candidate = Join-Path $folder.FullName 'python.exe'
                if (Test-Path -LiteralPath $candidate) {
                    $found = Get-PythonInfo $candidate
                    if ($found) { return $found }
                }
            }
        }
    }
    throw 'Install Python 3.10, 3.11 or 3.12 x64 first (including venv/pip), then rerun this installer.'
}

try {
    Set-Location -LiteralPath $projectDir
    $env:PYTHONUTF8 = '1'
    $env:PYTHONIOENCODING = 'utf-8'
    if ($Check) {
        $selectedPython = if (Test-Path -LiteralPath $environmentPython) { $environmentPython } else { Find-Python }
        Write-Host "Checking HERAMAN with $selectedPython"
        & $selectedPython (Join-Path $PSScriptRoot 'verify_environment.py') --gui
        if ($LASTEXITCODE -ne 0) { throw 'Environment validation failed.' }
        exit 0
    }
    if ($Recreate -and (Test-Path -LiteralPath $environmentDir)) {
        $item = Get-Item -LiteralPath $environmentDir
        $resolved = [IO.Path]::GetFullPath((Resolve-Path -LiteralPath $environmentDir).Path)
        $expected = [IO.Path]::GetFullPath((Join-Path $projectDir '.venv-heraman'))
        if ($resolved -ne $expected -or $item.Parent.FullName -ne $projectDir -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
            throw 'Refusing to remove an environment outside the HERAMAN folder or through a link.'
        }
        # Only the verified environment directory is removed; user data is separate.
        Remove-Item -LiteralPath $resolved -Recurse -Force
    }
    if (Test-Path -LiteralPath $environmentDir) {
        if (!(Test-Path -LiteralPath $environmentPython) -or !(Get-PythonInfo $environmentPython)) {
            throw 'Existing environment is invalid or was copied from another PC. Close HERAMAN and rerun INSTALL HERAMAN.bat -Recreate.'
        }
    } else {
        $selectedPython = Find-Python
        Write-Host "Creating dedicated environment with $selectedPython"
        & $selectedPython -m venv $environmentDir
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the virtual environment.' }
    }
    if (!$env:PIP_CERT) {
        $caOutput = & $environmentPython (Join-Path $PSScriptRoot 'windows_ca_bundle.py') (Join-Path $environmentDir 'windows-ca.pem')
        if ($LASTEXITCODE -ne 0) { throw 'Could not read Windows trusted certificates.' }
        $env:PIP_CERT = ($caOutput -join "").Trim()
    }
    & $environmentPython -m pip install --upgrade pip --disable-pip-version-check --retries 1 --timeout 20
    if ($LASTEXITCODE -ne 0) { throw 'pip update failed. Check Internet/proxy access.' }
    & $environmentPython -m pip install --only-binary=:all: --disable-pip-version-check --retries 1 --timeout 20 -r (Join-Path $PSScriptRoot 'requirements-64bit.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Check Internet/proxy access and the Python version.' }
    & $environmentPython (Join-Path $PSScriptRoot 'verify_environment.py') --gui
    if ($LASTEXITCODE -ne 0) { throw 'Environment validation failed. Review the import/DLL error above.' }
    $configFile = Join-Path $projectDir 'config\runtime_64.json'
    if (Test-Path -LiteralPath $configFile) {
        $config = Get-Content -LiteralPath $configFile -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($config.python64) { Copy-Item -LiteralPath $configFile -Destination ($configFile + '.before-install.json') -Force }
        $config.python64 = $null
        $config.expected_gui_bits = 64
        $config | ConvertTo-Json | Set-Content -LiteralPath $configFile -Encoding UTF8
    }
    Write-Host 'HERAMAN is ready. Start HERAMAN LAUNCHER.bat.' -ForegroundColor Green
    if ($env:MSI_PYTHON64) { Write-Host 'Note: MSI_PYTHON64 is set and overrides the normal launcher environment.' }
    exit 0
} catch {
    Write-Host "HERAMAN installation failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
