# RC Studio Windows setup. Run from any PowerShell working directory.
$ErrorActionPreference = "Stop"
$Project = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Venv = Join-Path $Project ".venv"
$Python = Join-Path $Venv "Scripts\python.exe"
Set-Location $Project

if (-not (Test-Path $Python)) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 -m venv $Venv
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python -m venv $Venv
    } else {
        throw "Python 3.11+ was not found. Install Python first."
    }
    if ($LASTEXITCODE -ne 0) { throw "Python virtual environment creation failed." }
}

& $Python -c "import sys; assert sys.version_info >= (3,11), 'Python 3.11+ required'"
if ($LASTEXITCODE -ne 0) { throw "Python version is too old." }
& $Python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }
& $Python -m pip install -e ".[all]"
if ($LASTEXITCODE -ne 0) { throw "RC Studio dependencies installation failed." }

Write-Host ""
Write-Host "RC Studio installed successfully." -ForegroundColor Green
Write-Host "Launch GUI: powershell -ExecutionPolicy Bypass -File .\scripts\start.ps1"
Write-Host "Run tests : powershell -ExecutionPolicy Bypass -File .\scripts\test.ps1"
