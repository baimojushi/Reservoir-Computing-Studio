# Launch RC Studio GUI; the loopback compute service starts automatically.
$ErrorActionPreference = "Stop"
$Project = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $Project ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Run scripts/setup.ps1 first." }
Set-Location $Project
& $Python -m rcstudio gui
