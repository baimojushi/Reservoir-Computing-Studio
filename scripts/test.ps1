# Run headless compute + service integration tests.
$ErrorActionPreference = "Stop"
$Project = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $Project ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Run scripts/setup.ps1 first." }
Set-Location $Project
& $Python -m pytest tests -v
if ($LASTEXITCODE -ne 0) { throw "Tests failed." }
