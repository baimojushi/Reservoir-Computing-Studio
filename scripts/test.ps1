# Run headless compute + service integration tests.
$ErrorActionPreference = "Stop"
$Project = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $Project ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Run scripts/setup.ps1 first." }
Set-Location $Project
& $Python -m pytest tests learning_tests -v
if ($LASTEXITCODE -ne 0) { throw "Tests failed." }

& $Python -m rcstudio check
if ($LASTEXITCODE -ne 0) { throw "RC Studio health check failed." }
