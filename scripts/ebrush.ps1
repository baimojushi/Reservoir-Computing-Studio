# EBrush Motion Predictor V1: demo / import / train without accessing any hardware.
param(
    [switch]$Demo,
    [switch]$Train,
    [string]$Source = ""
)
$ErrorActionPreference = "Stop"
$Project = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $Project ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Run scripts\setup.ps1 first." }
Set-Location $Project
if ($Demo) {
    if ($Train) { & $Python -m rcstudio project ebrush demo --train }
    else { & $Python -m rcstudio project ebrush demo }
} elseif ($Source) {
    & $Python -m rcstudio project ebrush import $Source
} elseif ($Train) {
    & $Python -m rcstudio project ebrush train --wait
} else {
    & $Python -m rcstudio project ebrush sessions
}
if ($LASTEXITCODE -ne 0) { throw "RC Studio EBrush command failed ($LASTEXITCODE)" }
