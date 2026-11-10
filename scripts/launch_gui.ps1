# Optional windowed RC Studio launcher. Never stops E-Brush or other Python tasks.
param(
    [string]$Workspace = ""
)
$ErrorActionPreference = "Stop"
$Project = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $Project ".venv\Scripts\pythonw.exe"
if (-not (Test-Path $Python)) { throw "Run scripts\setup.ps1 first." }
if (-not $Workspace) {
    if ($env:LOCALAPPDATA) { $Workspace = Join-Path $env:LOCALAPPDATA "RCStudio" }
    else { $Workspace = Join-Path $HOME ".rcstudio" }
}
$env:RCSTUDIO_HOME = $Workspace
$env:RCSTUDIO_PORT = "8766"
Start-Process -FilePath $Python -ArgumentList "-m", "rcstudio", "gui" -WorkingDirectory $Project
