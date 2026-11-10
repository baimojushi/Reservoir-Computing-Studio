# RC Studio desktop launcher: stop stale workbench/service processes, then open the GUI.
param(
    [string]$Workspace = "D:\项目\AI工具开发\RC\rcstudio-home"
)
$ErrorActionPreference = "Stop"
$Project = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $Project ".venv\Scripts\pythonw.exe"

try {
    if (-not (Test-Path $Python)) {
        throw "Python environment missing. Run scripts\setup.ps1 first.`n$Python"
    }

    # Only RC Studio entry points are stopped: "-m rcstudio gui" / "-m rcstudio serve".
    # Other python or pythonw processes are never touched.
    $stale = @(Get-CimInstance Win32_Process -Filter "Name LIKE 'python%'" |
        Where-Object { $_.CommandLine -match '-m\s+rcstudio\s+(gui|serve)\b' })
    foreach ($item in $stale) {
        Stop-Process -Id $item.ProcessId -Force -ErrorAction SilentlyContinue
    }
    if ($stale.Count -gt 0) { Start-Sleep -Milliseconds 800 }

    $env:RCSTUDIO_HOME = $Workspace
    Start-Process -FilePath $Python -ArgumentList "-m", "rcstudio", "gui" `
        -WorkingDirectory $Project
} catch {
    Add-Type -AssemblyName System.Windows.Forms
    [System.Windows.Forms.MessageBox]::Show(
        $_.Exception.Message, "RC Studio 启动失败") | Out-Null
    exit 1
}
