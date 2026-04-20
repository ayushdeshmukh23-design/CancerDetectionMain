param(
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

Write-Host "Creating virtual environment (.venv)..."
& $PythonExe -m venv .venv

$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "Virtual environment python not found at $venvPython"
}

Write-Host "Upgrading pip/setuptools/wheel..."
& $venvPython -m pip install --upgrade pip setuptools wheel

Write-Host "Installing project dependencies..."
& $venvPython -m pip install -r "breast_cancer_detection\requirements.txt"

Write-Host "Installing package in editable mode..."
& $venvPython -m pip install -e .

Write-Host "Setup complete."
Write-Host "Activate with: .\.venv\Scripts\Activate.ps1"
Write-Host "Run app with: streamlit run breast_cancer_detection\app.py"
