# BioRefusalAudit — Windows PowerShell environment setup.
# Creates a Python 3.13 virtualenv at .venv and installs project + dev extras.

param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

Write-Host "[setup] Using: $Python"
& $Python --version

if (-not (Test-Path ".venv")) {
    Write-Host "[setup] Creating .venv..."
    & $Python -m venv .venv
}

$activate = Join-Path ".venv" "Scripts" "Activate.ps1"
. $activate

Write-Host "[setup] Upgrading pip..."
python -m pip install --upgrade pip

Write-Host "[setup] Installing project in editable mode with [dev,dashboard] extras..."
pip install -e ".[dev,dashboard]"

Write-Host "[setup] Running unit tests..."
python -m pytest tests/ -q -m "not integration"

Write-Host "[setup] Done. Activate with: .\.venv\Scripts\Activate.ps1"
Write-Host "[setup] Run eval with:      .\scripts\run_eval.ps1"
Write-Host "[setup] Launch dashboard:   streamlit run app\dashboard.py"
