#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "Creating virtual environment (.venv)..."
"$PYTHON_BIN" -m venv .venv

VENV_PY="$REPO_ROOT/.venv/bin/python"
if [[ ! -f "$VENV_PY" ]]; then
  echo "Virtual environment python not found at $VENV_PY" >&2
  exit 1
fi

echo "Upgrading pip/setuptools/wheel..."
"$VENV_PY" -m pip install --upgrade pip setuptools wheel

echo "Installing project dependencies..."
"$VENV_PY" -m pip install -r requirements.txt

echo "Installing package in editable mode..."
"$VENV_PY" -m pip install -e .

echo "Setup complete."
echo "Activate with: source .venv/bin/activate"
echo "Start backend API: uvicorn breast_cancer_detection.backend.api:app --reload"
echo "Start frontend UI: cd frontend && npm run dev"
