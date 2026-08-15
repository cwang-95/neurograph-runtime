#!/bin/bash
# Install the self-contained Graph 3.0 runtime for this checkout.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV="${NEUROGRAPH_GRAPH3_VENV:-$ROOT/.venv}"
VENV_PYTHON=""

if [[ -x "$VENV/bin/python" ]]; then
  VENV_PYTHON="$VENV/bin/python"
elif [[ -x "$VENV/Scripts/python.exe" ]]; then
  VENV_PYTHON="$VENV/Scripts/python.exe"
else
  "$PYTHON_BIN" -m venv "$VENV"
  if [[ -x "$VENV/bin/python" ]]; then
    VENV_PYTHON="$VENV/bin/python"
  elif [[ -x "$VENV/Scripts/python.exe" ]]; then
    VENV_PYTHON="$VENV/Scripts/python.exe"
  fi
fi

if [[ -z "$VENV_PYTHON" ]]; then
  echo "无法定位虚拟环境 Python: $VENV" >&2
  exit 2
fi

"$VENV_PYTHON" -m pip install --upgrade pip >/dev/null
"$VENV_PYTHON" -m pip install -r "$ROOT/requirements-graph3.txt" -r "$ROOT/requirements-graph3-ann.txt"

echo "Graph 3.0 runtime ready: $VENV"
echo "Run tests: $VENV_PYTHON -m pytest tests -q"
echo "Set NEUROGRAPH_GRAPH3_STORAGE when using an existing external corpus."
