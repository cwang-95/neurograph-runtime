#!/bin/bash
# Install the self-contained Graph 3.0 runtime for this checkout.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV="${NEUROGRAPH_GRAPH3_VENV:-$ROOT/.venv}"

if [[ ! -x "$VENV/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV"
fi

"$VENV/bin/python" -m pip install --upgrade pip >/dev/null
"$VENV/bin/python" -m pip install -r "$ROOT/requirements-graph3.txt" -r "$ROOT/requirements-graph3-ann.txt"

echo "Graph 3.0 runtime ready: $VENV"
echo "Run tests: $VENV/bin/python -m pytest tests -q"
echo "Set NEUROGRAPH_GRAPH3_STORAGE when using an existing external corpus."
