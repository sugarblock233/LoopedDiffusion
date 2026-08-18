#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV="${VENV:-$ROOT/.venv}"

command -v "$PYTHON_BIN" >/dev/null
if [[ ! -x "$VENV/bin/python" ]]; then
  "$PYTHON_BIN" -m venv --system-site-packages "$VENV"
fi

PYTHON="$VENV/bin/python"
"$PYTHON" -m pip install --upgrade pip setuptools wheel

if ! "$PYTHON" -c 'import torch, torchvision' >/dev/null 2>&1; then
  if [[ -z "${TORCH_INDEX_URL:-}" ]]; then
    echo "PyTorch/torchvision are unavailable." >&2
    echo "Load the cluster PyTorch module or set TORCH_INDEX_URL, then rerun." >&2
    exit 1
  fi
  "$PYTHON" -m pip install torch torchvision --index-url "$TORCH_INDEX_URL"
fi

"$PYTHON" -m pip install -r "$ROOT/requirements.txt"
"$PYTHON" -c 'import diffusers, pyarrow, torch, torchvision; print("torch", torch.__version__, "cuda", torch.version.cuda, "torchvision", torchvision.__version__)'
