#!/bin/bash
# Setup a local .venv environment in project root (Compute Canada compliant)

set -euo pipefail

PYTHON_VER="3.10"
VENV_DIR="$(pwd)/.venv"

# Load system Python (adjust if needed)
module load python/$PYTHON_VER
module load rdkit
module load autodock-vina
module load openbabel
# Load CUDA if you plan to use GPUs (safe to ignore on login node)
module load cuda/12.2 || module load cuda/11.8 || true

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
  echo "[info] Creating new venv at $VENV_DIR"
  python -m venv "$VENV_DIR"
else
  echo "[info] Reusing existing venv at $VENV_DIR"
fi

# Activate it
source "$VENV_DIR/bin/activate"

# Upgrade pip/setuptools/wheel
pip install --upgrade pip setuptools wheel

# Install requirements
if [ -f requirements.txt ]; then
  echo "[info] Installing from requirements.txt"
  pip install -r requirements.txt
else
  echo "[warn] requirements.txt not found!"
fi

# Quick sanity check
python - <<'PY'
import torch
print("Torch:", torch.__version__, "| CUDA avail:", torch.cuda.is_available(), "| CUDA:", torch.version.cuda)
try:
    import torch_geometric
    print("PyG:", torch_geometric.__version__)
except Exception as e:
    print("PyG not installed or failed:", e)
PY

echo "[ok] Env ready. Activate with:"
echo "  source .venv/bin/activate"
