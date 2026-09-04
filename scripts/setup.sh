#!/usr/bin/env bash
# Arranque del reto en una maquina nueva.
#   ./scripts/setup.sh            -> torch CPU
#   ./scripts/setup.sh cuda       -> torch NVIDIA (CUDA 12.4)
#   ./scripts/setup.sh rocm       -> torch AMD (ROCm 6.4)
set -euo pipefail
cd "$(dirname "$0")/.."

case "${1:-cpu}" in
  cuda) INDEX="https://download.pytorch.org/whl/cu124" ;;
  rocm) INDEX="https://download.pytorch.org/whl/rocm6.4" ;;
  cpu)  INDEX="https://download.pytorch.org/whl/cpu" ;;
  *) echo "Uso: $0 [cpu|cuda|rocm]"; exit 1 ;;
esac

echo "==> venv"
python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip

echo "==> torch (${1:-cpu})"
.venv/bin/pip install torch torchvision --index-url "$INDEX"

echo "==> resto de dependencias"
.venv/bin/pip install -q -r requirements.txt

echo "==> datos: Intel Image Classification (~363 MB)"
.venv/bin/python scripts/prepare_intel_dataset.py

echo "==> datos: PathMNIST (~1 GB)"
.venv/bin/python scripts/prepare_pathmnist_dataset.py

echo
echo "Listo. Abre el notebook con:"
echo "  .venv/bin/jupyter lab S3_Reto_Hibridacion_Chimera_tester.ipynb"
