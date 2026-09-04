#!/usr/bin/env bash
# Encadena todo lo que falta, sin intervencion: espera a que termine la
# instalacion de torch, instala el resto de dependencias, comprueba el GPU,
# relanza la descarga de PathMNIST en paralelo y ejecuta el notebook.
set -uo pipefail
cd /home/mark02/reto_chimera

export HSA_OVERRIDE_GFX_VERSION=10.3.0   # RX 6650 XT es gfx1032; ROCm solo
                                         # trae kernels para gfx1030

echo "[$(date +%H:%M:%S)] esperando a que termine pip de torch"
while pgrep -f "pip install torch torchvision" > /dev/null; do sleep 15; done

if ! .venv/bin/python -c "import torch" 2>/dev/null; then
  echo "[$(date +%H:%M:%S)] torch no quedo instalado; reintentando"
  .venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4 || exit 1
fi

echo "[$(date +%H:%M:%S)] instalando el resto de dependencias"
.venv/bin/pip install -q -r requirements.txt nbformat nbclient || exit 1

echo "[$(date +%H:%M:%S)] comprobando GPU"
.venv/bin/python - <<'PY'
import torch
print("torch:", torch.__version__)
print("gpu disponible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    a = torch.randn(4000, 4000, device="cuda")
    torch.cuda.synchronize()
    print("matmul en GPU ok:", float((a @ a).abs().sum()) > 0)
else:
    print("SIN GPU -> el entrenamiento va a CPU y tarda horas")
PY

echo "[$(date +%H:%M:%S)] relanzando descarga de PathMNIST (se reanuda)"
setsid nohup .venv/bin/python scripts/prepare_pathmnist_dataset.py \
  > prep_path.log 2>&1 < /dev/null &

echo "[$(date +%H:%M:%S)] ejecutando notebook"
.venv/bin/python scripts/ejecutar_notebook.py
echo "[$(date +%H:%M:%S)] FIN, codigo=$?"
