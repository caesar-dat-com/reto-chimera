"""
reanudar_finetune.py

Retoma la corrida en la etapa 3 (fine-tuning) sin repetir las dos primeras.

Por que existe: el fine-tuning es la celda mas cara del reto y si el kernel se
muere ahi (timeout de nbclient, corte de luz, OOM) se pierde el estado en
memoria, y ejecutar_notebook.py solo sabe empezar de cero: 36 min de la
arquitectura propia + 2.5 h de transfer antes de volver al punto de la caida.

Los pesos que hacen falta ya estan en disco: entregas/grupoNN_transfer.pth es
exactamente el modelo con el que arranca el fine-tuning. Este script levanta un
kernel, reconstruye ahi el mismo estado que habria dejado la corrida completa
(loaders, modelo cargado desde ese .pth, f1_propio y f1_transfer reevaluados) y
desde ahi ejecuta las celdas del notebook tal cual, sin tocarles una linea, para
que sus salidas sean reales.

Comprueba que los F1 reevaluados coincidan con los que el notebook ya tiene
guardados: si no coinciden, el estado reconstruido no es el de la corrida y
aborta en vez de entregar numeros inventados.

Uso:
    python scripts/reanudar_finetune.py
    python scripts/reanudar_finetune.py --timeout-celda 36000
"""

import argparse
import os
import re
import sys

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

DIR_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DIR_SCRIPTS)

from ejecutar_notebook import (  # noqa: E402
    NOTEBOOK, RAIZ, guardar, indice_de, kernel_de_este_python,
)

SEMILLA = """
# --- Estado reconstruido tras la caida del kernel en el fine-tuning ---------
# Equivale a haber corrido las celdas 14/15/19/20 de nuevo, pero cargando los
# pesos que esas celdas ya dejaron en disco.
import torch as _torch
from chimera_blocks import ChimeraNet as _ChimeraNet

_ruta_propio = f"./entregas/grupo{CODIGO_GRUPO}_arquitectura_propia.pth"
_ruta_transfer = f"./entregas/grupo{CODIGO_GRUPO}_transfer.pth"

checkpoint_propio = _torch.load(_ruta_propio, map_location=device, weights_only=False)
checkpoint_transfer = _torch.load(_ruta_transfer, map_location=device, weights_only=False)

_modelo_intel = _ChimeraNet.from_config(checkpoint_propio["config_arquitectura"]).to(device)
_modelo_intel.load_state_dict(checkpoint_propio["model_state_dict"])
f1_propio, y_true_propio, y_pred_propio = evaluar_f1(_modelo_intel, test_loader_intel, device)
del _modelo_intel

# modelo_propio queda como lo dejo el transfer: backbone de Intel + cabeza de 9
# clases. El fine-tuning parte justo de aqui.
modelo_propio = _ChimeraNet.from_config(checkpoint_transfer["config_arquitectura"]).to(device)
modelo_propio.load_state_dict(checkpoint_transfer["model_state_dict"])
f1_transfer, y_true_t, y_pred_t = evaluar_f1(modelo_propio, test_loader_path, device)

historial_propio = {"val_f1": checkpoint_propio["historial_val_f1"]}
historial_transfer = {"val_f1": checkpoint_transfer["historial_val_f1"]}
tiempo_inicio_propio = tiempo_inicio_transfer = None

print(f"f1_propio={f1_propio:.4f}  f1_transfer={f1_transfer:.4f}")
"""


def f1_guardado_en_notebook(nb, patron):
    for celda in nb.cells:
        for salida in celda.get("outputs", []):
            texto = "".join(salida.get("text", ""))
            m = re.search(patron, texto)
            if m:
                return float(m.group(1))
    return None


def salida_de(celda):
    return "".join("".join(o.get("text", "")) for o in celda.get("outputs", []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout-celda", type=int, default=36000,
                    help="segundos maximos por celda (default 10 h; el fine-tuning "
                         "de 10 epocas a 256x256 no cabe en las 4 h por defecto)")
    args = ap.parse_args()

    nb = nbformat.read(NOTEBOOK, as_version=4)

    i_config = indice_de(nb, "device = torch.device(")
    i_intel = indice_de(nb, "DATA_DIR_INTEL =")
    i_path = indice_de(nb, "DATA_DIR_PATH =")
    i_finetune = indice_de(nb, 'nombre="finetune"')

    esperado_propio = f1_guardado_en_notebook(nb, r"F1 macro final \(arquitectura propia\): ([\d.]+)")
    esperado_transfer = f1_guardado_en_notebook(nb, r"F1 macro final \(transfer learning\): ([\d.]+)")
    if esperado_propio is None or esperado_transfer is None:
        raise SystemExit("El notebook no tiene los F1 de las etapas 1 y 2: no hay nada "
                         "que reanudar, corre scripts/ejecutar_notebook.py desde cero.")

    cliente = NotebookClient(nb, timeout=args.timeout_celda,
                             kernel_name=kernel_de_este_python(),
                             resources={"metadata": {"path": RAIZ}})

    with cliente.setup_kernel():
        for i in (i_config, i_intel, i_path):
            print(f"  [{i:>2}] preparando contexto", flush=True)
            cliente.execute_cell(nb.cells[i], i)
            guardar(nb)

        print("  [--] reconstruyendo el estado del fine-tuning", flush=True)
        # nbclient escribe la celda ejecutada en nb.cells[indice], asi que la
        # semilla tiene que existir en el notebook mientras corre. Se saca
        # despues: no forma parte de la entrega, solo repone el estado perdido.
        semilla = nbformat.v4.new_code_cell(SEMILLA)
        nb.cells.append(semilla)
        try:
            cliente.execute_cell(semilla, len(nb.cells) - 1)
        finally:
            nb.cells.pop()
        texto = salida_de(semilla)
        print("       " + texto.strip())

        m = re.search(r"f1_propio=([\d.]+)\s+f1_transfer=([\d.]+)", texto)
        if not m:
            raise SystemExit(f"La semilla no devolvio los F1:\n{texto}")
        propio, transfer = float(m.group(1)), float(m.group(2))
        if abs(propio - esperado_propio) > 1e-4 or abs(transfer - esperado_transfer) > 1e-4:
            raise SystemExit(
                "El estado reconstruido NO coincide con la corrida guardada:\n"
                f"  arquitectura propia: notebook {esperado_propio} vs recargado {propio}\n"
                f"  transfer:            notebook {esperado_transfer} vs recargado {transfer}\n"
                "Abortado: entregar estos numeros seria mezclar dos corridas distintas.")
        print(f"       coinciden con el notebook ({esperado_propio} / {esperado_transfer})")

        for i in range(i_finetune, len(nb.cells)):
            if nb.cells[i].cell_type != "code":
                continue
            primera = "".join(nb.cells[i].source).strip().splitlines()
            print(f"  [{i:>2}] {primera[0][:70] if primera else '(vacia)'}", flush=True)
            try:
                cliente.execute_cell(nb.cells[i], i)
            except CellExecutionError:
                guardar(nb)
                raise
            guardar(nb)
            print("       ok", flush=True)

    print("\nFine-tuning reanudado y notebook actualizado.")
    entregas = os.path.join(RAIZ, "entregas")
    for n in sorted(os.listdir(entregas)):
        print(f"  {n}  ({os.path.getsize(os.path.join(entregas, n))/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
