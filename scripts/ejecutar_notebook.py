"""
ejecutar_notebook.py

Ejecuta el notebook del reto y deja las salidas guardadas dentro del .ipynb
(ese archivo con resultados es uno de los entregables).

Lo hace en dos fases con UN SOLO kernel vivo, porque las tres etapas son
secuenciales y comparten estado en memoria:

    fase 1  celdas 0..N   -> arquitectura propia sobre Intel
    (espera a que exista data/pathmnist_subset, si aun se esta descargando)
    fase 2  celdas N+1..  -> transfer learning y fine-tuning sobre PathMNIST

Guarda el notebook despues de cada celda, asi que si algo revienta a mitad de
camino quedan las salidas de todo lo que si corrio.

Uso:
    python scripts/ejecutar_notebook.py
    python scripts/ejecutar_notebook.py --solo-fase1
    python scripts/ejecutar_notebook.py --espera-datos 7200
"""

import argparse
import json
import os
import sys
import tempfile
import time

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK = os.path.join(RAIZ, "S3_Reto_Hibridacion_Chimera_tester.ipynb")
DATOS_PATH = os.path.join(RAIZ, "data", "pathmnist_subset")

# La fase 1 termina en la celda que guarda el checkpoint de la arquitectura
# propia; la fase 2 arranca en la que carga PathMNIST. Se localizan por
# contenido y no por indice fijo, para que no se rompa si alguien inserta una
# celda de markdown.
MARCA_FIN_FASE1 = "componente=\"arquitectura_propia\""
MARCA_INICIO_FASE2 = "DATA_DIR_PATH ="


def kernel_de_este_python():
    """Registra un kernelspec temporal que apunta a sys.executable.

    Sin esto, nbclient resuelve el kernel "python3" por el kernelspec que
    encuentre primero, que puede ser el de OTRO entorno (otro usuario, otro
    venv, otra build de torch). Paso una tarde entrenando en CPU porque el
    kernel resuelto era un venv con torch+rocm6.4 sin GPU, mientras el
    interprete que lanzaba el script si tenia GPU. El kernel manda: es el que
    ejecuta el notebook.
    """
    dir_jupyter = tempfile.mkdtemp(prefix="chimera-kernel-")
    dir_kernel = os.path.join(dir_jupyter, "kernels", "chimera")
    os.makedirs(dir_kernel)
    spec = {
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": f"chimera ({sys.executable})",
        "language": "python",
    }
    with open(os.path.join(dir_kernel, "kernel.json"), "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=1)

    anterior = os.environ.get("JUPYTER_PATH")
    os.environ["JUPYTER_PATH"] = (
        dir_jupyter if not anterior else dir_jupyter + os.pathsep + anterior
    )
    return "chimera"


def indice_de(nb, marca):
    for i, celda in enumerate(nb.cells):
        if celda.cell_type == "code" and marca in "".join(celda.source):
            return i
    raise SystemExit(f"No se encontro la celda que contiene {marca!r}")


def guardar(nb):
    with open(NOTEBOOK, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)


def ejecutar_rango(cliente, nb, desde, hasta, etiqueta):
    """Ejecuta las celdas [desde, hasta] y guarda tras cada una."""
    print(f"\n===== {etiqueta}: celdas {desde}..{hasta} =====", flush=True)
    for i in range(desde, hasta + 1):
        celda = nb.cells[i]
        if celda.cell_type != "code":
            continue
        primera = "".join(celda.source).strip().splitlines()
        resumen = primera[0][:70] if primera else "(vacia)"
        print(f"  [{i:>2}] {resumen}", flush=True)
        inicio = time.time()
        try:
            cliente.execute_cell(celda, i)
        except CellExecutionError as e:
            guardar(nb)
            print(f"\n  FALLO la celda {i} tras {time.time()-inicio:.0f}s")
            print(f"  {e}")
            raise
        guardar(nb)
        print(f"       ok ({time.time()-inicio:.0f}s)", flush=True)


def esperar_datos(segundos_max):
    if os.path.isdir(DATOS_PATH):
        return True
    print(f"\nEsperando a que exista {DATOS_PATH} (max {segundos_max/60:.0f} min)...",
          flush=True)
    limite = time.time() + segundos_max
    while time.time() < limite:
        if os.path.isdir(DATOS_PATH):
            print("  datos listos", flush=True)
            return True
        time.sleep(30)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo-fase1", action="store_true",
                    help="para en la arquitectura propia, sin transfer ni fine-tuning")
    ap.add_argument("--espera-datos", type=int, default=7200,
                    help="segundos maximos esperando PathMNIST (default 2h)")
    ap.add_argument("--timeout-celda", type=int, default=14400,
                    help="segundos maximos por celda (default 4h)")
    args = ap.parse_args()

    nb = nbformat.read(NOTEBOOK, as_version=4)
    fin_fase1 = indice_de(nb, MARCA_FIN_FASE1)
    inicio_fase2 = indice_de(nb, MARCA_INICIO_FASE2)

    cliente = NotebookClient(nb, timeout=args.timeout_celda,
                             kernel_name=kernel_de_este_python(),
                             resources={"metadata": {"path": RAIZ}})

    inicio_total = time.time()
    # Las dos fases van en el MISMO kernel a proposito: transfer y fine-tuning
    # parten de modelo_propio ya entrenado, que vive en memoria. No hay forma
    # de correr la fase 2 sola sin volver a entrenar la fase 1.
    with cliente.setup_kernel():
        ejecutar_rango(cliente, nb, 0, fin_fase1, "FASE 1 - arquitectura propia")
        print(f"\nFase 1 lista en {(time.time()-inicio_total)/60:.1f} min")
        print("Entregable parcial: entregas/grupo*_arquitectura_propia.pth")

        if args.solo_fase1:
            print("\n--solo-fase1: no se sigue con transfer/fine-tuning")
            return

        if not esperar_datos(args.espera_datos):
            guardar(nb)
            raise SystemExit(f"Se agoto la espera de {DATOS_PATH}. "
                             "Corre scripts/prepare_pathmnist_dataset.py y vuelve "
                             "a lanzar este script (reentrena la fase 1).")

        ejecutar_rango(cliente, nb, inicio_fase2, len(nb.cells) - 1,
                       "FASE 2 - transfer y fine-tuning")

    print(f"\nTerminado en {(time.time()-inicio_total)/60:.1f} min")
    print(f"Notebook con resultados: {NOTEBOOK}")
    entregas = os.path.join(RAIZ, "entregas")
    if os.path.isdir(entregas):
        for n in sorted(os.listdir(entregas)):
            ruta = os.path.join(entregas, n)
            print(f"  {n}  ({os.path.getsize(ruta)/1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
