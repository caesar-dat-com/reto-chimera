"""
empaquetar_entrega.py

Cierra la corrida local: verifica que el .ipynb no traiga salidas ajenas,
comprueba que esten los cuatro entregables y arma entregas/grupo77_entrega.zip.

Es el equivalente local de las dos ultimas celdas de colab_run.ipynb.

Uso:
    python scripts/empaquetar_entrega.py
    python scripts/empaquetar_entrega.py --grupo 77
"""

import argparse
import json
import os
import shutil
import sys
import zipfile

import nbformat
from datetime import datetime, timezone

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK = "S3_Reto_Hibridacion_Chimera_tester.ipynb"


def limpiar_salidas_ajenas(desde):
    """Vacia las salidas que no sean de la corrida de este grupo.

    El .ipynb del enunciado venia ejecutado por el profesor (checkpoints
    grupo0_*.pth). Si una celda no llego a correr aqui, su salida seguiria
    siendo la de el y se entregaria como propia.

    El criterio es la marca de tiempo que nbclient deja en
    metadata.execution: una salida es nuestra si la celda se ejecuto en o
    despues de `desde`. Comparar la salida byte a byte contra la del repo NO
    sirve: las celdas deterministas (tablas de costos, conteos del dataset)
    imprimen exactamente lo mismo en cualquier corrida y se borrarian aunque
    si las hayamos ejecutado.
    """
    actual = nbformat.read(os.path.join(RAIZ, NOTEBOOK), as_version=4)

    ajenas = []
    for i, celda in enumerate(actual.cells):
        if celda.cell_type != "code" or not celda.get("outputs"):
            continue
        marca = celda.get("metadata", {}).get("execution", {}).get("iopub.execute_input", "")
        if marca[:len(desde)] >= desde:
            continue
        celda["outputs"] = []
        celda["execution_count"] = None
        ajenas.append(i)

    if ajenas:
        print(f"Salidas del profesor borradas en las celdas: {ajenas}")
        print(">>> La corrida quedo INCOMPLETA: esas celdas no se ejecutaron aqui.")
    else:
        print("Todas las celdas de codigo tienen salidas de esta corrida.")

    with open(os.path.join(RAIZ, NOTEBOOK), "w", encoding="utf-8") as f:
        nbformat.write(actual, f)
    return not ajenas


def verificar_entregables(grupo):
    esperados = [
        f"grupo{grupo}_arquitectura_propia.pth",
        f"grupo{grupo}_transfer.pth",
        f"grupo{grupo}_finetune.pth",
        f"grupo{grupo}_comprobante.json",
    ]
    completo = True
    print("\nEntregables en ./entregas/:")
    for n in esperados:
        ruta = os.path.join(RAIZ, "entregas", n)
        if os.path.exists(ruta):
            print(f"  OK      {n}  ({os.path.getsize(ruta)/1e6:.1f} MB)")
        else:
            print(f"  FALTA   {n}")
            completo = False
    return completo


def empaquetar(grupo):
    entregas = os.path.join(RAIZ, "entregas")
    os.makedirs(entregas, exist_ok=True)
    shutil.copy(os.path.join(RAIZ, NOTEBOOK),
                os.path.join(entregas, f"grupo{grupo}_notebook_ejecutado.ipynb"))

    destino = os.path.join(RAIZ, f"grupo{grupo}_entrega.zip")
    with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED) as z:
        for n in sorted(os.listdir(entregas)):
            z.write(os.path.join(entregas, n), n)
        z.write(os.path.join(RAIZ, "chimera_blocks.py"), "chimera_blocks.py")

    print(f"\n{destino}  ({os.path.getsize(destino)/1e6:.1f} MB)")
    for n in zipfile.ZipFile(destino).namelist():
        print("  -", n)
    return destino


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grupo", type=int, default=77)
    ap.add_argument("--desde", default=None,
                    help="fecha ISO (UTC) desde la que las salidas son de esta "
                         "corrida; default: hoy")
    args = ap.parse_args()

    desde = args.desde or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    limpio = limpiar_salidas_ajenas(desde)
    completo = verificar_entregables(args.grupo)

    comprobante = os.path.join(RAIZ, "entregas", f"grupo{args.grupo}_comprobante.json")
    if os.path.exists(comprobante):
        print("\n--- Resultados ---")
        with open(comprobante, encoding="utf-8") as f:
            print(json.dumps(json.load(f), indent=2, ensure_ascii=False))

    if not (limpio and completo):
        print("\nLa entrega NO esta completa; no se empaqueta.")
        return 1

    empaquetar(args.grupo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
