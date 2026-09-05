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
import subprocess
import sys
import zipfile

import nbformat

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK = "S3_Reto_Hibridacion_Chimera_tester.ipynb"


def limpiar_salidas_ajenas():
    """Vacia las salidas que sean byte a byte iguales a las del repo.

    El .ipynb del enunciado venia ejecutado por el profesor (checkpoints
    grupo0_*.pth). Si una celda no llego a correr aqui, su salida seguiria
    siendo la de el y se entregaria como propia.
    """
    original = nbformat.reads(
        subprocess.run(["git", "show", f"HEAD:{NOTEBOOK}"], cwd=RAIZ,
                       capture_output=True, text=True).stdout,
        as_version=4,
    )
    actual = nbformat.read(os.path.join(RAIZ, NOTEBOOK), as_version=4)

    ajenas = []
    for i, (c_orig, c_act) in enumerate(zip(original.cells, actual.cells)):
        if c_act.cell_type != "code" or not c_act.get("outputs"):
            continue
        if c_orig.get("outputs") and c_orig["outputs"] == c_act["outputs"]:
            c_act["outputs"] = []
            c_act["execution_count"] = None
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
    args = ap.parse_args()

    limpio = limpiar_salidas_ajenas()
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
