"""
limpiar_salidas_ajenas.py

El .ipynb original del reto viene con las salidas de la corrida del profesor
(checkpoints 'grupo0_*.pth', sus F1, sus fingerprints). Si se entrega asi,
esos numeros pasan por propios. Este script borra las salidas de las celdas
que TODAVIA no ha ejecutado este grupo.

Uso:
    python scripts/limpiar_salidas_ajenas.py --hasta 12 --salida entrega.ipynb
        -> conserva las salidas de las celdas 0..12 (las que si corrimos)
           y vacia el resto
"""

import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrada", default="S3_Reto_Hibridacion_Chimera_tester.ipynb")
    ap.add_argument("--hasta", type=int, required=True,
                    help="ultima celda cuyas salidas SI son de este grupo")
    ap.add_argument("--salida", required=True)
    args = ap.parse_args()

    nb = json.load(open(args.entrada, encoding="utf-8"))
    limpiadas = []
    for i, celda in enumerate(nb["cells"]):
        if celda["cell_type"] != "code":
            continue
        if i > args.hasta and celda.get("outputs"):
            limpiadas.append(i)
            celda["outputs"] = []
            celda["execution_count"] = None

    with open(args.salida, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")

    print(f"Salidas ajenas borradas en las celdas: {limpiadas}")
    print(f"Escrito: {args.salida}")


if __name__ == "__main__":
    sys.exit(main())
