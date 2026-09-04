"""
configurar_notebook.py

Rellena SOLO las dos celdas marcadas `TODO (grupo)` del notebook del reto:
el codigo de grupo y la configuracion de arquitectura. No toca ninguna otra
celda a proposito: el checklist del enunciado exige que el notebook corra de
arriba a abajo sin ediciones fuera de esas dos.

Uso:
    python scripts/configurar_notebook.py --grupo 7 --candidato mixto_barato_ancho
    python scripts/configurar_notebook.py --grupo 7 --candidato residual_profundo --dry-run
"""

import argparse
import json
import os
import re
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "scripts"))

NOTEBOOK = os.path.join(RAIZ, "S3_Reto_Hibridacion_Chimera_tester.ipynb")


def cargar_candidatos():
    """Lee CANDIDATOS de barrido_arquitecturas.py sin importar torch."""
    texto = open(os.path.join(RAIZ, "scripts", "barrido_arquitecturas.py"),
                 encoding="utf-8").read()
    bloque = re.search(r"^CANDIDATOS = \[.*?^\]", texto, re.S | re.M)
    espacio = {}
    exec(bloque.group(0), espacio)  # noqa: S102 - fuente propia del repo
    return {c["nombre"]: c for c in espacio["CANDIDATOS"]}


def costos():
    """Lee las tablas de costos de chimera_blocks.py sin importar torch."""
    texto = open(os.path.join(RAIZ, "chimera_blocks.py"), encoding="utf-8").read()
    espacio = {}
    for nombre in ("BLOCK_COSTS", "OPTIMIZER_COSTS", "SCHEDULER_COSTS",
                   "LOSS_COSTS", "COMPONENT_COSTS"):
        bloque = re.search(rf"^{nombre} = \{{.*?^\}}", texto, re.S | re.M)
        exec(bloque.group(0), espacio)  # noqa: S102
    return espacio


def presupuesto(cfg, tablas):
    total = sum(tablas["BLOCK_COSTS"][b] for b in cfg["bloques"])
    total += tablas["OPTIMIZER_COSTS"][cfg["optimizador"]]
    total += tablas["SCHEDULER_COSTS"][cfg["scheduler"]]
    total += tablas["LOSS_COSTS"][cfg["loss"]]
    comp = tablas["COMPONENT_COSTS"]
    if cfg["conv_custom"]:
        total += comp["conv_custom"]
    if cfg["pooling_extra"] == "maxpool":
        total += comp["maxpool_extra"]
    elif cfg["pooling_extra"] == "avgpool":
        total += comp["avgpool_extra"]
    if cfg["gap"]:
        total += comp["global_avg_pool"]
    if cfg["dropout"]:
        total += comp[cfg["dropout"][0]]
    return total


def celda_configuracion(cfg, total):
    bloques = json.dumps(cfg["bloques"], ensure_ascii=False)
    dropout = "None" if cfg["dropout"] is None else f'("{cfg["dropout"][0]}", {cfg["dropout"][1]})'
    pooling = "None" if cfg["pooling_extra"] is None else f'"{cfg["pooling_extra"]}"'
    return f'''# Configuracion del grupo -- candidato "{cfg["nombre"]}" ({total} pts)
MI_BLOQUES = {bloques}
MI_OPTIMIZADOR = "{cfg["optimizador"]}"
MI_SCHEDULER = "{cfg["scheduler"]}"
MI_LOSS = "{cfg["loss"]}"
MI_CONV_CUSTOM = {cfg["conv_custom"]}
MI_POOLING_EXTRA = {pooling}     # None | "maxpool" | "avgpool"
MI_GLOBAL_AVG_POOL = {cfg["gap"]}
MI_DROPOUT = {dropout}                # None | ("dropout", p) | ("dropout_espacial", p)

total_pts, presupuesto_valido, _ = calcular_presupuesto(
    MI_BLOQUES, MI_OPTIMIZADOR, MI_SCHEDULER, MI_LOSS,
    usar_conv_custom=MI_CONV_CUSTOM, pooling_extra=MI_POOLING_EXTRA,
    usar_global_avg_pool=MI_GLOBAL_AVG_POOL, dropout=MI_DROPOUT,
)
assert presupuesto_valido, (
    f"Su combinacion no cumple el presupuesto ({{PRESUPUESTO_MIN}}-{{PRESUPUESTO_MAX}} pts). "
    f"Total actual: {{total_pts}}. Ajustenla antes de seguir -- esto NO se puede entregar asi."
)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grupo", type=int, required=True)
    ap.add_argument("--candidato", default="mixto_barato_ancho")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    candidatos = cargar_candidatos()
    if args.candidato not in candidatos:
        raise SystemExit(f"Candidato desconocido: {args.candidato}. "
                         f"Opciones: {list(candidatos)}")
    cfg = candidatos[args.candidato]
    tablas = costos()
    total = presupuesto(cfg, tablas)
    if not 150 <= total <= 200:
        raise SystemExit(f"'{args.candidato}' gasta {total} pts, fuera del rango 150-200.")

    nb = json.load(open(NOTEBOOK, encoding="utf-8"))
    cambios = []

    for celda in nb["cells"]:
        if celda["cell_type"] != "code":
            continue
        fuente = "".join(celda["source"])

        if "CODIGO_GRUPO = 0" in fuente:
            fuente = fuente.replace(
                "# TODO (grupo): reemplacen 0 por el codigo real de su grupo antes de correr.\n"
                "CODIGO_GRUPO = 0",
                f"CODIGO_GRUPO = {args.grupo}")
            celda["source"] = fuente.splitlines(keepends=True)
            cambios.append(f"CODIGO_GRUPO -> {args.grupo}")

        elif 'MI_BLOQUES = ["capa1"' in fuente:
            nueva = celda_configuracion(cfg, total)
            celda["source"] = nueva.splitlines(keepends=True)
            cambios.append(f"configuracion -> {cfg['nombre']} ({total} pts, "
                           f"{len(cfg['bloques'])} bloques)")

    if len(cambios) != 2:
        raise SystemExit(f"Se esperaban 2 celdas TODO, se tocaron {len(cambios)}: {cambios}. "
                         "El notebook ya fue editado o cambio de forma.")

    for c in cambios:
        print(" ", c)

    if args.dry_run:
        print("\n--dry-run: no se escribio nada")
        return

    with open(NOTEBOOK, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"\nNotebook actualizado: {NOTEBOOK}")


if __name__ == "__main__":
    sys.exit(main())
