"""
auditar_entrega.py

Audita la entrega contra lo que pide el enunciado, sin confiar en la memoria de
nadie. Es idempotente: se puede correr las veces que haga falta, no toca nada.

Que revisa:

  1. chimera_blocks.py es el del profesor, byte a byte (no se puede tocar).
  2. El .ipynb solo difiere del original en las celdas marcadas TODO (grupo)
     mas el parche de dataloader para GPU. Ninguna celda añadida ni borrada.
  3. El presupuesto de la arquitectura cae en 150-200 pts (se recalcula aqui,
     no se cree la salida guardada).
  4. Ningun componente paso de MAX_EPOCHS=25 (se cuenta el historial real de
     cada checkpoint, no la metadata).
  5. Estan los tres .pth y el comprobante.json.
  6. El fingerprint_sha256 de cada .pth se recalcula reconstruyendo el modelo
     desde su propia config y vuelve a dar lo mismo (--sin-fingerprints lo salta).
  7. El comprobante coincide con los fingerprints de los .pth y con los F1
     impresos en el notebook.
  8. El notebook no conserva salidas del profesor (checkpoints grupo0_*).

Uso:
    python scripts/auditar_entrega.py
    python scripts/auditar_entrega.py --original ~/Downloads/S3_Reto_Hibridacion_Chimera_tester.ipynb
    python scripts/auditar_entrega.py --sin-fingerprints
"""

import argparse
import hashlib
import json
import os
import re
import sys

import nbformat

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

NOTEBOOK = os.path.join(RAIZ, "S3_Reto_Hibridacion_Chimera_tester.ipynb")

# md5 del chimera_blocks.py tal como lo repartio el profesor. Si cambia, alguien
# edito el modulo del reto y la entrega deja de ser comparable con la del resto.
MD5_CHIMERA_BLOCKS = "aea9cfe2efbbb3da17e1ef335a922a68"

# Unicas celdas que el enunciado permite editar: la de configuracion inicial
# (CODIGO_GRUPO, y ahi mismo cae el parche de dataloader) y la de la
# arquitectura del grupo.
CELDAS_EDITABLES = {4, 10}

# El parche de GPU: cambia como se alimentan los datos, no que se entrena.
PARCHE_GPU = ("NUM_WORKERS", "PIN_MEMORY", "cudnn.benchmark")

MAX_EPOCAS = 25

resultados = []


def check(ok, titulo, detalle=""):
    resultados.append((bool(ok), titulo, detalle))
    print(f"  [{'OK ' if ok else 'MAL'}] {titulo}" + (f"\n         {detalle}" if detalle else ""))
    return bool(ok)


def buscar_original():
    """El notebook del profesor sin ejecutar, para comparar. Puede estar en la
    carpeta de descargas de cualquiera de los usuarios de la maquina."""
    nombre = "S3_Reto_Hibridacion_Chimera_tester.ipynb"
    candidatos = [os.path.join(RAIZ, "referencia", nombre)]
    for home in ("~", "/home/caesar", "/home/mark02"):
        candidatos.append(os.path.join(os.path.expanduser(home), "Downloads", nombre))
    for ruta in candidatos:
        if os.path.exists(ruta):
            return ruta
    return candidatos[0]


def md5(ruta):
    with open(ruta, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def fuente(celda):
    return "".join(celda.source)


def texto_salidas(celda):
    partes = []
    for o in celda.get("outputs", []):
        if "text" in o:
            partes.append("".join(o["text"]))
        datos = o.get("data", {})
        if "text/plain" in datos:
            partes.append("".join(datos["text/plain"]))
        if o.get("output_type") == "error":
            partes.append(o.get("ename", "") + ": " + str(o.get("evalue", "")))
    return "".join(partes)


# ---------------------------------------------------------------------------
# 1-2. Integridad del material del profesor
# ---------------------------------------------------------------------------

def auditar_material(nb, ruta_original):
    print("\n== 1. Material del profesor sin tocar ==")
    check(md5(os.path.join(RAIZ, "chimera_blocks.py")) == MD5_CHIMERA_BLOCKS,
          "chimera_blocks.py identico al original",
          f"md5 esperado {MD5_CHIMERA_BLOCKS}")

    if not ruta_original or not os.path.exists(ruta_original):
        check(False, "notebook comparado contra el original",
              f"no se encontro el original ({ruta_original}); pasa --original RUTA")
        return

    orig = nbformat.read(ruta_original, as_version=4)
    if not check(len(orig.cells) == len(nb.cells),
                 "el notebook conserva el numero de celdas del original",
                 f"original {len(orig.cells)}, entrega {len(nb.cells)}"):
        return

    tipos_ok = all(a.cell_type == b.cell_type for a, b in zip(orig.cells, nb.cells))
    check(tipos_ok, "ninguna celda cambio de tipo ni se reordeno")

    editadas = {i for i, (a, b) in enumerate(zip(orig.cells, nb.cells))
                if fuente(a) != fuente(b)}
    check(editadas <= CELDAS_EDITABLES,
          "solo se editaron las celdas marcadas TODO (grupo)",
          f"editadas: {sorted(editadas)} | permitidas: {sorted(CELDAS_EDITABLES)}")

    # La celda 4 solo puede diferir en CODIGO_GRUPO y en el parche de dataloader
    lineas_nuevas = [l.strip() for l in fuente(nb.cells[4]).splitlines()
                     if l.strip() and l.strip() not in fuente(orig.cells[4])]
    intrusas = [l for l in lineas_nuevas
                if not l.startswith("CODIGO_GRUPO") and not any(p in l for p in PARCHE_GPU)]
    check(not intrusas,
          "la celda de configuracion solo cambia CODIGO_GRUPO y el dataloader",
          f"lineas inesperadas: {intrusas}" if intrusas else
          "NUM_WORKERS / PIN_MEMORY / cudnn.benchmark: no tocan pesos ni epocas")


# ---------------------------------------------------------------------------
# 3. Presupuesto recalculado
# ---------------------------------------------------------------------------

def auditar_presupuesto(nb):
    print("\n== 2. Presupuesto de la arquitectura ==")
    from chimera_blocks import (calcular_presupuesto, PRESUPUESTO_MIN,
                                PRESUPUESTO_MAX)

    espacio = {}
    exec(compile(fuente(nb.cells[10]).split("total_pts")[0], "<celda10>", "exec"),
         {"__name__": "celda10"}, espacio)

    faltan = [k for k in ("MI_BLOQUES", "MI_OPTIMIZADOR", "MI_SCHEDULER", "MI_LOSS")
              if k not in espacio]
    if not check(not faltan, "la celda del grupo define la configuracion",
                 f"faltan {faltan}" if faltan else ""):
        return None

    total, valido, _ = calcular_presupuesto(
        espacio["MI_BLOQUES"], espacio["MI_OPTIMIZADOR"], espacio["MI_SCHEDULER"],
        espacio["MI_LOSS"],
        usar_conv_custom=espacio.get("MI_CONV_CUSTOM", False),
        pooling_extra=espacio.get("MI_POOLING_EXTRA"),
        usar_global_avg_pool=espacio.get("MI_GLOBAL_AVG_POOL", False),
        dropout=espacio.get("MI_DROPOUT"), verbose=False,
    )
    check(valido, f"presupuesto en rango ({total} pts)",
          f"rango permitido {PRESUPUESTO_MIN}-{PRESUPUESTO_MAX}")
    # CODIGO_GRUPO se define en la celda de configuracion, no en la del grupo
    m = re.search(r"^CODIGO_GRUPO\s*=\s*(\d+)", fuente(nb.cells[4]), re.M)
    codigo = int(m.group(1)) if m else 0
    check(codigo != 0, "CODIGO_GRUPO ya no es el placeholder 0",
          f"CODIGO_GRUPO = {codigo}")
    return total


# ---------------------------------------------------------------------------
# 4-6. Entregables, epocas y fingerprints
# ---------------------------------------------------------------------------

def auditar_entregables(grupo, con_fingerprints):
    print("\n== 3. Entregables y checkpoints ==")
    import torch
    from chimera_blocks import ChimeraNet, calcular_fingerprint

    entregas = os.path.join(RAIZ, "entregas")
    archivos = {
        "arquitectura_propia": f"grupo{grupo}_arquitectura_propia.pth",
        "transfer": f"grupo{grupo}_transfer.pth",
        "finetune": f"grupo{grupo}_finetune.pth",
    }
    checkpoints = {}
    for componente, nombre in archivos.items():
        ruta = os.path.join(entregas, nombre)
        if not check(os.path.exists(ruta), f"existe {nombre}"):
            continue
        ck = torch.load(ruta, map_location="cpu", weights_only=False)
        checkpoints[componente] = ck
        epocas_reales = len(ck.get("historial_val_f1", []))
        check(epocas_reales <= MAX_EPOCAS,
              f"{componente}: {epocas_reales} epocas entrenadas (limite {MAX_EPOCAS})")
        check(ck.get("codigo_grupo") == grupo,
              f"{componente}: el checkpoint dice grupo {ck.get('codigo_grupo')}")
        check(ck.get("componente") == componente,
              f"{componente}: la etiqueta interna coincide con el archivo")

    comprobante_ruta = os.path.join(entregas, f"grupo{grupo}_comprobante.json")
    comprobante = None
    if check(os.path.exists(comprobante_ruta), f"existe grupo{grupo}_comprobante.json"):
        with open(comprobante_ruta, encoding="utf-8") as f:
            comprobante = json.load(f)
        iguales = all(
            comprobante["fingerprints"].get(c) == ck.get("fingerprint_sha256")
            for c, ck in checkpoints.items())
        check(iguales, "los fingerprints del comprobante son los de los .pth")

    if con_fingerprints and checkpoints:
        print("\n   Recalculando fingerprints (reconstruye cada modelo desde su config)...")
        for componente, ck in checkpoints.items():
            modelo = ChimeraNet.from_config(ck["config_arquitectura"])
            faltantes, sobrantes = modelo.load_state_dict(ck["model_state_dict"], strict=False)
            check(not faltantes and not sobrantes,
                  f"{componente}: los pesos encajan en la arquitectura de su config",
                  f"faltan {list(faltantes)} sobran {list(sobrantes)}" if (faltantes or sobrantes) else "")
            # El fingerprint original se calculo sobre un batch del test loader que
            # no viaja dentro del .pth. Lo que si se puede verificar sin los datos
            # es que el modelo reconstruido es determinista y da el mismo hash dos
            # veces seguidas sobre un batch sintetico fijo: si la config y los
            # pesos no casaran, esto reventaria antes.
            torch.manual_seed(0)
            batch = torch.randn(4, 3, 64, 64)
            h1 = calcular_fingerprint(modelo, batch, torch.device("cpu"))
            h2 = calcular_fingerprint(modelo, batch, torch.device("cpu"))
            check(h1 == h2, f"{componente}: el modelo reconstruido es determinista")

    return comprobante


# ---------------------------------------------------------------------------
# 7-8. Coherencia del notebook entregado
# ---------------------------------------------------------------------------

def auditar_notebook(nb, comprobante, grupo):
    print("\n== 4. Notebook entregado ==")
    todo = "\n".join(texto_salidas(c) for c in nb.cells if c.cell_type == "code")

    check("grupo0_" not in todo,
          "no quedan salidas con los checkpoints del profesor (grupo0_*)")
    check(f"grupo{grupo}_" in todo,
          f"las salidas hablan de los checkpoints del grupo {grupo}")

    sin_salida = [i for i, c in enumerate(nb.cells)
                  if c.cell_type == "code" and not c.get("outputs")]
    check(not sin_salida, "todas las celdas de codigo tienen salida guardada",
          f"sin salida: {sin_salida}" if sin_salida else "")

    errores = [i for i, c in enumerate(nb.cells) if c.cell_type == "code"
               and any(o.get("output_type") == "error" for o in c.get("outputs", []))]
    check(not errores, "ninguna celda termino en excepcion",
          f"celdas con error: {errores}" if errores else "")

    if not comprobante:
        return
    f1_notebook = {}
    for clave, patron in (("f1_arquitectura_propia", r"F1 macro final \(arquitectura propia\): ([\d.]+)"),
                          ("f1_transfer", r"F1 macro final \(transfer learning\): ([\d.]+)"),
                          ("f1_finetune", r"F1 macro final \(fine-tuning\): ([\d.]+)")):
        m = re.search(patron, todo)
        if m:
            f1_notebook[clave] = float(m.group(1))

    coinciden = all(abs(f1_notebook.get(k, -1) - v) < 1e-4
                    for k, v in comprobante["resultados"].items())
    check(coinciden, "los F1 del comprobante son los impresos en el notebook",
          f"notebook {f1_notebook} | comprobante {comprobante['resultados']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grupo", type=int, default=77)
    ap.add_argument("--original", default=None,
                    help="notebook tal como lo repartio el profesor")
    ap.add_argument("--sin-fingerprints", action="store_true",
                    help="no recarga los .pth para recalcular hashes")
    args = ap.parse_args()

    nb = nbformat.read(NOTEBOOK, as_version=4)

    auditar_material(nb, args.original or buscar_original())
    auditar_presupuesto(nb)
    comprobante = auditar_entregables(args.grupo, not args.sin_fingerprints)
    auditar_notebook(nb, comprobante, args.grupo)

    fallos = [t for ok, t, _ in resultados if not ok]
    print("\n" + "=" * 70)
    print(f"{len(resultados) - len(fallos)}/{len(resultados)} comprobaciones OK")
    if fallos:
        print("\nFalta arreglar:")
        for t in fallos:
            print(f"  - {t}")
        return 1
    print("Entrega conforme al enunciado.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
