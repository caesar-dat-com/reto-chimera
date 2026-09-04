"""
prepare_intel_dataset.py

Descarga Intel Image Classification (6 clases de paisajes) y arma
./data/intel_subset/{train,test}/<clase>/ en formato ImageFolder, que es lo
que espera el notebook del reto.

Fuente: espejo publico en HuggingFace (sin credenciales, sin cuenta de Kaggle).

Uso:
    python scripts/prepare_intel_dataset.py                 # 800 img/clase train
    python scripts/prepare_intel_dataset.py --por-clase 400 --test-por-clase 200
"""

import argparse
import os
import random
import shutil
import sys
import zipfile

import requests

URL_ZIP = "https://huggingface.co/datasets/miladfa7/Intel-Image-Classification/resolve/main/archive.zip"
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(RAIZ, "data", "_cache")
DESTINO = os.path.join(RAIZ, "data", "intel_subset")
EXTENSIONES = (".jpg", ".jpeg", ".png")


def descargar(url, ruta):
    if os.path.exists(ruta) and os.path.getsize(ruta) > 1_000_000:
        print(f"  ya esta descargado: {ruta} ({os.path.getsize(ruta)/1e6:.0f} MB)")
        return
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    print(f"  descargando {url}")
    parcial = ruta + ".part"
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        bajado = 0
        with open(parcial, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bajado += len(chunk)
                if total:
                    print(f"\r  {bajado/1e6:7.0f} / {total/1e6:.0f} MB", end="", flush=True)
    print()
    os.replace(parcial, ruta)


def localizar_splits(raiz_extraida):
    """Encuentra las carpetas de train y test dentro del zip, sin asumir la
    profundidad exacta (el archivo de Kaggle trae seg_train/seg_train/...)."""
    candidatos = {}
    for actual, subdirs, _ in os.walk(raiz_extraida):
        base = os.path.basename(actual).lower()
        if base in ("seg_train", "seg_test", "train", "test"):
            # nos quedamos con el nivel que contiene directamente las clases
            hijos = [d for d in subdirs if not d.startswith(".")]
            if not hijos:
                continue
            muestra = os.path.join(actual, hijos[0])
            tiene_imagenes = any(
                n.lower().endswith(EXTENSIONES) for n in os.listdir(muestra)
            ) if os.path.isdir(muestra) else False
            if not tiene_imagenes:
                continue
            clave = "train" if "train" in base else "test"
            candidatos.setdefault(clave, actual)
    faltan = {"train", "test"} - set(candidatos)
    if faltan:
        raise RuntimeError(f"No se encontraron los splits {faltan} dentro de {raiz_extraida}")
    return candidatos["train"], candidatos["test"]


def armar_split(origen, destino, por_clase, semilla):
    rng = random.Random(semilla)
    if os.path.isdir(destino):
        shutil.rmtree(destino)
    total = 0
    clases = sorted(d for d in os.listdir(origen) if os.path.isdir(os.path.join(origen, d)))
    for clase in clases:
        archivos = sorted(
            n for n in os.listdir(os.path.join(origen, clase))
            if n.lower().endswith(EXTENSIONES)
        )
        rng.shuffle(archivos)
        if por_clase is not None:
            archivos = archivos[:por_clase]
        carpeta = os.path.join(destino, clase)
        os.makedirs(carpeta, exist_ok=True)
        for nombre in archivos:
            shutil.copy2(os.path.join(origen, clase, nombre), os.path.join(carpeta, nombre))
        total += len(archivos)
        print(f"    {clase:<12} {len(archivos):>5} imagenes")
    return clases, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--por-clase", type=int, default=800,
                    help="imagenes por clase en train (default 800; usa 0 para todas)")
    ap.add_argument("--test-por-clase", type=int, default=300,
                    help="imagenes por clase en test (default 300; usa 0 para todas)")
    ap.add_argument("--semilla", type=int, default=42)
    args = ap.parse_args()

    por_clase = args.por_clase or None
    test_por_clase = args.test_por_clase or None

    zip_local = os.path.join(CACHE, "intel_archive.zip")
    print("1) Descarga")
    descargar(URL_ZIP, zip_local)

    extraido = os.path.join(CACHE, "intel_raw")
    if not os.path.isdir(extraido):
        print("2) Descomprimiendo")
        os.makedirs(extraido, exist_ok=True)
        with zipfile.ZipFile(zip_local) as z:
            z.extractall(extraido)
    else:
        print("2) Ya estaba descomprimido")

    origen_train, origen_test = localizar_splits(extraido)
    print(f"   train crudo: {origen_train}")
    print(f"   test  crudo: {origen_test}")

    print("3) Armando subset")
    print("  train:")
    clases, n_train = armar_split(origen_train, os.path.join(DESTINO, "train"), por_clase, args.semilla)
    print("  test:")
    _, n_test = armar_split(origen_test, os.path.join(DESTINO, "test"), test_por_clase, args.semilla + 1)

    print(f"\nListo -> {DESTINO}")
    print(f"  clases ({len(clases)}): {clases}")
    print(f"  train: {n_train} imagenes  |  test: {n_test} imagenes")


if __name__ == "__main__":
    sys.exit(main())
