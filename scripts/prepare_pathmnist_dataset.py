"""
prepare_pathmnist_dataset.py

Descarga PathMNIST (histologia de tejido colorrectal, 9 clases) desde el
repositorio oficial de MedMNIST en Zenodo y arma
./data/pathmnist_subset/{train,test}/<clase>/ en formato ImageFolder.

Resolucion: default 128 (4 GB). El enunciado dice que las imagenes vienen a
256x256 upsampleadas desde el maximo nativo de MedMNIST (224x224), y el
notebook aplica transforms.Resize((256, 256)) encima.

2026-09-04: el default era 64. En histologia la clase depende de textura
celular fina, y subir 64 -> 256 (x4) la borra: el transfer learning se quedo en
F1 0.4873 contra una meta de 0.87. A 128 el upsampling es solo x2.

RAM: el .npz comprimido no admite lectura parcial, asi que cargarlo entero
costaba ~13 GB a 224. Ahora los .npy internos se extraen a disco y se leen con
mmap_mode="r", de modo que volcar_split solo pagina las filas que necesita.
Con eso --resolucion 224 tambien corre en Colab (disco, no RAM).

Uso:
    python scripts/prepare_pathmnist_dataset.py
    python scripts/prepare_pathmnist_dataset.py --por-clase 800 --resolucion 128
"""

import argparse
import os
import shutil
import sys
import zipfile

import numpy as np
import requests
from PIL import Image

URL_BASE = "https://zenodo.org/records/10519652/files"
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(RAIZ, "data", "_cache")
DESTINO = os.path.join(RAIZ, "data", "pathmnist_subset")

# Orden oficial de las etiquetas de PathMNIST (indice -> nombre de carpeta),
# con los mismos nombres que usa el enunciado del reto.
CLASES = [
    "adipose",              # 0
    "background",           # 1
    "debris",               # 2
    "lymphocytes",          # 3
    "mucus",                # 4
    "smooth_muscle",        # 5
    "normal_colon_mucosa",  # 6
    "cancer_stroma",        # 7
    "adenocarcinoma",       # 8
]


def descargar(url, ruta):
    if os.path.exists(ruta) and os.path.getsize(ruta) > 1_000_000:
        print(f"  ya esta descargado: {ruta} ({os.path.getsize(ruta)/1e6:.0f} MB)")
        return
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    parcial = ruta + ".part"
    # Reanuda con HTTP Range si ya hay un .part: el archivo son ~1 GB desde
    # Zenodo, que va lento, y no vale la pena rempezar de cero si se corta.
    ya = os.path.getsize(parcial) if os.path.exists(parcial) else 0
    cabeceras = {"Range": f"bytes={ya}-"} if ya else {}
    print(f"  descargando {url}" + (f" (reanudando desde {ya/1e6:.0f} MB)" if ya else ""))
    with requests.get(url, stream=True, timeout=120, headers=cabeceras) as r:
        if ya and r.status_code == 200:
            # el servidor ignoro el Range: hay que empezar de cero
            print("  el servidor no acepta reanudar, se baja completo")
            ya = 0
        elif ya and r.status_code != 206:
            r.raise_for_status()
        else:
            r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) + ya
        bajado = ya
        with open(parcial, "ab" if ya else "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bajado += len(chunk)
                if total:
                    print(f"\r  {bajado/1e6:7.0f} / {total/1e6:.0f} MB", end="", flush=True)
    print()
    os.replace(parcial, ruta)


def abrir_mmap(npz_local):
    """Extrae los .npy internos del .npz a disco y los abre con mmap.

    np.load() sobre el .npz descomprime el arreglo completo en RAM (13 GB a
    224x224). Extraerlos primero mueve ese costo al disco, que en Colab sobra,
    y deja que volcar_split lea por fila.
    """
    salida = {}
    with zipfile.ZipFile(npz_local) as z:
        for clave in ("train_images", "train_labels", "test_images", "test_labels"):
            destino = os.path.join(CACHE, f"{os.path.basename(npz_local)}.{clave}.npy")
            if not (os.path.exists(destino) and os.path.getsize(destino) > 1000):
                print(f"  extrayendo {clave} -> disco")
                with z.open(clave + ".npy") as src, open(destino, "wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 24)
            salida[clave] = np.load(destino, mmap_mode="r")
    return salida


def volcar_split(imagenes, etiquetas, destino, por_clase, semilla):
    rng = np.random.default_rng(semilla)
    if os.path.isdir(destino):
        shutil.rmtree(destino)
    etiquetas = etiquetas.reshape(-1)
    total = 0
    for indice, nombre in enumerate(CLASES):
        posiciones = np.flatnonzero(etiquetas == indice)
        rng.shuffle(posiciones)
        if por_clase is not None:
            posiciones = posiciones[:por_clase]
        carpeta = os.path.join(destino, nombre)
        os.makedirs(carpeta, exist_ok=True)
        for n, pos in enumerate(posiciones):
            Image.fromarray(np.asarray(imagenes[pos])).save(os.path.join(carpeta, f"{nombre}_{n:05d}.png"))
        total += len(posiciones)
        print(f"    {nombre:<22} {len(posiciones):>5} imagenes")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--por-clase", type=int, default=500,
                    help="imagenes por clase en train (default 500; 0 = todas)")
    ap.add_argument("--test-por-clase", type=int, default=150,
                    help="imagenes por clase en test (default 150; 0 = todas)")
    ap.add_argument("--resolucion", type=int, default=128, choices=[28, 64, 128, 224],
                    help="resolucion nativa a descargar (default 128)")
    ap.add_argument("--semilla", type=int, default=42)
    args = ap.parse_args()

    por_clase = args.por_clase or None
    test_por_clase = args.test_por_clase or None

    sufijo = "" if args.resolucion == 28 else f"_{args.resolucion}"
    archivo = f"pathmnist{sufijo}.npz"
    npz_local = os.path.join(CACHE, archivo)

    print("1) Descarga")
    descargar(f"{URL_BASE}/{archivo}", npz_local)

    print("2) Abriendo el .npz por mmap (extrae a disco, no a RAM)")
    datos = abrir_mmap(npz_local)
    train_x, train_y = datos["train_images"], datos["train_labels"]
    test_x, test_y = datos["test_images"], datos["test_labels"]
    print(f"   train: {train_x.shape}  |  test: {test_x.shape}")

    print("3) Escribiendo PNGs")
    print("  train:")
    n_train = volcar_split(train_x, train_y, os.path.join(DESTINO, "train"),
                           por_clase, args.semilla)
    print("  test:")
    n_test = volcar_split(test_x, test_y, os.path.join(DESTINO, "test"),
                          test_por_clase, args.semilla + 1)

    print(f"\nListo -> {DESTINO}")
    print(f"  clases ({len(CLASES)}): {CLASES}")
    print(f"  train: {n_train} imagenes  |  test: {n_test} imagenes")
    print(f"  resolucion nativa: {args.resolucion}x{args.resolucion} "
          f"(el notebook la sube a 256x256 con transforms.Resize)")


if __name__ == "__main__":
    sys.exit(main())
