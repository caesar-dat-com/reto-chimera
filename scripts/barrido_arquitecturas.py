"""
barrido_arquitecturas.py

Prueba varias configuraciones de ChimeraNet con pocas epocas sobre Intel, para
decidir cual vale la pena entrenar en serio (25 epocas) como entrega.

No produce entregables: es exploracion. Solo valida presupuesto (150-200 pts)
y reporta F1 macro en validacion.

Uso:
    python scripts/barrido_arquitecturas.py --epocas 5
    python scripts/barrido_arquitecturas.py --epocas 3 --solo 0,1,2   # candidatos sueltos
"""

import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

from chimera_blocks import (  # noqa: E402
    calcular_presupuesto, ChimeraNet, CRITERIONS, OPTIMIZERS,
    construir_scheduler, evaluar_f1,
)

# Candidatos: cada uno es una estrategia de diseno distinta, no variaciones
# cosmeticas. La idea es comparar filosofias (denso vs residual vs barato-y-
# ancho), no hacer grid search ciego.
CANDIDATOS = [
    {
        "nombre": "residual_profundo",
        "bloques": ["resnet_entrada", "resnet_intermedia", "resnet_intermedia"],
        "optimizador": "adamw", "scheduler": "cosine", "loss": "label_smoothing",
        "conv_custom": False, "pooling_extra": None, "gap": True,
        "dropout": ("dropout", 0.3),
    },
    {
        "nombre": "denso_reutiliza",
        "bloques": ["densenet_entrada", "densenet_intermedia", "densenet_entrada",
                    "densenet_entrada", "convnext_entrada"],
        "optimizador": "adamw", "scheduler": "cosine", "loss": "label_smoothing",
        "conv_custom": False, "pooling_extra": None, "gap": True,
        "dropout": ("dropout_espacial", 0.2),
    },
    {
        "nombre": "mixto_barato_ancho",
        "bloques": ["mobilenet_entrada", "inception_entrada", "inception_intermedia",
                    "convnext_intermedia", "mobilenet_intermedia", "resnet_entrada",
                    "vgg_entrada"],
        "optimizador": "adamw", "scheduler": "cosine", "loss": "label_smoothing",
        "conv_custom": True, "pooling_extra": None, "gap": True,
        "dropout": ("dropout", 0.3),
    },
    {
        "nombre": "vgg_clasico",
        "bloques": ["vgg_entrada", "vgg_intermedia", "resnet_intermedia"],
        "optimizador": "sgd", "scheduler": "steplr", "loss": "crossentropy",
        "conv_custom": True, "pooling_extra": None, "gap": True,
        "dropout": ("dropout", 0.3),
    },
    {
        "nombre": "convnext_moderno",
        "bloques": ["convnext_entrada", "convnext_intermedia", "convnext_intermedia",
                    "resnet_intermedia", "mobilenet_intermedia"],
        "optimizador": "adamw", "scheduler": "cosine", "loss": "focal",
        "conv_custom": False, "pooling_extra": None, "gap": True,
        "dropout": ("dropout", 0.2),
    },
]


def cargar_datos(img_size, batch_size, semilla, workers):
    data_dir = os.path.join(RAIZ, "data", "intel_subset")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Falta {data_dir}. Corre antes: python scripts/prepare_intel_dataset.py")

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tr_train = transforms.Compose([
        transforms.Resize((img_size, img_size)), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(mean, std)])
    tr_eval = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(), transforms.Normalize(mean, std)])

    full = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=tr_train)
    full_eval = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=tr_eval)

    gen = torch.Generator().manual_seed(semilla)
    n_val = int(0.15 * len(full))
    idx = torch.randperm(len(full), generator=gen).tolist()
    train_ds = torch.utils.data.Subset(full, idx[:len(full) - n_val])
    val_ds = torch.utils.data.Subset(full_eval, idx[len(full) - n_val:])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    return train_loader, val_loader, len(full.classes)


def probar(cfg, train_loader, val_loader, num_classes, epocas, device):
    total, valido, _ = calcular_presupuesto(
        cfg["bloques"], cfg["optimizador"], cfg["scheduler"], cfg["loss"],
        usar_conv_custom=cfg["conv_custom"], pooling_extra=cfg["pooling_extra"],
        usar_global_avg_pool=cfg["gap"], dropout=cfg["dropout"], verbose=False)
    if not valido:
        print(f"  [{cfg['nombre']}] {total} pts -> FUERA DE RANGO, se salta")
        return {"nombre": cfg["nombre"], "puntos": total, "valido": False}

    modelo = ChimeraNet(
        cfg["bloques"], num_classes=num_classes, usar_conv_custom=cfg["conv_custom"],
        pooling_extra=cfg["pooling_extra"], usar_global_avg_pool=cfg["gap"],
        dropout=cfg["dropout"]).to(device)

    with torch.no_grad():
        x, _ = next(iter(train_loader))
        modelo(x.to(device))

    n_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    optimizador = OPTIMIZERS[cfg["optimizador"]](modelo.parameters())
    scheduler = construir_scheduler(cfg["scheduler"], optimizador, max_epochs=epocas)
    criterion = CRITERIONS[cfg["loss"]]()

    inicio = time.time()
    mejor_f1 = 0.0
    for epoca in range(1, epocas + 1):
        modelo.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizador.zero_grad()
            loss = criterion(modelo(x), y)
            loss.backward()
            optimizador.step()
        f1, _, _ = evaluar_f1(modelo, val_loader, device)
        mejor_f1 = max(mejor_f1, f1)
        if scheduler is not None:
            scheduler.step(f1) if cfg["scheduler"] == "plateau" else scheduler.step()
        print(f"  [{cfg['nombre']}] epoca {epoca}/{epocas}  f1_val={f1:.4f}")

    duracion = time.time() - inicio
    print(f"  [{cfg['nombre']}] {total} pts | {n_params:,} params | "
          f"mejor f1_val={mejor_f1:.4f} | {duracion/60:.1f} min")
    return {"nombre": cfg["nombre"], "puntos": total, "valido": True,
            "parametros": n_params, "mejor_f1_val": round(mejor_f1, 4),
            "minutos": round(duracion / 60, 2), "config": cfg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epocas", type=int, default=5)
    ap.add_argument("--img", type=int, default=64)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--solo", type=str, default=None,
                    help="indices separados por coma, ej: 0,2,4")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device} | epocas por candidato: {args.epocas}")

    train_loader, val_loader, num_classes = cargar_datos(
        args.img, args.batch, args.semilla, args.workers)

    candidatos = CANDIDATOS
    if args.solo:
        indices = [int(i) for i in args.solo.split(",")]
        candidatos = [CANDIDATOS[i] for i in indices]

    resultados = []
    for cfg in candidatos:
        print(f"\n=== {cfg['nombre']} ===")
        resultados.append(probar(cfg, train_loader, val_loader, num_classes,
                                 args.epocas, device))

    validos = [r for r in resultados if r.get("valido")]
    validos.sort(key=lambda r: r["mejor_f1_val"], reverse=True)
    print("\n=== RANKING (F1 macro en validacion) ===")
    print(f"{'candidato':<24}{'pts':>6}{'params':>12}{'f1_val':>10}{'min':>8}")
    for r in validos:
        print(f"{r['nombre']:<24}{r['puntos']:>6}{r['parametros']:>12,}"
              f"{r['mejor_f1_val']:>10.4f}{r['minutos']:>8.1f}")

    salida = os.path.join(RAIZ, "barrido_resultados.json")
    with open(salida, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    print(f"\nGuardado en {salida}")


if __name__ == "__main__":
    sys.exit(main())
