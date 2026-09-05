# Estado al 2026-09-05, 19:30

Resultados y justificación de la entrega: **[ENTREGA.md](ENTREGA.md)**.

## Grupo 77

| Integrante | Codigo |
|---|---|
| Cesar Armando Reyes Oliveros | 2236379 |
| Juan Pablo Maya | 2236377 |

## Donde quedo

Dos de los tres componentes entrenados **en GPU** y guardados:

| Entregable | Estado | F1 macro (test) |
|---|---|---|
| `entregas/grupo77_arquitectura_propia.pth` | listo, 25 epocas, 36 min | 0.8723 |
| `entregas/grupo77_transfer.pth` | listo, 25 epocas, 148 min | 0.4330 |
| `entregas/grupo77_finetune.pth` | **falta** | — |
| `entregas/grupo77_comprobante.json` | **falta** (lo escribe la ultima celda) | — |
| `.ipynb` con salidas | celdas 0..20, todas de esta corrida | — |

El fine-tuning corrio 3 h 30 de sus ~10 epocas y se detuvo antes de cerrar.

## El GPU: resuelto

`torch 2.9.1+rocm6.4` daba `hsa_init() -> 4104` y `cuda.is_available() False`.
**Con las ruedas de rocm6.2 funciona**: `torch 2.5.1+rocm6.2` ve la GPU. El
entorno bueno es `/home/caesar/chimera/.venv` y hay que correr **como caesar**,
que es quien esta en el grupo `render` (mark02 no tiene acceso a `/dev/kfd`):

```bash
sudo -u caesar bash -c 'cd /home/caesar/chimera/repo && \
  HSA_OVERRIDE_GFX_VERSION=10.3.0 HIP_VISIBLE_DEVICES=0 \
  .venv/bin/python -u scripts/reanudar_finetune.py'
```

Referencia de tiempos en la RX 6700S: arquitectura propia 36 min, transfer
148 min, fine-tuning ~15 min por epoca.

## Tres trampas que costaron horas

1. **El kernel manda, no el interprete.** `nbclient` resolvia el kernelspec
   `python3` al primero que encontrara — el de otro venv con rocm6.4 sin GPU —
   mientras el python que lanzaba el script si veia la GPU. El notebook entrenaba
   en CPU y el log decia `Dispositivo: cuda` de una corrida anterior.
   `ejecutar_notebook.py` ahora registra un kernelspec temporal apuntando a
   `sys.executable`.
2. **El timeout por celda de nbclient son 4 h.** El fine-tuning no cabe. Con el
   corte se va el kernel y con el `modelo_propio`, que solo vive en memoria:
   volver al mismo punto costaba 3 h de reentrenar las etapas 1 y 2.
   `reanudar_finetune.py` retoma desde `grupo77_transfer.pth`.
3. **Comparar salidas byte a byte para detectar las del profesor borra las
   propias.** Las celdas deterministas (tablas de costos, conteos del dataset)
   imprimen lo mismo en cualquier corrida. El criterio ahora es la marca de
   tiempo de `metadata.execution`.

## Como continuar

```bash
python scripts/reanudar_finetune.py     # ~2.5 h, no repite etapas 1 ni 2
python scripts/empaquetar_entrega.py    # limpia salidas ajenas y arma el zip
python scripts/auditar_entrega.py       # 13 comprobaciones contra el enunciado
```

Alternativa sin GPU local: **[`colab_run.ipynb`](colab_run.ipynb)** en una T4
([abrir en Colab](https://colab.research.google.com/github/caesar-dat-com/reto-chimera/blob/main/colab_run.ipynb)).
Ojo: corre las tres etapas desde cero, no reanuda.

## La corrida duplicada en CPU: cerrada

Habia una segunda corrida del mismo notebook en `/home/mark02/reto_chimera`
(`run_cpu.sh`), que alguna sesion relanzaba sola cada vez que se mataba. Iba en
CPU — ese venv tiene `torch+rocm6.4`, que no ve la GPU, y mark02 tampoco esta en
el grupo `render` — y se comia los 16 hilos, asi que frenaba el dataloader de la
corrida buena. Terminada el 2026-09-05 14:35; su ultimo log cierra en
`DeadKernelError: Kernel died`.

Lo unico que alcanzo a producir fue su propio
`entregas/grupo77_arquitectura_propia.pth`. **No sirve para completar esta
entrega**: son pesos de otra corrida, y el `fingerprint_sha256` ata cada
checkpoint a unos pesos concretos. Mezclar archivos de dos corridas da un
comprobante que no cuadra con ninguna.
