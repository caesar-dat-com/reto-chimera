# Reto de Hibridación de Arquitecturas CNN (Chimera)

Notebook del reto + los dos scripts que faltaban para que corra de arriba a
abajo sin explotar. El enunciado original asume que el profesor ya dejó los
datasets preparados; aquí se descargan solos, sin cuenta de Kaggle.

## Arranque (una sola vez)

```bash
git clone <URL-DE-ESTE-REPO> reto-chimera
cd reto-chimera
./scripts/setup.sh cpu     # o: cuda (NVIDIA) / rocm (AMD)
```

Eso crea el `.venv`, instala PyTorch para tu hardware, y baja los dos datasets
a `./data/`. Tarda un rato la primera vez (~1,4 GB de descarga).

Luego:

```bash
.venv/bin/jupyter lab S3_Reto_Hibridacion_Chimera_tester.ipynb
```

## Qué hay que editar en el notebook

Solo las celdas marcadas `TODO (grupo)`:

1. `CODIGO_GRUPO = 0` → el código real del grupo (es la semilla del split y el
   nombre de los `.pth`).
2. `MI_BLOQUES = ["capa1", "capa2", "capa3"]` → **esto revienta tal cual**, son
   nombres inventados. Hay que poner nombres reales del catálogo de
   `chimera_blocks.py` (`BLOCK_COSTS`) y que el total quede entre 150 y 200 pts.

Ejemplo válido (177 pts):

```python
MI_BLOQUES = ["resnet_entrada", "resnet_intermedia", "densenet_entrada",
              "convnext_intermedia", "mobilenet_entrada"]
MI_OPTIMIZADOR = "adamw"        # 6
MI_SCHEDULER   = "cosine"       # 5
MI_LOSS        = "label_smoothing"  # 4
MI_CONV_CUSTOM = False
MI_POOLING_EXTRA = None
MI_GLOBAL_AVG_POOL = True       # 3
MI_DROPOUT = ("dropout", 0.3)   # 2
```

## Entregables que produce

Al terminar el notebook, en `./entregas/`:

| Archivo | Qué es |
|---|---|
| `grupo<N>_arquitectura_propia.pth` | modelo entrenado desde cero en Intel |
| `grupo<N>_transfer.pth` | transfer learning sobre PathMNIST (backbone congelado) |
| `grupo<N>_finetune.pth` | fine-tuning completo sobre PathMNIST |
| `grupo<N>_comprobante.json` | F1 de los tres + fingerprints |

Más el propio `.ipynb` ejecutado, con las salidas visibles.

## Datasets

| Script | Dataset | Fuente | Descarga |
|---|---|---|---|
| `scripts/prepare_intel_dataset.py` | Intel Image Classification, 6 clases | espejo en HuggingFace | ~363 MB |
| `scripts/prepare_pathmnist_dataset.py` | PathMNIST, 9 clases | MedMNIST oficial (Zenodo) | ~1 GB |

Por defecto arman subsets (800 img/clase Intel train, 500 PathMNIST) para que
el entrenamiento quepa en un portátil. Se ajusta con `--por-clase`.

**Nota de resolución:** PathMNIST se baja en 64×64 y el notebook la sube a
256×256 con `transforms.Resize`, tal como el enunciado ya hacía desde 224×224.
La variante nativa de 224 pesa 12 GB y su arreglo de train no cabe en RAM de
14 GB (un `.npz` no se puede leer parcialmente). Con
`--resolucion 128` se usa la de 128×128 si tienes RAM de sobra.

## Gotchas conocidos del notebook original

- `MI_BLOQUES` trae nombres placeholder que no existen en el catálogo → `ValueError`.
- El fine-tuning corre con `max_epochs=10` pero guarda el checkpoint diciendo
  `epocas_entrenadas=MAX_EPOCHS` (25). Es un dato de metadata inconsistente.
- Las celdas de umbral imprimen textos que no coinciden con la comparación
  (dice `>= 0.90` y compara contra `0.75`).
- Si `MI_GLOBAL_AVG_POOL = False`, la cabeza es `LazyLinear`: el forward de
  verificación **tiene** que correr antes de crear el optimizador. El notebook
  ya lo hace, no muevas ese orden.
- `data/`, `entregas/` y `.venv/` están en `.gitignore`: nadie sube 1,4 GB al repo.
