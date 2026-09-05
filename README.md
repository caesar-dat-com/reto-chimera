# Reto de Hibridación de Arquitecturas CNN (Chimera) — Grupo 77

> **Estado actual del trabajo: [ESTADO.md](ESTADO.md)**

> **¿Sin GPU? Corre todo en Colab de un click:**
> [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/caesar-dat-com/reto-chimera/blob/main/colab_run.ipynb)
> — `colab_run.ipynb` clona el repo, baja los datasets, corre las tres etapas en una
> T4 y descarga el zip con los entregables. 45-75 min. Ver [abajo](#corrida-en-colab).

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

## Corrida en Colab

`colab_run.ipynb` hace de arriba a abajo lo que en local haría `setup.sh` +
`ejecutar_notebook.py`, pero sobre la T4 gratuita:

1. Verifica que el entorno tenga GPU (aborta con instrucciones si no).
2. Clona el repo e instala `nbclient` / `nbformat` (Colab ya trae torch+CUDA).
3. Baja y arma los dos datasets, y **verifica los conteos en disco** antes de gastar
   una hora de GPU.
4. Parchea el notebook para GPU: `NUM_WORKERS=2`, `PIN_MEMORY=True`,
   `cudnn.benchmark=True`. **No toca batch sizes, épocas, arquitectura ni semilla**
   (ver abajo por qué).
5. Corre `scripts/ejecutar_notebook.py` — tres etapas, un solo kernel.
6. Compara las salidas contra la versión de git y **vacía las que sigan siendo del
   profesor**, o sea las celdas que no llegaron a correr.
7. Empaqueta `grupo77_entrega.zip` (3 `.pth` + `comprobante.json` + notebook
   ejecutado + `chimera_blocks.py`) y lo descarga.

**Por qué no se sube el batch size en Colab.** Con 16 GB de VRAM cabría un batch
mayor y el transfer iría más rápido, pero `fingerprint_sha256` se calcula sobre el
primer batch del test loader (`BATCH_FIJO_PATH`), y ese batch **no se guarda dentro
del `.pth`**. Si el profesor recalcula el fingerprint con su loader de 16 y el nuestro
fue de 32, los hashes no coinciden y el checkpoint parece adulterado. `BATCH_SIZE = 32`
(Intel) y `BATCH_SIZE_PATH = 16` (PathMNIST) se quedan como están.

**Cuidado con la pestaña.** Colab desconecta las sesiones sin pestaña abierta; el disco
de la sesión se borra al desconectar. Hay que descargar el zip antes de cerrar (o
descomentar la celda final de respaldo en Drive).

## Datasets

| Script | Dataset | Fuente | Descarga |
|---|---|---|---|
| `scripts/prepare_intel_dataset.py` | Intel Image Classification, 6 clases | espejo en HuggingFace | ~363 MB |
| `scripts/prepare_pathmnist_dataset.py` | PathMNIST, 9 clases | MedMNIST oficial (Zenodo) | ~1 GB |

Por defecto arman subsets (800 img/clase Intel train, 500 PathMNIST) para que
el entrenamiento quepa en un portátil. Se ajusta con `--por-clase`.

**Nota de resolución:** `--resolucion` elige qué variante nativa de MedMNIST se
baja (28/64/128/224); el default es **128**. El notebook siempre sube a 256×256
con `transforms.Resize`, igual que el enunciado, que parte de 224×224.

La corrida entregada se armó con **`--resolucion 224`**, o sea la máxima nativa:
es exactamente el camino que describe el enunciado (224 nativo → upsample a 256)
y no mete borrosidad extra. A 64×64 el transfer se quedaba en F1 0.4873 porque
las imágenes de histología llegaban lavadas al `Resize((256,256))`.

El `.npz` de 224 pesa 12 GB y no cabe en 14 GB de RAM leyéndolo entero; el script
lo abre con `mmap` y extrae solo las imágenes que necesita, así que no hace falta
RAM de sobra.

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

## Reparto de trabajo (3 personas)

Ojo con el orden: los tres componentes son **secuenciales** — transfer y
fine-tuning parten del `modelo_propio` ya entrenado en Intel. Y solo **una**
corrida genera los `.pth` que se entregan (el `fingerprint_sha256` ata cada
archivo a unos pesos concretos). O sea: no se reparte "una persona por
componente"; se reparte exploración, y una sola máquina hace la corrida final.

| Rol | Quién | Qué hace | Entrega interna |
|---|---|---|---|
| **A — corrida oficial** | el equipo con la mejor GPU | correr el notebook completo con la config ganadora, 25 épocas | los 3 `.pth` + `comprobante.json` + `.ipynb` ejecutado |
| **B — barrido de arquitecturas** | 1 persona | `python scripts/barrido_arquitecturas.py --epocas 5` y reportar el ranking | `barrido_resultados.json` en un PR |
| **C — transfer/fine-tuning + informe** | 1 persona | probar lr y épocas de las secciones 7 y 8 sobre un `modelo_propio` de 5 épocas; arreglar los bugs del notebook; armar las gráficas y la justificación de por qué esos bloques | celdas de gráficas + texto de justificación en un PR |

Reglas para no pisarse:

- Nadie trabaja en `main`. Rama por persona: `git checkout -b barrido-<nombre>`.
- `data/` y `entregas/` están en `.gitignore`. No suban datasets ni `.pth` al repo.
- El `.ipynb` genera conflictos horribles en git. **Solo A edita el notebook.**
  B y C mandan sus cambios como scripts o como celdas pegables en el PR.
- `CODIGO_GRUPO` lo fija A y es el mismo para la entrega final.

### B: barrido de arquitecturas

Compara 5 filosofías de diseño distintas (residual profundo, denso,
mixto-barato-ancho, VGG clásico, ConvNeXt moderno), todas dentro del
presupuesto. Con 5 épocas basta para ordenarlas.

```bash
python scripts/barrido_arquitecturas.py --epocas 5
```

En CPU tarda; si va muy lento, `--solo 0,2,4` corre solo tres candidatos.
Lo que hay que reportar: el ranking de F1 en validación, los puntos que gasta
cada uno y los parámetros reales. Eso es literalmente el objetivo de
aprendizaje #1 del enunciado ("justificar en términos de costo computacional
real, no intuición").

### C: transfer, fine-tuning e informe

1. Entrenar un `modelo_propio` rápido (5 épocas) solo para tener un backbone.
2. Barrer la sección 7: lr de la cabeza (`1e-3` vs `3e-3`), y cuántas épocas
   antes de que se estanque.
3. Barrer la sección 8: lr del backbone (`1e-5` vs `1e-4`) y si conviene
   descongelar todo o solo la parte final de `features`.
4. Arreglar los bugs listados arriba en "Gotchas".
5. Gráficas: curva de `historial["val_f1"]` por época de los tres componentes,
   y matriz de confusión del test. Van como celdas nuevas al final.

## Aviso: el notebook original trae salidas del profesor

El `.ipynb` tal como lo repartieron **ya venía ejecutado por él**: las celdas 14,
15, 17, 19 y 20 traen sus resultados, sus tiempos y sus checkpoints
`grupo0_*.pth`. Entregarlo sin limpiar equivale a presentar sus números como
propios.

Para dejar solo lo que este grupo corrió de verdad:

```bash
python scripts/limpiar_salidas_ajenas.py --hasta 12 --salida entrega.ipynb
```

`--hasta N` conserva las salidas de las celdas `0..N` y vacía el resto.

De paso, sus salidas sirven de referencia de cuánto cuesta esto:

| Dato de la corrida del profesor | Valor |
|---|---|
| F1 arquitectura propia | 0.9005 |
| F1 transfer | 0.6413 (por debajo del umbral) |
| Duración del transfer | 7.096 s ≈ 2 h |
| Su subset de PathMNIST | 15.300 train / 2.700 val / 4.260 test |

O sea: el transfer a 256×256 son ~2 h **en su máquina**, y aun así no llegó al
umbral. Es la etapa cara del reto.
