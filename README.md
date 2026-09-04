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
