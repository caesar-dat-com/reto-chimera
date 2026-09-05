# Entrega — Reto Chimera · Grupo 77

| Integrante | Código |
|---|---|
| Cesar Armando Reyes Oliveros | 2236379 |
| Juan Pablo Maya | 2236377 |

Repositorio: <https://github.com/caesar-dat-com/reto-chimera>
Notebook con salidas: [`S3_Reto_Hibridacion_Chimera_tester.ipynb`](S3_Reto_Hibridacion_Chimera_tester.ipynb)

---

## 1. Resultados

Corrida del 2026-09-05 en una Radeon RX 6700S (ROCm 6.2), 25 épocas por
componente, batch 32 (Intel) y 16 (PathMNIST), semilla = código del grupo.

| Componente | F1 macro (test) | Umbral del enunciado | Estado |
|---|---|---|---|
| Arquitectura propia (Intel, desde cero) | **0.8723** | ≥0.85 → 4 · ≥0.95 → 5 | entregado |
| Transfer learning (PathMNIST, backbone congelado) | **0.4330** | ≥0.77 → 3 · ≥0.86 → 5 | entregado, bajo umbral |
| Fine-tuning (PathMNIST, red completa) | — | ≥0.78 → 3 · ≥0.88 → 5 | **no terminó** |

Entregables en `entregas/`:

| Archivo | Épocas | Duración | fingerprint_sha256 |
|---|---|---|---|
| `grupo77_arquitectura_propia.pth` | 25 | 36 min | `70a313506b7a4298…` |
| `grupo77_transfer.pth` | 25 | 148 min | `4147276a287d21fe…` |
| `grupo77_finetune.pth` | — | — | falta |
| `grupo77_comprobante.json` | — | — | falta (lo escribe la última celda, que depende del fine-tuning) |

### Por qué falta el fine-tuning

Se entrenó 3 h 30 de las ~10 épocas y se detuvo antes de cerrar. Dos motivos, en
orden de importancia:

1. **La etapa es cara en este hardware.** Descongelar la red completa a 256×256
   sale ~3× por época que el transfer con el backbone congelado. Con 3.825
   imágenes de train y batch 16 son ~15 min por época.
2. **Un primer intento murió por el timeout de nbclient** (4 h por celda, que es
   el default). Ese corte se llevó el kernel y con él `modelo_propio`, que solo
   vive en memoria. De ahí salió `scripts/reanudar_finetune.py`, que retoma
   desde `grupo77_transfer.pth` en vez de repetir las 3 h de las etapas 1 y 2.

Para cerrarlo, con el repo tal como está:

```bash
./scripts/setup.sh rocm    # o cuda
python scripts/reanudar_finetune.py     # ~2.5 h de GPU, no repite etapas 1 ni 2
python scripts/empaquetar_entrega.py    # limpia, verifica y arma el zip
python scripts/auditar_entrega.py       # 13 comprobaciones contra el enunciado
```

---

## 2. Justificación de la arquitectura (objetivo de aprendizaje #1)

Configuración elegida — **196 / 200 pts**, candidata `mixto_barato_ancho`:

```python
MI_BLOQUES = ["mobilenet_entrada", "inception_entrada", "inception_intermedia",
              "convnext_intermedia", "mobilenet_intermedia", "resnet_entrada",
              "vgg_entrada"]
MI_OPTIMIZADOR = "adamw"           # 6
MI_SCHEDULER   = "cosine"          # 5
MI_LOSS        = "label_smoothing" # 4
MI_CONV_CUSTOM = True              # 5
MI_GLOBAL_AVG_POOL = True          # 3
MI_DROPOUT = ("dropout", 0.3)      # 2
```

### El criterio: puntos por parámetro, no gusto por una familia

El costo en puntos del catálogo es proporcional a los parámetros reales de cada
bloque a 64 canales. Eso permite ordenar los bloques por lo que *rinde cada
punto*, que es exactamente lo que pide el objetivo #1:

| Bloque elegido | Puntos | Parámetros | Qué aporta |
|---|---:|---:|---|
| MobileNet entrada (depthwise separable) | 8 | 5.056 | profundidad casi gratis |
| Inception entrada (bottleneck agresivo) | 10 | 9.352 | multi-escala barato |
| Inception intermedia (bottleneck suave) | 20 | 29.712 | multi-escala con más ancho |
| ConvNeXt intermedia (expansión ×4) | 25 | 36.480 | kernel grande, campo receptivo |
| MobileNet intermedia (residual invertido) | 32 | 55.104 | residual + expansión |
| ResNet entrada (BasicBlock) | 38 | 74.112 | salto de identidad, gradiente sano |
| VGG entrada (2 conv 3×3) | 38 | 74.112 | filtros densos al final |
| **Total bloques** | **171** | **283.928** | 7 bloques, 6 familias |

Modelo completo: **331.678 parámetros entrenables**.

Las tres decisiones que se derivan de esa tabla:

1. **Siete bloques baratos antes que tres caros.** Con 171 pts se podían comprar
   tres bloques *intermedia* de VGG/ResNet (50 pts cada uno, 111.168 parámetros
   cada uno) y quedarse en tres niveles de profundidad. Comprando barato se
   consiguen **siete** niveles por los mismos puntos: más composición jerárquica,
   que es lo que sirve en 6 clases de paisajes donde la textura importa más que
   el detalle fino. El nombre interno del candidato — `mixto_barato_ancho` — es
   literalmente eso.
2. **VGG y ResNet cuestan igual; la conexión de salto es gratis.** A igualdad de
   puntos y de parámetros (38 pts / 74.112 en la variante *entrada*), el bloque
   residual entrega además un camino directo para el gradiente. Por eso entra
   ResNet, y VGG solo una vez y al final, donde ya hay pocos píxeles y los
   filtros densos son baratos en cómputo aunque no en parámetros.
3. **Los 15 pts de los componentes rinden más que un bloque más.** `AdamW` (6),
   `cosine` (5) y `label_smoothing` (4) suman 15 pts: menos que medio bloque
   *intermedia*, y actúan sobre las 25 épocas completas. El `global_avg_pool` (3
   pts) además **evita** el aplanado del mapa completo, que dispararía los
   parámetros de la cabeza sin costar un solo punto del presupuesto — el
   presupuesto no lo mide, pero la RAM sí.

El presupuesto queda en 196 de 200: 4 pts de margen, deliberado. Subir a
`vgg_intermedia` en vez de `vgg_entrada` costaría 12 pts más y se pasaría del
límite, que es motivo de no calificación.

### Qué dicen los resultados

- **0.8723 en Intel** con 331.678 parámetros y 25 épocas: la red aprende. La
  matriz de confusión concentra el error en `glacier` ↔ `mountain` (0.82/0.81 de
  F1), que es el par visualmente ambiguo del dataset; `forest` sale en 0.97.
- **0.4330 en transfer** contra 0.5448 en validación. La caída val→test no es
  sobreajuste al backbone: el test de PathMNIST viene de **otro centro clínico**
  (CRC-VAL-HE-7K), o sea otro tinte y otro escáner. Es el escenario duro a
  propósito.
- La corrida de referencia del profesor, con **4× más imágenes** (15.300 de
  train contra nuestras 3.825), sacó 0.6413 en la misma etapa. O sea: el techo
  del transfer aquí lo pone el tamaño del subset y el salto de dominio
  (paisajes a 64 px → histología a 256 px), no la arquitectura.

---

## 3. Checklist del enunciado

- [x] `CODIGO_GRUPO` reemplazado por el código real del grupo (77).
- [x] `calcular_presupuesto` corrió sin `AssertionError`: 196 pts, dentro de 150–200.
- [x] Ningún componente pasó de `MAX_EPOCHS = 25` (verificado contando el
      historial real dentro de cada `.pth`, no la metadata).
- [ ] Los tres `.pth` y el `comprobante.json` — **faltan el de fine-tuning y el
      comprobante**.
- [x] El notebook corre de arriba a abajo sin editar celdas fuera de las
      marcadas `TODO (grupo)`.

### Lo único que se tocó fuera de las celdas `TODO`

`NUM_WORKERS = 0 → 4`, `PIN_MEMORY = False → True` y `cudnn.benchmark = True`,
todo dentro de la celda de configuración. Son parámetros del *data loader*: no
cambian pesos, épocas, arquitectura, semilla ni batch size. Sin eso la GPU se
queda esperando al disco. `chimera_blocks.py` no se tocó — el auditor compara su
md5 contra el del profesor en cada corrida.

**Los batch size no se subieron a propósito**, aunque sobraba VRAM: el
`fingerprint_sha256` se calcula sobre el primer batch del test loader y ese batch
no viaja dentro del `.pth`. Con otro tamaño de batch, el hash que recalcule el
profesor no coincidiría y el checkpoint parecería adulterado.

---

## 4. Cómo verificar esta entrega

```bash
python scripts/auditar_entrega.py
```

Comprueba, sin creerle a ninguna salida guardada: el md5 de `chimera_blocks.py`,
que solo se editaron las celdas permitidas, que ninguna celda cambió de tipo o
de orden, el presupuesto recalculado desde cero, las épocas contadas del
historial de cada checkpoint, que los pesos encajen en la arquitectura que
declara su propia config, que el comprobante coincida con los `.pth` y con los
F1 impresos, y que no queden salidas del profesor (`grupo0_*`) en el notebook.
