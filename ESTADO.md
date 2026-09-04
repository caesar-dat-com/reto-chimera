# Estado al 2026-09-04, 10:00

## Grupo 77

| Integrante | Codigo |
|---|---|
| Cesar Armando Reyes Oliveros | 2236379 |
| Juan Pablo Maya | 2236377 |

## Hecho

- **Arquitectura definida y presupuesto validado: 196 / 200 pts.**
  7 bloques sobre 5 familias. Celda 10 del notebook, salida guardada.
  ```
  MI_BLOQUES = ["mobilenet_entrada", "inception_entrada", "inception_intermedia",
                "convnext_intermedia", "mobilenet_intermedia", "resnet_entrada",
                "vgg_entrada"]
  MI_OPTIMIZADOR = "adamw"; MI_SCHEDULER = "cosine"; MI_LOSS = "label_smoothing"
  MI_CONV_CUSTOM = True; MI_GLOBAL_AVG_POOL = True; MI_DROPOUT = ("dropout", 0.3)
  CODIGO_GRUPO = 77
  ```
- **Datos listos.** Intel: 6.600 imagenes (4.800 train / 1.800 test, 6 clases).
  PathMNIST: 5.850 imagenes (4.500 train / 1.350 test, 9 clases).
  Se rearman en cualquier maquina con los dos `scripts/prepare_*.py`.
- **Entorno reproducible.** `./scripts/setup.sh cpu|cuda|rocm`.
- **Notebook limpio.** El `.ipynb` original venia con las salidas de una corrida
  ajena (checkpoints `grupo0_*.pth`); se limpiaron.
- **Entrega parcial armada:** `entrega_parcial_grupo77.ipynb`.

## Corriendo ahora

Entrenamiento de la arquitectura propia sobre Intel, 25 epocas, **en CPU**:
20 min de reloj y 2h33 de CPU-time acumulado, sin terminar.

`scripts/ejecutar_notebook.py` sigue solo hasta el final: al cerrar la
arquitectura propia arranca transfer y despues fine-tuning, y guarda el
notebook celda por celda.

## Falta

Los tres `.pth` y el `comprobante.json`:

| Entregable | Estado |
|---|---|
| `entregas/grupo77_arquitectura_propia.pth` | entrenando |
| `entregas/grupo77_transfer.pth` | pendiente |
| `entregas/grupo77_finetune.pth` | pendiente |
| `.ipynb` con resultados | parcial (hasta la celda 12) |

## El problema abierto: no hay GPU

`torch 2.9.1+rocm6.4` instalado, pero `torch.cuda.is_available()` da `False`
en una RX 6650 XT (gfx1032).

Descartado:

- **Permisos.** `/dev/kfd` y `/dev/dri/renderD128` abren en `O_RDWR`
  (ACL puesto con `setfacl -m u:mark02:rw`). `renderD128` **es** la dGPU, no la
  integrada.
- **El override de arquitectura.** Falla igual con y sin
  `HSA_OVERRIDE_GFX_VERSION=10.3.0`, y con `ROCR_VISIBLE_DEVICES` / `HSA_ENABLE_SDMA=0`.
- **memlock.** `ulimit -l` = 1,9 GB.
- **El kernel ve las GPU.** `/sys/class/kfd/kfd/topology/nodes/1/properties`
  reporta `gfx_target_version 100302` (gfx1032), 56 SIMDs, `drm_render_minor 128`.

El fallo real esta un nivel mas abajo de HIP:

```
hsa_init()           -> 4104  = HSA_STATUS_ERROR_OUT_OF_RESOURCES
hsa_iterate_agents() -> 4107  = HSA_STATUS_ERROR_NOT_INITIALIZED
agentes encontrados: 0
```

Hipotesis viva: la maquina tiene 15 GB y solo **2 GB libres** porque el
entrenamiento en CPU se comio el resto; ROCm reserva memoria al inicializar.
Hay que reprobar el GPU **con la maquina descargada**:

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 .venv/bin/python -c \
  "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

Si sigue en `False`, la otra hipotesis es ROCm 6.4 contra kernel 7.0.11; se
probaria con las ruedas de `rocm6.2` (otra descarga de 4 GB).

## Por que importa el GPU

Referencia de la corrida ajena que venia en el notebook: el transfer learning
le tomo **7.096 s (~2 h)** con 15.300 imagenes de train, y aun asi saco
**F1 = 0.6413**, por debajo del umbral. Nuestro subset es ~4x mas chico, pero
en CPU a 256x256 el transfer y el fine-tuning no son horas sino mas de un dia.

## Como continuar

### Opcion 1: seguir en esta maquina (si aparece el GPU)

```bash
cd /home/mark02/reto_chimera
.venv/bin/python scripts/ejecutar_notebook.py
```

Reentrena desde cero las tres etapas: van en el mismo kernel a proposito,
porque transfer y fine-tuning parten del `modelo_propio` en memoria.

### Opcion 2 (recomendada): Google Colab, T4 gratis

Ya esta armado: **[`colab_run.ipynb`](colab_run.ipynb)** —
[abrir directo en Colab](https://colab.research.google.com/github/caesar-dat-com/reto-chimera/blob/main/colab_run.ipynb).

Entorno de ejecucion -> Cambiar tipo de entorno -> GPU (T4), y luego Ejecutar todas.
El notebook clona el repo, baja los datasets, verifica conteos, parchea el data
loader para GPU, corre las tres etapas, limpia las salidas del profesor que hayan
quedado sin sobrescribir y descarga `grupo77_entrega.zip`.

45-75 min de reloj. No cerrar la pestana: Colab mata las sesiones sin pestana
abierta y el disco de la sesion se borra al desconectar.

### Opcion 3: entregar solo la arquitectura propia

Cortar con `--solo-fase1` y entregar 1 de los 3 componentes.

## Antes de entregar cualquier cosa

```bash
python scripts/limpiar_salidas_ajenas.py --hasta <ultima celda que corrimos> --salida entrega.ipynb
```

Si no, se entregan las salidas del profesor (`grupo0_*.pth`) como propias.
