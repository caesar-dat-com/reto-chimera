# Entrega Grupo 77 — corrida optimizada (128 px / fine-tuning 25 epocas)

Ejecutada de arriba a abajo en Colab T4 el 2026-09-05.

## Cambios respecto a la corrida base (rama `entrega-actual`)

| Parametro | Base | Esta corrida | Motivo |
|---|---|---|---|
| `IMG_SIZE_PATH` (celda 17) | 256 | **128** | PathMNIST es nativo de 128 px; 256 era upsampling puro (4x de computo, cero informacion nueva) |
| `max_epochs` fine-tuning (celda 22) | 10 | **25** | La checklist de la celda 28 exige `MAX_EPOCHS=25` en los tres componentes; con 10 el fine-tuning no habia convergido (f1_val seguia subiendo en la ultima epoca) |

## Resultados

| Componente | F1 macro | Umbral tabla (0.87) |
|---|---|---|
| Arquitectura propia | 0.8697 | N/A (gradual) |
| Transfer learning | 0.4680 | No |
| Fine-tuning | **0.7671** | No |

Presupuesto: 196 / rango 150-200 — VALIDO.

## Comparativa contra la corrida base

| Metrica | Base | Optimizada | Delta |
|---|---|---|---|
| f1_arquitectura_propia | 0.8768 | 0.8697 | -0.0071 |
| f1_transfer | 0.5118 | 0.4680 | -0.0438 |
| f1_finetune | 0.6235 | **0.7671** | **+0.1436** |
| Duracion transfer | 1033 s | 270 s | -74 % |
| Duracion fine-tuning | 1253 s | 768 s | -39 % |
| Epocas fine-tuning | 10 | 25 | +15 |

Ambas corridas completaron 25 epocas de transfer. En la base el fine-tuning solo
corrio 10 y f1_val seguia subiendo (0.6450 -> 0.6769 en la ultima epoca): no habia
convergido. Aqui llega a f1_val=0.8634 con loss_train 0.49 y ya aplanado.

## Advertencia sobre el umbral impreso

Las celdas 20 y 23 imprimen un texto que no coincide con la comparacion que hacen
en codigo (dicen `>= 0.90` / `>= 0.87` pero comparan contra `>= 0.75` / `>= 0.76`).
Por eso la celda 23 imprime `True` con 0.7671. La tabla de la celda 25, que si usa
0.87, marca `False`. **Ningun componente alcanza 0.87 en ninguna de las dos corridas.**
