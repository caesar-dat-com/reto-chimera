# Informe — Corrida A (configuración original del notebook)

**Grupo:** 77
**Configuración:** `mixto_barato_ancho` (196 puntos)
**Hardware:** Google Colab, Tesla T4 (15 GB)
**Notebook:** `S3_Reto_Hibridacion_Chimera_tester.ipynb`

## Configuración utilizada

Bloques: mobilenet_entrada, inception_entrada, inception_intermedia, convnext_intermedia, mobilenet_intermedia, resnet_entrada, vgg_entrada

Optimizador: AdamW · Scheduler: coseno · Loss: label smoothing · Dropout: 0.3 · Global avg pool

Presupuesto: 196 / 200 puntos (rango válido 150–200)

## Resultados

| Componente | F1 macro | ¿Cumple 0.87? |
|---|---|---|
| Arquitectura propia | 0.8768 | Sí |
| Transfer learning | 0.5118 | No |
| Fine-tuning | 0.6235 | No |

## Lo que hice

Armé una arquitectura tipo Chimera combinando siete bloques preentrenados (MobileNet, Inception, ConvNeXt, ResNet y VGG) con una capa de convolución propia y global average pooling. La entrené desde cero sobre Intel Image Classification (4800 imágenes de entrenamiento, 1800 de prueba, 6 clases) durante 25 épocas con AdamW, scheduler coseno y label smoothing.

Después cargué ese modelo sobre PathMNIST (histopatología, 9 clases) en dos etapas: primero transfer learning con el backbone congelado, luego fine-tuning descongelando toda la red con learning rates diferenciados (1e-5 para el backbone, 1e-4 para la cabeza).

## Lo que aprendí

**1. Una arquitectura híbrida entrenada desde cero puede ser muy competitiva.** El modelo propio alcanzó 0.8768 en Intel, por encima del umbral. Combinar bloques de distintas familias (MobileNet por eficiencia, Inception por multi-escala, ConvNeXt por modernidad) dio una representación rica sin necesidad de preentrenamiento en este dataset.

**2. Transfer learning con backbone congelado fracasa cuando el dominio cambia mucho.** El modelo venía entrenado en paisajes naturales (montañas, playas, edificios) y se evaluó en imágenes de histopatología. Las features que aprendió no transferían: 0.5118 y estancado desde la época 16. Congelar el backbone no sirve cuando lo que aprendió no tiene nada que ver con la tarea nueva.

**3. El fine-tuning no terminó de converger.** Se ejecutaron 10 épocas y el F1 todavía iba subiendo (0.6450 → 0.6769 en las últimas). Cortar ahí da una impresión engañosa: parece que el fine-tuning "tampoco funciona", cuando en realidad no se le dio tiempo suficiente. Esto fue lo que me motivó a hacer una segunda corrida con más épocas.

**4. La resolución importa y no siempre más es mejor.** PathMNIST viene nativo a 128 px y el notebook lo subía a 256 px. Eso cuadruplicaba el cómputo sin añadir información real — era interpolación de píxeles inventados. En la segunda corrida bajé a 128 px y el fine-tuning convergió mejor y más rápido.

## Checkpoint

- `grupo77_arquitectura_propia.pth`
- `grupo77_transfer.pth`
- `grupo77_finetune.pth`
- `grupo77_comprobante.json`

Presupuesto: 196 pts · F1 arq propia: 0.8768 · F1 transfer: 0.5118 · F1 fine-tuning: 0.6235