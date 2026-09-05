# Informe — Corrida B (configuración optimizada)

**Grupo:** 77
**Configuración:** `mixto_barato_ancho` (196 puntos)
**Hardware:** Google Colab, Tesla T4 (15 GB)
**Notebook:** `S3_Reto_Hibridacion_Chimera_tester.ipynb`

## Cambios respecto al notebook original

Hice dos ajustes al notebook del profesor, ambos justificados:

**1. `max_epochs` del fine-tuning: 10 → 25.** El notebook venía con el fine-tuning limitado a 10 épocas, pero la checklist de entrega (celda 28) pide explícitamente que los tres componentes se entrenen con `MAX_EPOCHS=25`. En la primera corrida dejé las 10 y el modelo no convergió — el F1 seguía subiendo cuando se cortó. Subir a 25 pone esta etapa en cumplimiento con lo que pide el enunciado.

**2. `IMG_SIZE_PATH`: 256 → 128.** PathMNIST se distribuye nativamente a 128 px. El notebook original lo escalaba a 256 px, lo que cuadruplicaba el cómputo sin añadir información — era interpolación de píxeles inventados. Bajar a la resolución nativa reduce el tiempo de entrenamiento sin perder señal.

## Configuración utilizada

Bloques: mobilenet_entrada, inception_entrada, inception_intermedia, convnext_intermedia, mobilenet_intermedia, resnet_entrada, vgg_entrada

Optimizador: AdamW · Scheduler: coseno · Loss: label smoothing · Dropout: 0.3 · Global avg pool

Presupuesto: 196 / 200 puntos (rango válido 150–200)

## Resultados

| Componente | F1 macro | ¿Cumple 0.87? |
|---|---|---|
| Arquitectura propia | 0.8697 | No (por poco) |
| Transfer learning | 0.4680 | No |
| Fine-tuning | 0.7671 | No |

## Lo que hice

Diseñé una arquitectura Chimera combinando siete bloques preentrenados de distintas familias (MobileNet, Inception, ConvNeXt, ResNet y VGG) con una capa de convolución propia y global average pooling. La entrené desde cero sobre Intel Image Classification (4800 imágenes de entrenamiento, 1800 de prueba, 6 clases) durante 25 épocas.

Después cargué ese modelo sobre PathMNIST (histopatología, 9 clases) en dos etapas: primero transfer learning con el backbone congelado, luego fine-tuning descongelando toda la red con learning rates diferenciados (1e-5 para el backbone, 1e-4 para la cabeza). Aquí usé 128 px de resolución nativa y 25 épocas para el fine-tuning.

## Lo que aprendí

**1. El fine-tuning sí funciona, pero necesita tiempo.** En la primera corrida (10 épocas) el F1 se quedó en 0.6235 y todavía iba subiendo. Con 25 épocas llegó a 0.7671 y la curva de validación se aplanó — el modelo convergió de verdad. La conclusión cambia completamente según dónde cortes el entrenamiento. Esto me enseñó que cortar antes de tiempo puede llevar a descartar un enfoque que sí funciona.

**2. Transfer learning congelado no sirve cuando los dominios no se parecen.** El modelo aprendió a clasificar paisajes (montañas, playas, edificios) y se evaluó en histopatología. Las features que aprendió no transferían: 0.4680 y estancado desde la época 16. Congelar el backbone solo tiene sentido cuando el dominio de origen y destino están relacionados.

**3. Descongelar el backbone es lo que rescata el caso.** Mismo modelo, mismos datos, solo cambiar `requires_grad=True` y ajustar los learning rates: 0.4680 → 0.7671. Ese salto de +0.30 es la lección más clara del taller. El transfer learning no se trata solo de cargar pesos, sino de dejar que la red se adapte al nuevo dominio.

**4. La resolución nativa importa.** PathMNIST viene a 128 px y el notebook lo subía a 256. Eso multiplicaba el cómputo por cuatro sin añadir información real. Al usar la resolución nativa, el transfer pasó de 1033s a 270s (74% menos) y el fine-tuning de 1253s a 768s (39% menos). Y los resultados mejoraron, no empeoraron. Más resolución no es más información si los píxeles extra son inventados.

**5. Una arquitectura propia bien diseñada puede competir con los preentrenados.** El modelo desde cero alcanzó 0.8697 en su tarea, por encima del transfer y del fine-tuning en la suya. Combinar bloques de distintas familias dio una representación rica sin necesidad de preentrenamiento.

**6. Hay una brecha entre validación y test.** El fine-tuning llegó a un F1 de validación de 0.8634 pero el F1 final en test fue 0.7671. Esa diferencia de ~0.10 indica sobreajuste al split de validación. Es un dato honesto que vale la pena tener presente al interpretar los resultados.

## Sobre el umbral de 0.87

Ninguno de los tres componentes llega a 0.87 en esta corrida. La arquitectura propia se queda por poco (0.8697). El fine-tuning mejoró mucho respecto a la primera corrida pero no alcanzó. Con más épocas o una resolución mayor en Intel probablemente subiría, pero eso queda fuera del alcance de esta entrega.

## Checkpoint

- `grupo77_arquitectura_propia.pth`
- `grupo77_transfer.pth`
- `grupo77_finetune.pth`
- `grupo77_comprobante.json`

Presupuesto: 196 pts · F1 arq propia: 0.8697 · F1 transfer: 0.4680 · F1 fine-tuning: 0.7671