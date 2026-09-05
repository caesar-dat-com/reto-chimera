# Reto Chimera — Informe de la corrida base

**Grupo 77** · Ingeniería de Datos e IA · UAO

## Qué hice

Armé la arquitectura ChimeraNet con siete bloques y la corrí tal como venía el
notebook, sin cambiarle nada al código del profesor. La idea era tener primero
una línea base honesta: ver qué da la configuración por defecto antes de ponerme
a tocar cosas.

La configuración que elegí gastó 196 puntos de los 200 disponibles, así que me
quedé casi en el techo del presupuesto. Metí siete bloques mezclando familias
(MobileNet, Inception, ConvNeXt, ResNet y VGG), con AdamW, scheduler coseno y
label smoothing. Le agregué una convolución propia y dropout de 0.3.

```
Bloques:     171 pts
Optimizador: adamw -> 6 pts
Scheduler:   cosine -> 5 pts
Loss:        label_smoothing -> 4 pts
Componentes: conv_custom + global_avg_pool + dropout(0.3) -> 10 pts
TOTAL: 196 pts (rango 150-200) VALIDO
```

## Resultados

| Componente | F1 macro | Épocas | Duración |
|---|---|---|---|
| Arquitectura propia | 0.8768 | 25 | 224.9 s |
| Transfer learning | 0.5118 | 25 | 1033.3 s |
| Fine-tuning | 0.6235 | 10 | 1253.3 s |

## Lo que veo en estos números

**La arquitectura propia funcionó bien.** Entrenando desde cero en Intel llegó a
0.8768, y la curva se ve sana: arrancó en 0.6849 y terminó en 0.8523 de
validación, con la pérdida bajando de 1.21 a 0.51. No se quedó atascada ni se
disparó.

**El transfer learning fue el peor de los tres, y creo que tiene sentido.** Al
congelar el stem y las features, el modelo sigue viendo PathMNIST con los ojos
que aprendió mirando paisajes de Intel. Son dominios que no se parecen en nada:
un backbone que aprendió a separar montaña de playa no tiene por qué servir para
distinguir tipos de tejido. La curva lo confirma: subió hasta la época 16 más o
menos y de ahí se aplanó entre 0.48 y 0.49. Ya no estaba aprendiendo, solo
girando en el mismo punto.

**El fine-tuning se quedó corto, y esto es lo importante del informe.** Al
descongelar todo, el modelo empezó a mejorar de verdad: la pérdida cayó de 15.57
a 2.64 y el F1 de validación subió de 0.29 a 0.6769. Pero el notebook tenía el
fine-tuning fijo en 10 épocas y **en la última época todavía estaba subiendo**
(0.6450 → 0.6769). O sea, lo corté en la mitad de la subida. El 0.6235 que
reporta no es el techo del método, es donde alcanzó a llegar antes de que se le
acabaran las épocas.

## Un detalle del notebook que noté

El checkpoint del fine-tuning guarda `epocas_entrenadas=25` aunque en realidad
solo corrieron 10. La celda pasa la constante `MAX_EPOCHS` en vez del valor real
del bucle. No cambia el resultado, pero el archivo dice una cosa y el log dice
otra.

También noté que las celdas 20 y 23 imprimen un mensaje que no coincide con lo
que comparan: el texto dice `>= 0.90` y `>= 0.87`, pero la comparación real es
contra 0.75 y 0.76. La tabla del resumen sí usa 0.87. Lo menciono para que no se
confunda quien lea el notebook.

## Conclusión

Ninguno de los tres componentes llega al umbral de 0.87. La arquitectura propia
se queda cerca (0.8768 en su tarea), pero transfer y fine-tuning quedan lejos.

Mi lectura es que el fine-tuning no falló: quedó sin terminar. Por eso hice una
segunda corrida cambiando ese parámetro, y el resultado está en el otro informe.
