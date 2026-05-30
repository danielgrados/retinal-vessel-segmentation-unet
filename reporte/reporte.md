
# Segmentación de Vasos Retinianos mediante U-Net y Estudio de Ablación

## 1. Introducción
Este reporte presenta la implementación de una arquitectura U-Net entrenada desde cero para la segmentación automatizada de la vasculatura retiniana en imágenes de fondo de ojo...

## 2. Metodología
Se utilizó una arquitectura U-Net clásica con bloques de doble convolución, Batch Normalization y funciones de activación ReLU.

### Estudio de Ablación
Para evaluar el impacto de la función de optimización en el desbalance de clases (píxeles de fondo vs. vasos sanguíneos), se analizaron tres configuraciones:
1. **BCE sola**: Entropía cruzada binaria estándar.
2. **Dice sola**: Basada en el coeficiente de superposición.
3. **BCE + Dice (Ponderada)**: Nuestra propuesta con peso penalizado para la clase positiva.

## 3. Experimentos y Resultados
Los modelos se entrenaron utilizando el dataset **DRIVE** y se evaluó la capacidad de generalización en el dataset **CHASE_DB1**.

### Evolución del Entrenamiento
A continuación se presentan las curvas de pérdida y la métrica Dice obtenidas durante las épocas de entrenamiento:

![Curvas de Pérdida y Dice Score](figuras/figura1_curvas_ablacion.png)

### Tabla Resumen de Métricas
Puedes construir tablas nativas en Markdown para mostrar tus resultados de forma impecable:

| Modelo / Pérdida | Dice Score | Sensibilidad | Especificidad | AUC |
| :--- | :---: | :---: | :---: | :---: |
| BCE sola | 0.6214 | 0.5512 | 0.9812 | 0.8514 |
| Dice sola | 0.7415 | 0.7123 | 0.9645 | 0.8923 |
| **BCE + Dice (Ponderada)** | **0.7845** | **0.7612** | **0.9754** | **0.9241** |
| Adaptación CLAHE (DRIVE) | 0.8124 | 0.7945 | 0.9712 | 0.9415 |

> **Nota:** Los valores anteriores representan el mejor estado del modelo restaurado gracias a la configuración de *Early Stopping*.

### Análisis Cualitativo de Fallos
El comportamiento del modelo difiere significativamente al segmentar vasos principales en comparación con capilares finos:

![Análisis de Fallos](figuras/figura6_analisis_fallos.png)

