# Detector de Emociones con Redes Convolucionales (CNN)

## Descripción del detector de emociones

Este proyecto implementa un sistema de **detección de emociones** a partir de imágenes faciales utilizando una **Red Neuronal Convolucional (CNN)** con la librería `TensorFlow/Keras`.

El objetivo es entrenar un modelo capaz de clasificar una expresión facial detectada en una de las cinco categorías emocionales principales (`angry`, `happy`, `neutral`, `sad`, `surprise`) y el proyecto incluye una funcionalidad de **"Filtro por Emoción"** en tiempo real utilizando la cámara web.

## Dataset

El modelo fue entrenado con un dataset de imágenes faciales de emociones. Se partió del dataset de kaggle [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013?resource=download). En este, se contaba con dos carpetas, entrenamiento y validcación. Decidí juntar todas las imágenes de las diferentes emociones en una única carpeta, y a partir del script `split_dataset.py`, dido en conjuntos de entrenamiento, validación y prueba, organizados en la estructura de directorios: `emotion_dataset/{train, validation, test}/{emoción}`.

### Distribución de Clases

La distribución de imágenes en el dataset completo muestra un cierto **desbalance de clases**, lo cual fue abordado en el entrenamiento mediante el uso de **pesos de clase** (`class_weight='balanced'`).

| Emoción | Conteo Total | Proporción (%) |
| :---: | :---: | :---: |
| **happy** | 8989 | 29.7 |
| **neutral** | 6198 | 20.5 |
| **sad** | 6077 | 20.1 |
| **angry** | 4953 | 16.4 |
| **surprise** | 4002 | 13.2 |
| **Total** | 30219 | 100.0 |

## Arquitectura del Modelo CNN

El modelo utiliza una arquitectura CNN secuencial diseñada para clasificar imágenes en escala de grises de **48x48 píxeles**.

| Capa | Tipo de Capa | Parámetros Clave | Salida (Shape) |
| :--- | :--- | :--- | :--- |
| **1** | Conv2D | 32 filtros, (3,3), ReLU | (48, 48, 32) |
| | BatchNormalization | | |
| | MaxPooling2D | (2,2) | (24, 24, 32) |
| **2** | Conv2D | 64 filtros, (3,3), ReLU | (24, 24, 64) |
| | BatchNormalization | | |
| | MaxPooling2D | (2,2) | (12, 12, 64) |
| **3** | Conv2D | 128 filtros, (3,3), ReLU | (12, 12, 128) |
| | BatchNormalization | | |
| | MaxPooling2D | (2,2) | (6, 6, 128) |
| **4** | Flatten | | |
| **5** | Dense | 128 unidades, ReLU | 128 |
| **6** | Dropout | 50% | 128 |
| **7** | Dense | **5** unidades, Softmax | **5** |

El modelo fue compilado con el optimizador **Adam** ($\text{learning\_rate}=0.0003$) y la función de pérdida **Categorical Crossentropy**. Previamente se probó un valor de ritmo de aprendizaje superior (0.001, pero los saltos de la precisión en las imágenes de validación eran demasiado bruscos y arbitrarios) 

## Configuración de Entrenamiento

* **Generador de Datos:** Se utilizó `ImageDataGenerator` con aumento de datos (`rotation_range`, `width/height_shift`, `zoom_range`, `horizontal_flip`) para el *training set* para mejorar la robustez.
* **Pesos de Clase:** Se aplicaron pesos de clase calculados como `'balanced'` para mitigar el efecto del desbalance en el dataset.
* **Épocas:** 50.
* **Tamaño del Lote (Batch Size):** 64.

## Resultados del Entrenamiento

El modelo alcanzó los siguientes resultados después de 50 épocas de entrenamiento:

| Métrica | Training | Validation | Test |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 69.00% | 67.70% | **67.54%** |
| **Loss** | 0.8135 | 0.8431 | 0.8566 |

El modelo final fue guardado como `emotion_model.h5`.

### Gráficos de Evolución

![Accuracy del entrenamiento](outputs/output1.png)

El hecho de que la pérdida de validación no se dispare y se mantenga muy cerca de la pérdida de entrenamiento indica que el modelo no parece estar sufriendo de un sobreajuste severo (overfitting).

![Loss del entrenamiento](outputs/output.png)

El modelo alcanza una precisión final de casi el 70%. La diferencia entre la precisión de entrenamiento y validación es pequeña, lo que refuerza la idea de que el modelo generaliza bien (no hay overfitting significativo). Sin embargo, la volatilidad en la validación podría sugerir la necesidad de ajustar la tasa de aprendizaje o utilizar técnicas de regularización adicionales.

### Matriz de Confusión

![Matriz de confusión](outputs/output2.png)

El modelo demuestra una habilidad robusta para clasificar Felicidad y Sorpresa. No obstante, el rendimiento global del modelo se ve significativamente afectado por la dificultad para distinguir entre Neutral y Tristeza, que son las clases peor clasificadas. La principal fuente de error reside en la confusión mutua entre Neutral y Tristeza, sumado a que la Tristeza se confunde frecuentemente con el Enfado. Esto indica que el modelo necesita una mejor diferenciación de las características sutiles que separan estos estados emocionales.

# Detección de caras

El código incluye una funcionalidad de visión por computadora en tiempo real a través de la webcam:

## Detección de Cara y Emoción (penúltima celda en el notebook)

* Detecta un rostro utilizando el clasificador **Haar Cascade**.
* Clasifica la emoción del rostro detectado con el modelo CNN entrenado.
* Muestra la emoción y su confianza en el recuadro de la cara.
* Proyecta el *feed* de la cámara junto a un **emoticono** predefinido (cargado de la carpeta `filters`) que corresponde a la emoción detectada.

##  Detección de Cara y Aplicación de un Filtro (última celda en el notebook)

* Detecta rostros.
* Aplica un conjunto fijo de filtros decorativos (`hat`, `glasses`, `scarf`) sobre el rostro detectado, demostrando técnicas de superposición con canal alfa y transformación de imagen (`overlay_alpha`, `transform_filter`). Las posiuciones para el gorro, las gafas y la bufanda son calculadas de forma orientativa y geométrica.


### Dependencias y Ejecución

Asegúrate de tener instaladas las siguientes bibliotecas:

```bash
pip install tensorflow opencv-python numpy pandas matplotlib seaborn scikit-learn
```

Para ejecutar la funcionalidad de cámara en tiempo real, asegúrate de que:

* Tengas el archivo del modelo emotion_model.h5 en el mismo directorio.

* Tengas la estructura de directorios emotion_dataset/{train, validation, test} (necesaria para el generador de pruebas, aunque no se reentrene).

* Tengas la carpeta filters con las imágenes PNG/JPG para las emociones y/o los accesorios (hat.png, glasses.png, scarf.png, etc.) según el script que uses.

