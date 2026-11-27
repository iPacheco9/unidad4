# Pipeline Operativo de Detección de Obstáculos para Drones (FPV)

**Institución:** Tecnológico Nacional de México (TecNM Virtual)  
**Programa:** Maestría en Inteligencia Artificial  
**Asignatura:** Visión Inteligente  
**Unidad IV:** Despliegue de un Modelo de Visión Inteligente  
**Actividad 4.1:** Creación de un Pipeline Operativo  

---

## 1. Objetivo del Proyecto

Construir un flujo completo y modular de inferencia (*pipeline* operativo) en Python para la detección de obstáculos desde la cámara en primera persona (FPV) de un dron en entornos simulados. El sistema abarca desde la carga de un modelo entrenado con datos propios hasta la generación de salidas visuales estructuradas, operando tanto **por lotes (*batch*)** como en **tiempo real simulado** sobre secuencias de video.

---

## 2. Estructura del Proyecto

```text
unidad4/
├── data_prueba/                 # Lote de imágenes de prueba para inferencia por lotes
├── docs/                        # Gráficas de entrenamiento y evaluación (results.png, matrices)
├── models/
│   └── best.pt                  # Pesos del modelo YOLOv8n entrenado con dataset propio
├── output/
│   ├── lote/                    # Imágenes resultantes con bounding boxes y etiquetas
│   └── video_inferencia.mp4     # Video resultante de la inferencia en tiempo real simulado
├── actividad4_1_dataset.py      # Script de extracción y limpieza de fotogramas desde video
├── pipeline_inferencia.py       # Script principal del pipeline operativo
├── requirements.txt             # Dependencias del proyecto
├── vuelo_uncrashed.mp4          # Video fuente de simulación FPV para prueba en tiempo real
└── README.md                    # Documentación técnica y análisis de resultados
```

---

## 3. Dataset y Entrenamiento del Modelo

Para evitar el desfase visual (*domain gap*) entre fotografías reales y entornos de simulación, se construyó un dataset propio a partir de un recorrido en simulador FPV (*Uncrashed*):

1. **Extracción y limpieza:** Mediante `actividad4_1_dataset.py` (usando OpenCV con `AVFoundation`), se extrajeron fotogramas a intervalos regulares recortando automáticamente los bordes de telemetría y controles en pantalla.
2. **Etiquetado y aumento de datos:** Se etiquetaron **139 imágenes** en Roboflow con dos clases de obstáculos: `estructura` y `vegetacion`. Posteriormente se aplicó redimensionamiento a $640 \times 640$ y aumento de datos (*data augmentation* 3x), obteniendo un total de **417 imágenes** divididas en entrenamiento (80 %) y validación (20 %).
3. **Entrenamiento:** Se entrenó una arquitectura ligera **YOLOv8n** en Google Colab (GPU Tesla T4) durante **50 épocas** con tamaño de lote de 16, generando el archivo de pesos `models/best.pt`.

---

## 4. Arquitectura del Pipeline Operativo

El archivo `pipeline_inferencia.py` implementa el flujo completo de inferencia dividido en cuatro etapas modulares:

1. **Carga del modelo (`cargar_modelo`):** Inicializa el modelo entrenado `models/best.pt` mediante la librería `ultralytics`.
2. **Preprocesamiento (`preprocesar_imagen`):** Para entradas de video crudo, recorta la región de interés eliminando bordes negros, telemetría y joysticks; posteriormente estandariza la resolución a $640 \times 640$ píxeles.
3. **Inferencia (`ejecutar_inferencia_lote` y `ejecutar_inferencia_video`):**
   * **Por lotes:** Recorre las imágenes `.jpg` almacenadas en `data_prueba/`, ejecuta las predicciones y mide el tiempo total y el número de detecciones.
   * **Tiempo real simulado:** Lee los fotogramas de `vuelo_uncrashed.mp4`, ejecuta la inferencia cuadro por cuadro, calcula los FPS y almacena el resultado en video.
4. **Postprocesamiento (`postprocesar_resultados`):** Filtra las predicciones mediante un umbral de confianza mínimo (`CONF_THRESHOLD = 0.40`), dibuja los recuadros (*bounding boxes*) diferenciados por color según la clase, añade las etiquetas con su probabilidad y superpone un panel superior con el conteo de obstáculos y los FPS.

---

## 5. Instalación y Ejecución

### Requisitos previos
* Python 3.10 o superior
* Entorno virtual (Conda o `venv`)

### Instalación de dependencias
Clona o descarga el proyecto, activa tu entorno virtual e instala las librerías necesarias:

```bash
pip install -r requirements.txt
```

Contenido de `requirements.txt`:
```text
ultralytics
opencv-python
```

### Ejecución del pipeline
Asegúrate de contar con el modelo en `models/best.pt`, imágenes `.jpg` en `data_prueba/` y el video en `vuelo_uncrashed.mp4`. Luego ejecuta:

```bash
python pipeline_inferencia.py
```

Al finalizar, las imágenes procesadas se guardarán en `output/lote/` y el video procesado en `output/video_inferencia.mp4`.

---

## 6. Resultados y Análisis Crítico

### Desempeño del modelo entrenado
Durante el entrenamiento a lo largo de 50 épocas, las curvas de pérdida tanto en entrenamiento como en validación muestran un descenso constante y paralelo. En validación, tras algunos picos iniciales de inestabilidad en las primeras épocas, el error disminuye de manera sostenida hasta estabilizarse por debajo de $0.6$ en la pérdida de cajas y de clasificación, lo que indica una convergencia adecuada sin signos evidentes de sobreajuste. Al finalizar las 50 épocas, el modelo alcanza:

* **Precisión:** $\approx 95\,\%$
* **Exhaustividad (*Recall*):** $\approx 95\,\%$
* **mAP@50:** $> 96\,\%$
* **mAP@50-95:** $\approx 85\,\%$

*(Ver gráficas de evolución en `docs/results.png` y matrices en `docs/confusion_matrix_normalized.png`).*

### Interpretación de la matriz de confusión y limitaciones
La matriz de confusión confirma que el modelo separa correctamente ambas clases: identifica el **97 %** de las estructuras (111 aciertos frente a 1 confusión con vegetación y 3 omisiones) y el **100 %** de la vegetación etiquetada (102 aciertos sin omisiones).

Sin embargo, el análisis crítico revela dos aspectos relevantes a considerar antes de su despliegue en navegación autónoma:
1. **Falsos positivos provenientes del fondo:** La matriz registra una cantidad elevada de detecciones sobre el fondo (878 para `estructura` y 988 para `vegetacion`). Esto ocurre porque en las tomas aéreas existen elementos y follaje distantes que no se etiquetaron como obstáculos inmediatos, pero que el modelo detecta cuando se evalúa con umbrales de confianza bajos.
2. **Similitud entre conjuntos:** Dado que las imágenes de entrenamiento y validación se extrajeron de un mismo recorrido en simulación, existe correlación visual entre ambos conjuntos, lo que tiende a elevar de forma optimista las métricas de validación.

### Evaluación del pipeline operativo
En la ejecución local (Apple Silicon M1), el flujo completo funcionó de manera estable:
* **Inferencia por lotes:** Procesó **10 imágenes en 5.85 segundos** ($\approx 0.58\text{ s}$ por imagen), registrando **29 detecciones totales**.
* **Tiempo real simulado:** Procesó **200 fotogramas** continuos aplicando recorte, inferencia y renderizado de salidas en `output/video_inferencia.mp4`.

La inclusión del umbral de confianza de `0.40` en la etapa de postprocesamiento permitió filtrar las detecciones débiles del fondo y conservar únicamente los obstáculos relevantes.

---

## 7. Trabajo Futuro

* **Empaquetado y despliegue (Actividad 4.2):** Contenerizar el pipeline mediante Docker para garantizar reproducibilidad entre entornos.
* **Integración en tesis:** Ampliar el dataset con vuelos en distintos mapas y condiciones de iluminación en Gazebo, validar sobre recorridos completamente inéditos y acoplar las detecciones a un módulo de rastreo (Filtro de Kalman).
