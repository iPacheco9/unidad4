import cv2
import os
import time
import glob
from ultralytics import YOLO

# --- CONFIGURACIÓN ---
MODEL_PATH = "models/best.pt"
BATCH_INPUT_DIR = "data_prueba"
VIDEO_INPUT_PATH = "vuelo_uncrashed.mp4"
OUTPUT_DIR = "output"
CONF_THRESHOLD = 0.40  # Confianza mínima para mostrar una detección

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "lote"), exist_ok=True)


# 1. CARGA DEL MODELO
def cargar_modelo(ruta_modelo):
    print(f"[INFO] Cargando modelo desde: {ruta_modelo}")
    return YOLO(ruta_modelo)


# 2. PREPROCESAMIENTO
def preprocesar_imagen(imagen, es_video_crudo=False):
    """
    Si viene del video original, recorta los bordes de telemetría/joysticks.
    Luego estandariza el tamaño a 640x640 para la entrada del modelo.
    """
    if es_video_crudo:
        alto, ancho, _ = imagen.shape
        imagen = imagen[0:int(alto * 0.82), int(ancho * 0.15):int(ancho * 0.85)]
    
    imagen_redim = cv2.resize(imagen, (640, 640))
    return imagen_redim


# 3. POSTPROCESAMIENTO
def postprocesar_resultados(imagen, resultados, fps=None):
    """
    Dibuja las cajas (bounding boxes), etiquetas, confianza y un resumen en pantalla.
    """
    img_salida = imagen.copy()
    conteo_obstaculos = 0

    for r in resultados:
        cajas = r.boxes
        for caja in cajas:
            conf = float(caja.conf[0])
            if conf < CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, caja.xyxy[0])
            cls_id = int(caja.cls[0])
            nombre_clase = r.names[cls_id]
            conteo_obstaculos += 1

            # Color distinto según la clase
            color = (0, 255, 0) if cls_id == 0 else (255, 165, 0)
            cv2.rectangle(img_salida, (x1, y1), (x2, y2), color, 2)
            etiqueta = f"{nombre_clase} {conf:.2f}"
            cv2.putText(img_salida, etiqueta, (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Panel superior informativo
    texto_info = f"Obstaculos detectados: {conteo_obstaculos}"
    if fps is not None:
        texto_info += f" | FPS: {fps:.1f}"
    cv2.putText(img_salida, texto_info, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return img_salida, conteo_obstaculos


# 4A. INFERENCIA POR LOTES (BATCH)
def ejecutar_inferencia_lote(modelo, carpeta_entrada):
    print("\n--- INICIANDO INFERENCIA POR LOTES ---")
    rutas = glob.glob(os.path.join(carpeta_entrada, "*.jpg"))
    if not rutas:
        print(f"[AVISO] No se encontraron imágenes .jpg en '{carpeta_entrada}'.")
        return

    inicio = time.time()
    total_detecciones = 0

    for ruta in rutas:
        img = cv2.imread(ruta)
        img_prep = preprocesar_imagen(img, es_video_crudo=False)
        resultados = modelo(img_prep, verbose=False)
        img_final, conteo = postprocesar_resultados(img_prep, resultados)
        total_detecciones += conteo

        nombre = os.path.basename(ruta)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "lote", f"pred_{nombre}"), img_final)

    tiempo_total = time.time() - inicio
    print(f"[OK] Lote completado: {len(rutas)} imágenes en {tiempo_total:.2f}s "
          f"({total_detecciones} detecciones totales).")
    print(f"[OK] Resultados guardados en '{OUTPUT_DIR}/lote/'")


# 4B. INFERENCIA EN TIEMPO REAL SIMULADO (VIDEO)
def ejecutar_inferencia_video(modelo, ruta_video, max_frames=200):
    print("\n--- INICIANDO INFERENCIA EN TIEMPO REAL SIMULADO ---")
    cap = cv2.VideoCapture(ruta_video, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir el video '{ruta_video}'.")
        return

    salida_video_path = os.path.join(OUTPUT_DIR, "video_inferencia.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(salida_video_path, fourcc, 20.0, (640, 640))

    frame_count = 0
    while frame_count < max_frames:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        frame_prep = preprocesar_imagen(frame, es_video_crudo=True)
        resultados = modelo(frame_prep, verbose=False)
        
        fps = 1.0 / max(time.time() - t0, 1e-5)
        frame_final, _ = postprocesar_resultados(frame_prep, resultados, fps=fps)

        out.write(frame_final)
        cv2.imshow("Pipeline Operativo - Tiempo Real Simulado (Presiona 'q' para salir)", frame_final)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[OK] Video procesado ({frame_count} frames) y guardado en '{salida_video_path}'.")


if __name__ == "__main__":
    modelo_yolo = cargar_modelo(MODEL_PATH)
    ejecutar_inferencia_lote(modelo_yolo, BATCH_INPUT_DIR)
    ejecutar_inferencia_video(modelo_yolo, VIDEO_INPUT_PATH, max_frames=200)