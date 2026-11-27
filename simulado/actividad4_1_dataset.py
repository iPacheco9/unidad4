import cv2
import os

# Configuración
video_path = "vuelo_uncrashed.mp4"  # Cambia por el nombre de tu archivo
output_dir = "dataset_crudo"
intervalo_segundos = 0.8            # Extraer 1 imagen cada 0.8 segundos

os.makedirs(output_dir, exist_ok=True)
cap = cv2.VideoCapture(video_path, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("Error: No se pudo abrir el video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
salto_frames = max(1, int(fps * intervalo_segundos))

frame_actual = 0
guardadas = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    if frame_actual % salto_frames == 0:
        alto, ancho, _ = frame.shape
        
        # Recorte automático:
        # - Quitamos el 18% inferior (joysticks)
        # - Quitamos el 15% derecho (barra negra y telemetría del video 1)
        # (Si usas el video 2, puedes cambiar 0.85 a 1.0 para no cortar la derecha)
        #frame_limpio = frame[0:int(alto * 0.82), 0:int(ancho * 0.85)] #opción 1
        frame_limpio = frame[0:int(alto * 0.82), int(ancho * 0.15):int(ancho * 0.85)]
        
        nombre_archivo = os.path.join(output_dir, f"frame_{guardadas:04d}.jpg")
        cv2.imwrite(nombre_archivo, frame_limpio)
        guardadas += 1
        
    frame_actual += 1

cap.release()
print(f"Listo. Se extrajeron {guardadas} imágenes limpias en '{output_dir}'.")