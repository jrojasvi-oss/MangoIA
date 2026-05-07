# =========================================================================
# DETECCIÓN DE ANTRACNOSIS CON YOLOv8 PARA MANGOS
# Este script escanea la carpeta 'fruto' usando modelos predictivos.
# =========================================================================

import cv2
import pandas as pd
from ultralytics import YOLO
import glob
import os

# 1. Cargar el modelo YOLO
# Asegúrate de instalar 'ultralytics' con: pip install ultralytics
# Por defecto descargará YOLOv8 de segmentación (nano)
try:
    print("Cargando modelo neuronal YOLOv8...")
    model = YOLO('yolov8n-seg.pt') 
except Exception as e:
    print("Asegúrate de instalar 'ultralytics'. Error:", e)

def analizar_directorio_yolo(ruta_raiz):
    archivos = glob.glob(ruta_raiz + "/**/*.png", recursive=True)
    if not archivos:
        print("No se encontraron archivos PNG en", ruta_raiz)
        return pd.DataFrame()
        
    print(f"Recopiladas {len(archivos)} imágenes para analizar...")
    resultados = []

    for img_path in archivos:
        # Extraer metadatos de la ruta (asume la estructura descrita en el diseño DOCX)
        partes = img_path.split(os.sep)
        # La carpeta tratamiento suele estar algunos niveles arriba del archivo
        # fruto/fruto/TRICHODERMA/Sinpoda/septiembre/rr3.png
        tratamiento = "Desconocido"
        for idx, p in enumerate(partes):
            if p.upper() in ["PROTERRA", "SERENADE", "TESTIGO", "TRICHODERMA"]:
                tratamiento = p.upper()
                break
        
        try:
            # Realizar predicción con umbral de confianza 0.25 (ajustable)
            res = model.predict(img_path, conf=0.25, verbose=False)
            
            # Guardamos los resultados por cada imagen
            for r in res:
                resultados.append({
                    "Tratamiento": tratamiento,
                    "Archivo": os.basename(img_path),
                    "Num_Detecciones": len(r.boxes)
                })
        except Exception as e:
            print(f"Error procesando {img_path}: {e}")
            
    df = pd.DataFrame(resultados)
    return df

# Ejecutar el escaneo
ruta = r"C:\Users\juanv\Desktop\Analisis mango\fruto"
print("Iniciando escaneo de YOLO en los mangos...")
df_yolo = analizar_directorio_yolo(ruta)

if not df_yolo.empty:
    archivo_csv = "resultados_yolo_mangos.csv"
    df_yolo.to_csv(archivo_csv, index=False)
    print(f"\n¡Análisis finalizado! Los resúmenes de inteligencia artificial fueron guardados en '{archivo_csv}'")
else:
    print("\nNo se pudo completar el análisis.")
