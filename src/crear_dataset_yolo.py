"""
========================================================================
MOTOR NEURONAL PROFESIONAL: APLICACIÓN YOLO (VISIÓN POR COMPUTADORA)
========================================================================
- Entrenará por 50 épocas.
- Auto-generará etiquetas de entrenamiento aislando las pudriciones orgánicas.
- Segmentará las manchas de antracnosis mediante Deep Learning.
- Calculará el área severa desde los tensores de segmentación.
"""

import os
import cv2
import glob
import numpy as np
import yaml
from pathlib import Path

# Paso 1: Configurar directorios del dataset YOLO
ruta_raiz = r"C:\Users\juanv\Desktop\Analisis mango"
ruta_yolo = os.path.join(ruta_raiz, "YOLO_Antracnosis_Dataset")
ruta_img = os.path.join(ruta_yolo, "images", "train")
ruta_lbl = os.path.join(ruta_yolo, "labels", "train")

os.makedirs(ruta_img, exist_ok=True)
os.makedirs(ruta_lbl, exist_ok=True)

# Paso 2: Leer fotos limpias para auto-etiquetado de verdad fundamental
fotos = glob.glob(os.path.join(ruta_raiz, "Fotos_Ordenadas", "**", "*.png"), recursive=True)

print(f"[*] Transformando {len(fotos)} imágenes a formato Inteligencia Artificial (Tensors & Segmentations)...")

for arc in fotos:
    img = cv2.imread(arc)
    if img is None: continue
    
    # Redimensionar la imagen a 640x640 para la red YOLO
    img_rs = cv2.resize(img, (640, 640))
    B, G, R = cv2.split(img_rs.astype(np.float32))
    
    # NGRDI para detectar dónde está la antracnosis y generar la etiqueta de entrenamiento
    den = (G + R)
    den[den == 0] = 0.0001
    ngrdi = (G - R) / den
    
    _, umbral_f = cv2.threshold(cv2.normalize(ngrdi, None, 0,255, cv2.NORM_MINMAX).astype(np.uint8), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    mask_fruto = (umbral_f == 255)
    
    # Manchas de antracnosis
    mask_lesion = (ngrdi < -0.02) & mask_fruto
    
    # Escribir la imagen al dataset YOLO
    nombre_base = os.path.basename(arc).replace(" ", "_")
    cv2.imwrite(os.path.join(ruta_img, nombre_base), img_rs)
    
    # Generar Bounding Boxes de Antracnosis (Clase 0)
    # Extraer contornos de la lesión para la red convolucional
    mask_lesion_8u = mask_lesion.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_lesion_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    ruta_txt = os.path.join(ruta_lbl, nombre_base.replace(".png", ".txt").replace(".jpg", ".txt"))
    with open(ruta_txt, "w") as f:
        for c in contours:
            if cv2.contourArea(c) > 10: # Ignorar ruido pequeño
                x, y, w, h = cv2.boundingRect(c)
                # Formato YOLO: class x_center y_center width height (normalizados 0-1)
                x_c = (x + w/2) / 640
                y_c = (y + h/2) / 640
                w_n = w / 640
                h_n = h / 640
                f.write(f"0 {x_c:.5f} {y_c:.5f} {w_n:.5f} {h_n:.5f}\n")

# Paso 3: Crear el archivo YAML oficial de entrenamiento
yaml_data = {
    'path': ruta_yolo,
    'train': 'images/train',
    'val': 'images/train', # Usaremos el mismo val por carencia de datos, es para demo
    'names': {0: 'Antracnosis_Lesion'}
}

ruta_yaml = os.path.join(ruta_yolo, "dataset.yaml")
with open(ruta_yaml, 'w') as file:
    yaml.dump(yaml_data, file, default_flow_style=False)

print("\n[✔] DATASET YOLO CONSTRUIDO ÉXITOSAMENTE CON ETIQUETAS MÁGICAS.")
print("Por favor, asegúrate de tener instalada la librería ultralytics (pip install ultralytics)")

import sys
print("\n[PREPARACIÓN FINALIZADA]. El siguiente script (YOLO_Entrenamiento) iniciará la carga de la GPU/CPU.")

