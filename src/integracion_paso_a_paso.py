"""
🛠️ GUÍA DE INTEGRACIÓN PASO A PASO: YOLO + SAM + SPECTRAL
Este script sirve como documentación ejecutable para entender el flujo lógico.
"""

import cv2
import numpy as np
from ultralytics import YOLO, SAM

def tutorial_integracion(image_path, model_path):
    print("--- INICIANDO TUTORIAL DE INTEGRACIÓN ---")
    
    # PASO 1: CARGAR MODELOS
    print("[1] Cargando YOLOv26 y SAM...")
    yolo = YOLO(model_path)
    sam = SAM('sam_b.pt')
    
    # PASO 2: DETECCIÓN INICIAL
    print("[2] Ejecutando detección en imagen...")
    img = cv2.imread(image_path)
    results_yolo = yolo.predict(img, conf=0.5)[0]
    
    for box in results_yolo.boxes:
        bbox = box.xyxy[0].tolist() # Coordenadas [x1, y1, x2, y2]
        
        # PASO 3: SEGMENTACIÓN DE PRECISIÓN (SAM)
        print("[3] Segmentando fruto detectado con SAM...")
        results_sam = sam.predict(img, bboxes=[bbox])[0]
        mask = results_sam.masks.data[0].cpu().numpy().astype(np.uint8)
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        
        # PASO 4: ANÁLISIS ESPECTRAL (NGRDI)
        print("[4] Calculando índices espectrales dentro de la máscara...")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        R = img_rgb[:,:,0].astype(float)
        G = img_rgb[:,:,1].astype(float)
        
        ngrdi = (G - R) / (G + R + 1e-6)
        
        # Filtrar solo necrosis dentro del fruto
        necrosis_mask = (ngrdi < -0.075) & (mask > 0)
        
        # PASO 5: CUANTIFICACIÓN
        fruto_area = np.sum(mask > 0)
        necrosis_area = np.sum(necrosis_mask)
        severidad = (necrosis_area / fruto_area) * 100
        
        print(f"✅ Fruto detectado. Severidad calculada: {severidad:.2f}%")

if __name__ == "__main__":
    print("Este script es educativo. Para producción usa 'mango_disease_pipeline.py'")
