import os
import cv2
import numpy as np
import glob

# Rutas
ruta_limpia = r"C:\Users\juanv\Desktop\Analisis mango\Fotos_Ordenadas"
ruta_mascaras = r"C:\Users\juanv\Desktop\Analisis mango\Carpeta_Especial_Mascaras_Escala"

print("Iniciando algoritmo avanzado de Capas de Máscara (Background Removal + Escalas)...")
os.makedirs(ruta_mascaras, exist_ok=True)

archivos = glob.glob(os.path.join(ruta_limpia, "**", "*.png"), recursive=True)
print(f"Procesando {len(archivos)} imágenes...")

# Simulación de un factor de escala (ej. 1 pixel = 0.05 mm^2) común en microscopía / agronomía
FACTOR_ESCALA_CM2 = 0.005 

for arc in archivos:
    ruta_relativa = os.path.relpath(arc, ruta_limpia)
    ruta_dest_archivo = os.path.join(ruta_mascaras, ruta_relativa)
    
    os.makedirs(os.path.dirname(ruta_dest_archivo), exist_ok=True)

    img = cv2.imread(arc)
    if img is None: continue
        
    img_rz = cv2.resize(img, (400, 400)) # Mayor resolución para contornos
    
    # 1. ELIMINACIÓN DE FONDO CON MEJORAS (Canal Azul / B)
    # En hojas y frutos sobre fondos de contraste, el canal B o S (HSV) aísla bien
    hsv = cv2.cvtColor(img_rz, cv2.COLOR_BGR2HSV)
    _, s, _ = cv2.split(hsv)
    
    # Usamos umbral otsu en el canal Saturación para aislar mágicamente el mango del fondo negro/blanco
    _, mask_fondo = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Limpieza Morfológica (remueve imperfecciones de la máscara de fondo)
    kernel = np.ones((5,5), np.uint8)
    mask_fondo_limpia = cv2.morphologyEx(mask_fondo, cv2.MORPH_CLOSE, kernel)
    
    # 2. ESCALA DE TAMAÑO (Contornos y bounding box)
    contours, _ = cv2.findContours(mask_fondo_limpia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Imagen negra para pintar SOLO las capas de máscara "msck" puras
    analisis_capa = img_rz.copy()
    
    # Oscurecemos el fondo totalmente
    analisis_capa[mask_fondo_limpia == 0] = [0, 0, 0]
    
    area_total_px = 0
    if contours:
        # Tomar el contorno más grande (el mango)
        c_max = max(contours, key=cv2.contourArea)
        area_total_px = cv2.contourArea(c_max)
        
        # Calcular el área física estimada (Escala de Tamaño)
        area_fisica_estimada = area_total_px * FACTOR_ESCALA_CM2
        
        # Dibujar escala/bounding box en la capa de máscara
        x, y, w, h = cv2.boundingRect(c_max)
        cv2.rectangle(analisis_capa, (x, y), (x+w, y+h), (255, 255, 0), 2)
        cv2.drawContours(analisis_capa, [c_max], -1, (0, 255, 255), 2)
        
        texto_escala = f"Area: {round(area_fisica_estimada, 2)} cm2"
        cv2.putText(analisis_capa, texto_escala, (x, max(15, y - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imwrite(ruta_dest_archivo, analisis_capa)

print("\n[OK] ¡Carpeta especifica de Geometrías, Escalas y Background Removal generada!")
