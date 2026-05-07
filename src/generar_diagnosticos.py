import os
import cv2
import numpy as np
import glob

# Rutas
ruta_limpia = r"C:\Users\juanv\Desktop\Analisis mango\Fotos_Ordenadas"
ruta_diagnostico = r"C:\Users\juanv\Desktop\Analisis mango\Fotos_Diagnosticadas"

print("Iniciando generación masiva de máscaras de diagnóstico visual...")
os.makedirs(ruta_diagnostico, exist_ok=True)

archivos = glob.glob(os.path.join(ruta_limpia, "**", "*.png"), recursive=True)
print(f"Detectadas {len(archivos)} fotos ordenadas.")

for arc in archivos:
    # Mantener el mismo árbol exacto
    ruta_relativa = os.path.relpath(arc, ruta_limpia)
    ruta_dest_archivo = os.path.join(ruta_diagnostico, ruta_relativa)
    
    # Asegurar que el directorio destino existe
    os.makedirs(os.path.dirname(ruta_dest_archivo), exist_ok=True)

    img = cv2.imread(arc)
    if img is None:
        continue
        
    img_rz = cv2.resize(img, (224, 224))
    img_float = img_rz.astype(np.float32)
    B, G, R = cv2.split(img_float)
    
    # NGRDI Matemático (Pliman proxy)
    den = (G + R)
    den[den == 0] = 0.0001
    ngrdi = (G - R) / den
    
    ngrdi_norm = cv2.normalize(ngrdi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, umbral_fondo = cv2.threshold(ngrdi_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    fruto_mask = (umbral_fondo == 255)
    lesion_mask = (ngrdi < -0.02) & fruto_mask
    sano_mask = (ngrdi >= -0.02) & fruto_mask
    
    # Diagnóstico: Pintar Rojo Antracnosis y Verde Sano
    diagnostico = img_rz.copy()
    diagnostico[lesion_mask] = [0, 0, 255] # Rojo (BGR)
    diagnostico[sano_mask] = [0, 255, 0]   # Verde
    
    # Escribir la foto coloreada
    cv2.imwrite(ruta_dest_archivo, diagnostico)

print("\n¡Misión completada! Las fotos con segmentación visual se guardaron rigurosamente en:")
print(f"-> {ruta_diagnostico}")
