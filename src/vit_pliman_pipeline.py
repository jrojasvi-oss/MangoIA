import os
import cv2
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ==============================================================================
# PIPELINE DE ALTA TECNOLOGÍA: MATEMÁTICAS PLIMAN + CLUSTERING AVANZADO (CNN/ViT Proxy)
# ==============================================================================

def calcular_ngrdi(imagen):
    """
    Simula la matemática colorimétrica de Pliman.
    NGRDI = (G - R) / (G + R)
    Valores negativos indican presencia de tejido enfermo (marrón) o fondo oscuro.
    """
    img_float = imagen.astype(np.float32)
    B, G, R = cv2.split(img_float)
    
    # Evitar divisiones por cero
    denominador = (G + R)
    denominador[denominador == 0] = 0.0001
    
    ngrdi = (G - R) / denominador
    return ngrdi

def segmentar_sev_math(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None, 0, 0
    
    # Redimensionar para análisis rápido y alta dimensionalidad
    img_rz = cv2.resize(img, (224, 224))
    ngrdi = calcular_ngrdi(img_rz)
    
    # Separación Otsu para simular el comportamiento base de R
    # Normalizar NGRDI para Otsu (0-255)
    ngrdi_norm = cv2.normalize(ngrdi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, umbral_fondo = cv2.threshold(ngrdi_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Fruto vs Fondo: el fondo suele tener NGRDI distinto. Supongamos que > Otsu es fruto.
    fruto_mask = (umbral_fondo == 255)
    
    # Antracnosis: Rango NGRDI muy negativo.
    # Asumimos una calibración donde < 0 es lesión.
    lesion_mask = (ngrdi < -0.05) & fruto_mask
    sano_mask = (ngrdi >= -0.05) & fruto_mask
    
    # Conteo de pixeles
    pixeles_totales = np.sum(fruto_mask)
    if pixeles_totales == 0:
        return img_rz, 0, 0
    
    pixeles_enfermos = np.sum(lesion_mask)
    severidad = (pixeles_enfermos / pixeles_totales) * 100
    
    # Pintar diagnóstico (Rojo para enfermo, Verde para Sano)
    diagnostico = img_rz.copy()
    diagnostico[lesion_mask] = [0, 0, 255]   # Rojo (BGR)
    diagnostico[sano_mask] = [0, 255, 0]     # Verde
    
    return diagnostico, severidad, pixeles_totales

def escaneo_directorio_profundo(ruta):
    archivos = glob.glob(os.path.join(ruta, "**", "*.png"), recursive=True)
    if not archivos:
        print("No se encontraron imágenes.")
        return
    
    # Analizamos máximo 20 imágenes para rapidez en terminal
    archivos = archivos[:20] 
    
    resultados = []
    
    print("EXTRACCIÓN DE CARACTERÍSTICAS Y CÁLCULO DE ÁREAS (MOTOR PLIMAN-Py)")
    print("-" * 60)
    
    for arc in archivos:
        diag, sev, area = segmentar_sev_math(arc)
        if diag is not None:
            tratamiento = "Desc"
            for parte in arc.split(os.sep):
                if parte.upper() in ["PROTERRA", "SERENADE", "TESTIGO", "TRICHODERMA"]:
                    tratamiento = parte.upper()
                    
            resultados.append({
                "Archivo": os.path.basename(arc),
                "Tratamiento": tratamiento,
                "Severidad_%": round(sev, 2),
                "Pixeles_Mango": area
            })
            
            # Guardamos un ejemplo visual de la primera imagen
            if len(resultados) == 1:
                cv2.imwrite("ejemplo_diagnostico_calibracion.png", diag)
                print(f"[*] Guardada imagen de calibración visual: {os.path.basename(arc)}")
                
    df = pd.DataFrame(resultados)
    
    # Machine Learning Rápido (Clustering de estados de severidad simulando capas Densas)
    if len(df) > 5:
        print("\n[*] Aplicando Análisis Algorítmico (K-Means/ViT surrogate) para clústeres clínicos...")
        # Clusterizamos severidades en 3 niveles (Sano, Leve, Grave)
        kmeans = KMeans(n_clusters=3, random_state=42)
        df['Nivel_IA'] = kmeans.fit_predict(df[['Severidad_%']])
        
    print("\n" + "="*50)
    print("REPORTE ANALÍTICO (Head):")
    print(df.head(10).to_string())
    print("="*50)
    
    df.to_csv("severidad_analizada_python.csv", index=False)
    print("\n=> Pipeline Completado. Datos guardados en 'severidad_analizada_python.csv'")

if __name__ == "__main__":
    ruta_dataset = r"C:\Users\juanv\Desktop\Analisis mango\fruto_ext"
    escaneo_directorio_profundo(ruta_dataset)
