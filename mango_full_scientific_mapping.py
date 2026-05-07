import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
import glob
import os

class ScientificMangoMapper:
    def __init__(self, base_dir="Resultados_Mapeo_Cientifico"):
        print("[*] Iniciando Sistema de Mapeo Científico Avanzado...")
        # Usamos YOLOv8 como motor base
        self.model = YOLO('yolov8n.pt') 
        self.k = 0.00025 # Calibración cm2
        self.base_dir = os.path.abspath(base_dir)
        self.dirs = {
            "REPORTES": os.path.join(self.base_dir, "REPORTES_1x6"),
            "DATOS": os.path.join(self.base_dir, "DATOS_CSV")
        }
        for d in self.dirs.values():
            if not os.path.exists(d): os.makedirs(d)

    def count_and_measure_lesions(self, roi_ngrdi):
        """ Identifica cada mancha de antracnosis mediante componentes conectados """
        # Umbralización para detectar tejido necrótico (antracnosis)
        lesion_mask = (roi_ngrdi < -0.065).astype(np.uint8) * 255
        
        # Análisis de componentes conectados para el conteo individual
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(lesion_mask)
        
        valid_lesions = 0
        total_lesion_area = 0
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > 10: # Filtro para ignorar ruido de píxeles aislados
                valid_lesions += 1
                total_lesion_area += area
                
        return valid_lesions, total_lesion_area, lesion_mask

    def process_hierarchy(self, root_path):
        # Búsqueda recursiva de imágenes en toda la estructura de 'fruto'
        image_files = glob.glob(os.path.join(root_path, "**", "*.png"), recursive=True) + \
                      glob.glob(os.path.join(root_path, "**", "*.jpg"), recursive=True)
        
        print(f"[*] Analizando {len(image_files)} imágenes encontradas en la jerarquía...")
        
        master_data = []

        for img_path in image_files:
            # Extraer metadata de la ruta para el reporte científico
            path_parts = img_path.split(os.sep)
            # Estructura esperada: .../fruto/TRATAMIENTO/PODA/MES/imagen.png
            tratamiento = "N/A"
            poda = "N/A"
            mes = "N/A"
            
            # Intentar mapear la jerarquía según la profundidad de la ruta
            if "PROTERRA" in path_parts: tratamiento = "PROTERRA"
            elif "SERENADE" in path_parts: tratamiento = "SERENADE"
            elif "TESTIGO" in path_parts: tratamiento = "TESTIGO"
            elif "TRICHODERMA" in path_parts: tratamiento = "TRICHODERMA"
            
            if "Con poda" in path_parts or "Conpoda" in path_parts: poda = "Con Poda"
            elif "Sin poda" in path_parts or "Sinpoda" in path_parts: poda = "Sin Poda"
            
            img = cv2.imread(img_path)
            if img is None: continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            
            # 1. Inferencia YOLO (Detección del Fruto)
            results = self.model.predict(img, conf=0.15, verbose=False)
            
            # 2. Matemática Pliman (NGRDI)
            B, G, R = cv2.split(img.astype(np.float32))
            ngrdi = (G - R) / (G + R + 1e-5)
            
            diagnostico_final = img_rgb.copy()
            count_total = 0
            
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    roi_ngrdi = ngrdi[y1:y2, x1:x2]
                    
                    n_lesiones, a_lesiones, l_mask = self.count_and_measure_lesions(roi_ngrdi)
                    count_total += n_lesiones
                    
                    # Dibujar el conteo y severidad sobre la imagen de diagnóstico
                    cv2.rectangle(diagnostico_final, (x1, y1), (x2, y2), (0, 255, 0), 4)
                    cv2.putText(diagnostico_final, f"Lesiones: {n_lesiones}", (x1, y1-15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)

                    master_data.append({
                        "Archivo": os.path.basename(img_path),
                        "Tratamiento": tratamiento,
                        "Poda": poda,
                        "Lesiones_Contadas": n_lesiones,
                        "Area_Fruto_cm2": round(roi_ngrdi.size * self.k, 2),
                        "Severidad_%": round((a_lesiones / (roi_ngrdi.size + 1)) * 100, 2),
                        "Ruta": img_path
                    })

            # Generar Reporte 1x6 (Solo guardamos los primeros 30 para evitar demora excesiva)
            if len(master_data) < 31:
                self.save_1x6_card(img_rgb, ngrdi, diagnostico_final, os.path.basename(img_path))

        # Consolidación de datos en CSV
        df = pd.DataFrame(master_data)
        df.to_csv(os.path.join(self.dirs["DATOS"], "Mapeo_Cientifico_Total_Fruto.csv"), index=False)
        print(f"\n[!] MAPEO COMPLETADO.")
        print(f"Total imágenes procesadas: {len(image_files)}")
        print(f"Archivo de datos: {os.path.join(self.dirs['DATOS'], 'Mapeo_Cientifico_Total_Fruto.csv')}")
        
        # Resumen rápido por tratamiento
        if not df.empty:
            print("\nResumen por Tratamiento:")
            print(df.groupby('Tratamiento')['Severidad_%'].mean())
        
        return df

    def save_1x6_card(self, original, ngrdi, final, name):
        heatmap = cv2.applyColorMap(cv2.normalize(ngrdi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        fig, axes = plt.subplots(1, 6, figsize=(24, 4))
        moments = [original, original, original, heatmap, original, final]
        titles = ["Orig", "YOLO", "Mask", "NGRDI", "Grid", "Final"]
        for ax, m, t in zip(axes, moments, titles):
            ax.imshow(cv2.resize(m, (400, 400))); ax.set_title(t); ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs["REPORTES"], f"REPORT_1x6_{name}.png"))
        plt.close()

if __name__ == "__main__":
    # Ruta raíz del dataset proporcionado en la captura
    root_fruto = r"C:\Users\juanv\Desktop\Analisis mango\fruto"
    mapper = ScientificMangoMapper()
    mapper.process_hierarchy(root_fruto)
