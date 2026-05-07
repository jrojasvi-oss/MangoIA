import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
import glob
import os

class MassiveMangoAnalyzer:
    def __init__(self, pixel_cm2_ratio=0.00025):
        print("[*] Cargando motor masivo YOLOv26...")
        self.model = YOLO('yolov8n.pt') 
        self.k = pixel_cm2_ratio
        
    def calculate_ngrdi(self, img):
        B, G, R = cv2.split(img.astype(np.float32))
        return (G - R) / (G + R + 1e-5)

    def analyze_batch(self, input_path, output_csv):
        image_files = glob.glob(os.path.join(input_path, "*.jpg")) + glob.glob(os.path.join(input_path, "*.png"))
        print(f"[*] Procesando {len(image_files)} imágenes del set de validación...")
        
        all_data = []

        for img_path in image_files:
            img = cv2.imread(img_path)
            if img is None: continue
            
            results = self.model.predict(img, conf=0.15, verbose=False)
            ngrdi = self.calculate_ngrdi(img)
            
            for r in results:
                for i, box in enumerate(r.boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    roi_ngrdi = ngrdi[y1:y2, x1:x2]
                    
                    sev = (np.sum(roi_ngrdi < -0.05) / (roi_ngrdi.size + 1)) * 100
                    area_cm2 = roi_ngrdi.size * self.k
                    
                    all_data.append({
                        "Archivo": os.path.basename(img_path),
                        "ID_Fruto": i + 1,
                        "Area_cm2": round(area_cm2, 2),
                        "Severidad_%": round(sev, 2),
                        "Estado": "AFECTADO" if sev > 1.5 else "SANO"
                    })

        df = pd.DataFrame(all_data).sort_values(by="Severidad_%")
        df.to_csv(output_csv, index=False)
        return df

if __name__ == "__main__":
    valid_images = r"C:\Users\juanv\Desktop\Analisis mango\Train\Mango-Antracnosis.v1i.yolo26\valid\images"
    output = "REPORTE_MASIVO_ANTRACO_MANGO.csv"
    
    analyzer = MassiveMangoAnalyzer()
    df_results = analyzer.analyze_batch(valid_images, output)
    
    print(f"\n[!] ANÁLISIS COMPLETADO. Datos guardados en {output}")
    print("\nResumen Estadístico:")
    print(df_results.groupby('Estado').agg({'Area_cm2': 'mean', 'Severidad_%': 'mean', 'ID_Fruto': 'count'}))
