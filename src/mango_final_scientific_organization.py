import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
import glob
import os

class ScientificMangoOrganizer:
    def __init__(self, base_dir="Estudio_Resultados_Mango"):
        print("[*] Cargando motor masivo YOLOv26...")
        self.model = YOLO('yolov8n.pt')
        self.k = 0.00025
        self.base_dir = os.path.abspath(base_dir)
        self.dirs = {
            "RUNS": os.path.join(self.base_dir, "RUNS"),
            "REPORTES": os.path.join(self.base_dir, "REPORTES"),
            "PLIMAN": os.path.join(self.base_dir, "PLIMAN")
        }
        for d in self.dirs.values():
            if not os.path.exists(d): os.makedirs(d)

    def calculate_ngrdi(self, img):
        B, G, R = cv2.split(img.astype(np.float32))
        return (G - R) / (G + R + 1e-5)

    def process_all(self, input_path):
        image_files = glob.glob(os.path.join(input_path, "*.jpg")) + glob.glob(os.path.join(input_path, "*.png"))
        image_files = image_files[:47] # Procesar las 47 imágenes
        all_stats = []

        print(f"[*] Generando reporte científico para {len(image_files)} imágenes...")

        for idx, img_path in enumerate(image_files):
            name = os.path.basename(img_path).split('.')[0]
            img = cv2.imread(img_path)
            if img is None: continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]

            # 1. YOLO (RUNS)
            results = self.model.predict(img, conf=0.15, verbose=False)
            m2_yolo = img_rgb.copy()
            detecciones = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(m2_yolo, (x1, y1), (x2, y2), (0, 255, 0), 4)
                    detecciones.append(box)
            cv2.imwrite(os.path.join(self.dirs["RUNS"], f"DET_{name}.jpg"), cv2.cvtColor(m2_yolo, cv2.COLOR_RGB2BGR))

            # 2. PLIMAN (NGRDI)
            ngrdi = self.calculate_ngrdi(img)
            norm_ngrdi = cv2.normalize(ngrdi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            m4_heatmap = cv2.applyColorMap(norm_ngrdi, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(self.dirs["PLIMAN"], f"NGRDI_{name}.jpg"), m4_heatmap)

            # 3. REPORTES (LOGICA 1x6)
            m5_grid = img_rgb.copy()
            for i in range(1, 16):
                cv2.line(m5_grid, (0, i*h//16), (w, i*h//16), (255,255,255), 1)
                cv2.line(m5_grid, (i*w//16, 0), (i*w//16, h), (255,255,255), 1)

            fig, axes = plt.subplots(1, 6, figsize=(24, 4))
            moments = [img_rgb, m2_yolo, img_rgb, cv2.cvtColor(m4_heatmap, cv2.COLOR_BGR2RGB), m5_grid, m2_yolo]
            titles = ["Original", "YOLO", "Mask", "NGRDI", "Grid", "Final"]
            for ax, m, t in zip(axes, moments, titles):
                ax.imshow(cv2.resize(m, (300, 300))); ax.set_title(t); ax.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(self.dirs["REPORTES"], f"REPORTE_1x6_{name}.png"))
            plt.close()

            # Cálculo de severidad para el CSV
            sev_promedio = 0
            if detecciones:
                for box in detecciones:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    roi_ngrdi = ngrdi[y1:y2, x1:x2]
                    sev_promedio += (np.sum(roi_ngrdi < -0.05) / (roi_ngrdi.size + 1)) * 100
                sev_promedio /= len(detecciones)

            all_stats.append({"Archivo": name, "Detecciones": len(detecciones), "Severidad_Media_%": round(sev_promedio, 2)})

        pd.DataFrame(all_stats).to_csv(os.path.join(self.dirs["PLIMAN"], "Datos_Consolidados_Estudio.csv"), index=False)
        print(f"[!] Proceso masivo completado. Carpetas listas en '{self.base_dir}'")

if __name__ == "__main__":
    valid_path = r"C:\Users\juanv\Desktop\Analisis mango\Train\Mango-Antracnosis.v1i.yolo26\valid\images"
    organizer = ScientificMangoOrganizer()
    organizer.process_all(valid_path)
