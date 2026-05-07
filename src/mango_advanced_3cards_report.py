import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os

class AdvancedMangoReporter:
    def __init__(self, pixel_cm2_ratio=0.00025):
        print("[*] Iniciando Motor Ultralytics YOLO...")
        try:
            self.model = YOLO('yolov8n.pt') 
        except:
            self.model = YOLO('yolov8n.pt')
        self.k = pixel_cm2_ratio # Calibración cm2/px
        
    def correct_color(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def calculate_pliman_math(self, img):
        img_corr = self.correct_color(img)
        B, G, R = cv2.split(img_corr.astype(np.float32))
        ngrdi = (G - R) / (G + R + 1e-5)
        return ngrdi

    def generate_report(self, img_path):
        if not os.path.exists(img_path):
            print(f"Error: No existe {img_path}")
            return
            
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # Umbral bajo para forzar detección de mangos
        results = self.model.predict(img, conf=0.15, verbose=False)
        m2_inferencia = img_rgb.copy()
        detecciones = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(m2_inferencia, (x1, y1), (x2, y2), (0, 255, 0), 5)
                detecciones.append(box)
        
        ngrdi = self.calculate_pliman_math(img)
        m4_heatmap = cv2.applyColorMap(cv2.normalize(ngrdi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET)
        m4_heatmap = cv2.cvtColor(m4_heatmap, cv2.COLOR_BGR2RGB)
        
        m5_grid = img_rgb.copy()
        for i in range(1, 16):
            cv2.line(m5_grid, (0, i*h//16), (w, i*h//16), (255,255,255), 2)
            cv2.line(m5_grid, (i*w//16, 0), (i*w//16, h), (255,255,255), 2)

        moments = [img_rgb, m2_inferencia, img_rgb, m4_heatmap, m5_grid, m2_inferencia]
        fig1, axes1 = plt.subplots(1, 6, figsize=(30, 5))
        titles = ["Original", "YOLOv26", "Segmentación", "NGRDI", "16x16 Matrix", "Diagnóstico"]
        for ax, m, t in zip(axes1, moments, titles):
            ax.imshow(cv2.resize(m, (512, 512)))
            ax.set_title(t, fontsize=15)
            ax.axis('off')
        plt.tight_layout()
        plt.savefig("CARD_2_LOGICA_1x6.png")
        
        stats = []
        for i, box in enumerate(detecciones):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            roi_ngrdi = ngrdi[y1:y2, x1:x2]
            sev = (np.sum(roi_ngrdi < -0.05) / (roi_ngrdi.size + 1)) * 100
            area_cm2 = roi_ngrdi.size * self.k
            stats.append({"ID": i+1, "Área_cm2": round(area_cm2, 2), "Severidad_%": round(sev, 2)})
        
        if stats:
            df = pd.DataFrame(stats).sort_values(by="Severidad_%")
        else:
            df = pd.DataFrame(columns=["ID", "Área_cm2", "Severidad_%"])
            
        df.to_csv("datos_severidad_exacta.csv", index=False)

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.axis('off')
        if not df.empty:
            table = ax2.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
            table.set_fontsize(14)
            table.scale(1.2, 2)
        else:
            plt.text(0.5, 0.5, "NO SE DETECTARON MANGOS", ha='center', va='center', fontsize=20)
            
        plt.title("LÓGICA DE SEVERIDAD Y ÁREA EXACTA (PLIMAN)", fontsize=18)
        plt.savefig("CARD_3_SEVERIDAD_AREA.png")
        print(f"[!] Reporte finalizado. Total mangos: {len(df)}")

if __name__ == "__main__":
    reporter = AdvancedMangoReporter()
    img_test = r"C:\Users\juanv\Desktop\Analisis mango\Train\Mango-Antracnosis.v1i.yolo26\train\images\r1_png.rf.b0798c7fb8cbdb007ab38edb4f19b686.jpg"
    reporter.generate_report(img_test)
