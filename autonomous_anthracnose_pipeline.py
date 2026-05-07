import cv2
import numpy as np
import pandas as pd
import os
import time
from ultralytics import YOLO, SAM
import matplotlib.pyplot as plt

class AutonomousAnthracnoseModel:
    """
    Modelo Autónomo para Diagnóstico de Antracnosis en Mangos (RGB-Only).
    Optimizado para APLIMAN y Proyectos de Investigación Agrícola.
    """
    
    def __init__(self, yolo_weights, sam_weights='sam_b.pt', output_dir='RESULTADOS_AUTONOMOS_APLIMAN'):
        print(f"[*] Inicializando Sistema Autónomo v3.0...")
        self.yolo = YOLO(yolo_weights)
        self.sam = SAM(sam_weights)
        self.output_dir = output_dir
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # Parámetros de detección y severidad
        self.conf_threshold = 0.45
        self.necrosis_threshold_ngrdi = -0.075
        self.necrosis_threshold_rgbvi = 0.1 # Valores < 0.1 indican estrés/daño
        self.min_severity_for_incidence = 2.0 # % mínimo para considerar un fruto como "incidente" (enfermo)

    def calculate_spectral_indices(self, roi_rgb):
        """ Calcula índices espectrales especializados para RGB """
        R = roi_rgb[:,:,0].astype(float)
        G = roi_rgb[:,:,1].astype(float)
        B = roi_rgb[:,:,2].astype(float)
        
        # NGRDI (Normalized Green Red Difference Index)
        ngrdi = (G - R) / (G + R + 1e-8)
        
        # RGBVI (RGB Vegetation Index) - Solicitado por el usuario
        # RGBVI = (G^2 - R*B) / (G^2 + R*B)
        rgbvi = (G**2 - R*B) / (G**2 + R*B + 1e-8)
        
        # GLI (Green Leaf Index)
        gli = (2*G - R - B) / (2*G + R + B + 1e-8)
        
        return ngrdi, rgbvi, gli

    def process_image(self, image_path):
        """ Procesa una imagen individual: Detección -> Segmentación -> Severidad """
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_display = img.copy()
        
        # 1. Detección YOLO
        results = self.yolo.predict(img, conf=self.conf_threshold, verbose=False)
        
        fruit_data = []
        
        for i, r in enumerate(results):
            for box in r.boxes:
                bbox = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, bbox)
                
                # 2. Segmentación SAM (Máscara precisa)
                sam_results = self.sam.predict(img, bboxes=[bbox], verbose=False)
                if not sam_results or sam_results[0].masks is None:
                    continue
                
                mask = sam_results[0].masks.data[0].cpu().numpy().astype(np.uint8)
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
                
                # Recorte de la máscara y ROI
                fruit_mask_roi = mask[y1:y2, x1:x2]
                roi_rgb = img_rgb[y1:y2, x1:x2]
                
                if roi_rgb.size == 0: continue
                
                # 3. Análisis Espectral (NGRDI + RGBVI)
                ngrdi, rgbvi, gli = self.calculate_spectral_indices(roi_rgb)
                
                # Definición de Necrosis (Combinada para mayor precisión)
                # Usamos NGRDI para tejido oscuro y RGBVI para verificar estrés
                mask_necrosis = (ngrdi < self.threshold_ngrdi_logic(ngrdi)) & (fruit_mask_roi > 0)
                # Opcional: Refinar con RGBVI si se desea
                # mask_necrosis = (rgbvi < self.necrosis_threshold_rgbvi) & (fruit_mask_roi > 0)
                
                area_fruto = np.sum(fruit_mask_roi > 0)
                area_necrosis = np.sum(mask_necrosis > 0)
                
                severity = (area_necrosis / area_fruto * 100) if area_fruto > 0 else 0
                
                # 4. Visualización (Grid 64x64 para teñido)
                h_roi, w_roi = roi_rgb.shape[:2]
                cell_h, cell_w = h_roi / 64, w_roi / 64
                
                roi_viz = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR)
                for row in range(64):
                    for col in range(64):
                        cy1, cy2 = int(row * cell_h), int((row + 1) * cell_h)
                        cx1, cx2 = int(col * cell_w), int((col + 1) * cell_w)
                        
                        if cy2 > h_roi: cy2 = h_roi
                        if cx2 > w_roi: cx2 = w_roi
                        
                        cell_mask = mask_necrosis[cy1:cy2, cx1:cx2]
                        cell_fruit = fruit_mask_roi[cy1:cy2, cx1:cx2]
                        
                        if np.sum(cell_fruit) > 0:
                            cell_sev = np.sum(cell_mask) / np.sum(cell_fruit)
                            color = (0, 0, 255) if cell_sev > 0.05 else (0, 255, 0)
                            alpha = 0.4 if cell_sev > 0.05 else 0.1
                            
                            sub = roi_viz[cy1:cy2, cx1:cx2]
                            overlay = np.full(sub.shape, color, dtype=np.uint8)
                            roi_viz[cy1:cy2, cx1:cx2] = cv2.addWeighted(sub, 1-alpha, overlay, alpha, 0)
                
                # Actualizar imagen general
                img_display[y1:y2, x1:x2] = roi_viz
                cv2.rectangle(img_display, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(img_display, f"Sev: {severity:.1f}%", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                fruit_data.append({
                    "Fruto_ID": i,
                    "Severidad_%": round(severity, 2),
                    "Incidencia": 1 if severity > self.min_severity_for_incidence else 0,
                    "NGRDI_Promedio": round(np.mean(ngrdi[fruit_mask_roi > 0]), 4),
                    "RGBVI_Promedio": round(np.mean(rgbvi[fruit_mask_roi > 0]), 4)
                })
                
        return img_display, fruit_data

    def threshold_ngrdi_logic(self, ngrdi):
        """ Lógica adaptativa para umbral de necrosis """
        return self.necrosis_threshold_ngrdi

    def run_batch(self, input_folder):
        """ Ejecuta el análisis en todas las imágenes de una carpeta """
        images = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"[*] Iniciando procesamiento de {len(images)} imágenes...")
        
        all_results = []
        
        for img_name in images:
            start_time = time.time()
            img_path = os.path.join(input_folder, img_name)
            
            processed_img, fruit_data = self.process_image(img_path)
            
            if processed_img is not None:
                # Guardar imagen diagnosticada
                out_img_path = os.path.join(self.output_dir, f"DIAGNOSTICO_{img_name}")
                cv2.imwrite(out_img_path, processed_img)
                
                # Consolidar datos
                for data in fruit_data:
                    data["Imagen"] = img_name
                    all_results.append(data)
            
            elapsed = time.time() - start_time
            print(f"    [+] {img_name} procesada en {elapsed:.2f}s")

        # Generar Reporte Final
        if all_results:
            df = pd.DataFrame(all_results)
            
            # Cálculos de Epidemiología (APLIMAN)
            total_frutos = len(df)
            total_enfermos = df["Incidencia"].sum()
            incidencia_global = (total_enfermos / total_frutos * 100) if total_frutos > 0 else 0
            severidad_media = df["Severidad_%"].mean()
            
            print(f"\n" + "="*40)
            print(f" RESUMEN EPIDEMIOLÓGICO (APLIMAN)")
            print(f"="*40)
            print(f" Total Frutos Analizados: {total_frutos}")
            print(f" Incidencia Global:      {incidencia_global:.2f}%")
            print(f" Severidad Media:        {severidad_media:.2f}%")
            print(f"="*40)
            
            # Guardar CSV
            csv_path = os.path.join(self.output_dir, "REPORTE_EPIDEMIOLOGICO_APLIMAN.csv")
            df.to_csv(csv_path, index=False)
            print(f"[*] Reporte guardado en: {csv_path}")
            
            # Crear Gráfica de Resumen
            self.generate_summary_plots(df)

    def generate_summary_plots(self, df):
        """ Genera visualizaciones clave para el reporte """
        plt.figure(figsize=(12, 5))
        
        # Subplot 1: Distribución de Severidad
        plt.subplot(1, 2, 1)
        plt.hist(df["Severidad_%"], bins=10, color='salmon', edgecolor='black')
        plt.title("Distribución de Severidad (%)")
        plt.xlabel("Severidad (%)")
        plt.ylabel("Frecuencia (Frutos)")
        
        # Subplot 2: NGRDI vs RGBVI
        plt.subplot(1, 2, 2)
        plt.scatter(df["NGRDI_Promedio"], df["RGBVI_Promedio"], c=df["Severidad_%"], cmap='RdYlGn_r')
        plt.colorbar(label='Severidad (%)')
        plt.title("Correlación NGRDI vs RGBVI")
        plt.xlabel("NGRDI")
        plt.ylabel("RGBVI")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "RESUMEN_GRAFICO_APLIMAN.png"))
        plt.close()

if __name__ == "__main__":
    # RUTAS CONFIGURABLES
    PESOS_YOLO = r"C:\Users\juanv\Desktop\Analisis mango\runs\detect\Entrenamiento_Cientifico\Mango_YOLOv26_NUEVO_SET3\weights\best.pt"
    # PESOS_SAM = 'sam_b.pt' # Ya está por defecto
    INPUT_DIR = r"C:\Users\juanv\Desktop\Analisis mango\TEST_APLIMAN"
    
    # Instanciar y Correr
    pipeline = AutonomousAnthracnoseModel(yolo_weights=PESOS_YOLO)
    pipeline.run_batch(INPUT_DIR)
