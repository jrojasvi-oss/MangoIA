import cv2
import numpy as np
import os
import pandas as pd
from ultralytics import YOLO, SAM

class MasterRobustIntegrator:
    def __init__(self, source_path, target_path, yolo_weights, sam_weights='sam_b.pt'):
        print(f"[*] Iniciando Integración Robusta Maestra (YOLO + SAM + Grid Científico)...")
        self.source = os.path.abspath(source_path)
        self.target = os.path.abspath(target_path)
        
        if not os.path.exists(self.target):
            os.makedirs(self.target)
            
        self.yolo = YOLO(yolo_weights)
        self.sam = SAM(sam_weights)
        
    def analyze_ngrdi_grid(self, roi_img, mask, grid_size=64):
        """ Aplica análisis NGRDI solo sobre la zona segmentada por SAM """
        h, w = roi_img.shape[:2]
        cell_h, cell_w = h / grid_size, w / grid_size
        
        # Análisis NGRDI
        R = roi_img[:,:,0].astype(float)
        G = roi_img[:,:,1].astype(float)
        ngrdi = (G - R) / (G + R + 1e-5)
        
        # Máscara de necrosis basada en NGRDI (Lógica inicial)
        mask_necrosis = (ngrdi < -0.075).astype(np.uint8) * 255
        # Intersección con la máscara de SAM para evitar falsos positivos fuera del fruto
        mask_real_damage = cv2.bitwise_and(mask_necrosis, mask.astype(np.uint8) * 255)
        
        output_roi = roi_img.copy()
        
        for r in range(grid_size):
            for c in range(grid_size):
                cy1, cy2 = int(r * cell_h), int((r + 1) * cell_h)
                cx1, cx2 = int(c * cell_w), int((c + 1) * cell_w)
                
                if cy2 <= cy1 or cx2 <= cx1: continue
                
                cell_damage = mask_real_damage[cy1:cy2, cx1:cx2]
                cell_fruit = (mask[cy1:cy2, cx1:cx2] > 0).sum()
                
                if cell_fruit > 0:
                    area_n = (cell_damage > 0).sum()
                    sev = (area_n / cell_fruit) * 100
                    
                    # Color según severidad (Lógica inicial robusta)
                    color = [255, 0, 0] if sev > 4 else [0, 255, 0]
                    alpha = 0.5 if sev > 4 else 0.2
                    
                    sub = output_roi[cy1:cy2, cx1:cx2]
                    colored = cv2.addWeighted(sub, 1-alpha, np.full(sub.shape, color, dtype=np.uint8), alpha, 0)
                    output_roi[cy1:cy2, cx1:cx2] = colored
                    
        total_afe = round(((mask_real_damage > 0).sum() / ((mask > 0).sum() + 1e-5)) * 100, 2)
        return total_afe, output_roi

    def process_all(self):
        results_data = []
        
        for root, dirs, files in os.walk(self.source):
            if any(x in root.upper() for x in ["RESULTADOS", "FINAL", "SAM", "YOLO"]): continue
            
            rel_path = os.path.relpath(root, self.source)
            target_dir = os.path.join(self.target, rel_path)
            
            images = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not images: continue
            
            if not os.path.exists(target_dir): os.makedirs(target_dir)
            print(f"[*] Procesando {len(images)} imágenes en: {rel_path}")

            for img_name in images:
                img_path = os.path.join(root, img_name)
                img = cv2.imread(img_path)
                if img is None: continue
                
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w = img.shape[:2]
                
                # 1. Detectar frutos con YOLO
                results_yolo = self.yolo.predict(img, conf=0.45, iou=0.5, verbose=False)
                
                final_img = img.copy()
                
                for r in results_yolo:
                    for box in r.boxes:
                        bbox = box.xyxy[0].tolist()
                        x1, y1, x2, y2 = map(int, bbox)
                        
                        # 2. Segmentar con SAM para precisión de borde
                        results_sam = self.sam.predict(img, bboxes=[bbox], verbose=False)
                        
                        if results_sam[0].masks is not None:
                            # Convertir máscara de bool a uint8 para resize
                            mask = results_sam[0].masks.data[0].cpu().numpy().astype(np.uint8)
                            mask = cv2.resize(mask, (w, h))
                            
                            # 3. Aplicar lógica de Grid inicial solo dentro de la máscara de SAM
                            roi_rgb = img_rgb[y1:y2, x1:x2]
                            roi_mask = mask[y1:y2, x1:x2]
                            
                            afe, visual_roi_rgb = self.analyze_ngrdi_grid(roi_rgb, roi_mask)
                            
                            # Reinsertar ROI procesado
                            visual_roi_bgr = cv2.cvtColor(visual_roi_rgb, cv2.COLOR_RGB2BGR)
                            final_img[y1:y2, x1:x2] = visual_roi_bgr
                            
                            # Etiquetado robusto
                            label = f"Afeccion: {afe}%"
                            cv2.rectangle(final_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
                            cv2.putText(final_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                            
                            results_data.append({
                                "Imagen": img_name,
                                "Carpeta": rel_path,
                                "Afectacion_%": afe
                            })

                cv2.imwrite(os.path.join(target_dir, f"ROBUST_INTEGRATED_{img_name}"), final_img)

        pd.DataFrame(results_data).to_csv(os.path.join(self.target, "CONSOLIDADO_ROBUSTO_INTEGRADO.csv"), index=False)
        print(f"[!] PROCESO ROBUSTO FINALIZADO.")

if __name__ == "__main__":
    YOLO_W = r"C:\Users\juanv\Desktop\Analisis mango\runs\detect\Entrenamiento_Cientifico\Mango_YOLOv26_NUEVO_SET3\weights\best.pt"
    SOURCE = r"C:\Users\juanv\Desktop\Analisis mango\fruto\fruto"
    TARGET = r"C:\Users\juanv\Desktop\Analisis mango\FRUTO_INTEGRACION_ROBUSTA_MAESTRA"
    
    integrator = MasterRobustIntegrator(SOURCE, TARGET, YOLO_W)
    integrator.process_all()
