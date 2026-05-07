import cv2
import numpy as np
import pandas as pd
import os
from ultralytics import YOLO

class MasterScientificSegmenter:
    def __init__(self, source_path, target_path, weights_path):
        print(f"[*] Iniciando Segmentación Científica Masiva en Dataset Original...")
        self.source = os.path.abspath(source_path)
        self.target = os.path.abspath(target_path)
        self.model = YOLO(weights_path)
        
    def apply_scientific_segmentation(self, img_rgb, roi_box):
        """ Aplica malla 64x64 con teñido binario (Verde/Rojo) sobre el ROI """
        x1, y1, x2, y2 = roi_box
        roi = img_rgb[y1:y2, x1:x2]
        if roi.size == 0: return 0, 0, None
        
        h, w = roi.shape[:2]
        cell_h, cell_w = h / 64, w / 64
        
        # Análisis NGRDI para detectar tejido oscuro (Necrosis)
        R = roi[:,:,0].astype(float)
        G = roi[:,:,1].astype(float)
        ngrdi = (G - R) / (G + R + 1e-5)
        
        # Máscaras de aislamiento del fruto
        mask_fruto = (ngrdi > -0.12).astype(np.uint8) * 255
        # Necrosis estricta: Píxeles negros/cafés
        mask_necrosis = (ngrdi < -0.075).astype(np.uint8) * 255
        mask_real_damage = cv2.bitwise_and(mask_necrosis, mask_fruto)
        
        visual_roi = roi.copy()
        
        for r in range(64):
            for c in range(64):
                cy1, cy2 = int(r * cell_h), int((r + 1) * cell_h)
                cx1, cx2 = int(c * cell_w), int((c + 1) * cell_w)
                
                # Evitar desbordamiento
                cy2 = min(cy2, h); cx2 = min(cx2, w)
                if cy2 <= cy1 or cx2 <= cx1: continue
                
                cell_mask = mask_real_damage[cy1:cy2, cx1:cx2]
                cell_fruto = mask_fruto[cy1:cy2, cx1:cx2]
                
                area_f = np.sum(cell_fruto > 0)
                if area_f > 0:
                    area_n = np.sum(cell_mask > 0)
                    sev = (area_n / area_f) * 100
                    
                    # Lógica de Color Binario para el Dataset Original
                    if sev > 4: # Umbral de detección de mancha
                        color = [255, 0, 0] # ROJO: Afectado
                        alpha = 0.5
                    else:
                        color = [0, 255, 0] # VERDE: Sano
                        alpha = 0.2
                        
                    sub = visual_roi[cy1:cy2, cx1:cx2]
                    colored = cv2.addWeighted(sub, 1-alpha, np.full(sub.shape, color, dtype=np.uint8), alpha, 0)
                    visual_roi[cy1:cy2, cx1:cx2] = colored
        
        total_afe = round((np.sum(mask_real_damage > 0) / (np.sum(mask_fruto > 0) + 1e-5)) * 100, 2)
        return 100 - total_afe, total_afe, visual_roi

    def run_massive_analysis(self):
        master_results = []
        
        for root, dirs, files in os.walk(self.source):
            if any(x in root for x in ["FRUTO_CLON", "FRUTO_ANALISIS", "ANALISIS_ROBOFLOW"]): continue
            
            rel_path = os.path.relpath(root, self.source)
            target_dir = os.path.join(self.target, rel_path)
            if not os.path.exists(target_dir): os.makedirs(target_dir)
            
            images = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not images: continue
            
            print(f"[*] Procesando Segmentación en: {rel_path} ({len(images)} fotos)")

            for img_name in images:
                img_path = os.path.join(root, img_name)
                img = cv2.imread(img_path)
                if img is None: continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                results = self.model.predict(img, conf=0.45, iou=0.5, verbose=False)
                img_final = img_rgb.copy()
                
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        sano, afe, visual_roi = self.apply_scientific_segmentation(img_rgb, (x1, y1, x2, y2))
                        
                        if visual_roi is not None:
                            img_final[y1:y2, x1:x2] = visual_roi
                            cv2.rectangle(img_final, (x1, y1), (x2, y2), (255, 255, 255), 2)
                            label = f"Salud: {sano}% | Afectado: {afe}%"
                            cv2.putText(img_final, label, (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                            
                            master_results.append({
                                "Imagen": img_name,
                                "Tratamiento": rel_path,
                                "Salud_%": sano,
                                "Afectado_%": afe
                            })

                cv2.imwrite(os.path.join(target_dir, f"SEGMENT_{img_name}"), cv2.cvtColor(img_final, cv2.COLOR_RGB2BGR))

        pd.DataFrame(master_results).to_csv(os.path.join(self.target, "CONSOLIDADO_SEGMENTACION_FINAL.csv"), index=False)
        print(f"\n[!] ANÁLISIS MASIVO COMPLETADO.")
        print(f"Resultados en: {self.target}")

if __name__ == "__main__":
    pesos = r"C:\Users\juanv\Desktop\Analisis mango\runs\detect\Entrenamiento_Cientifico\Mango_YOLOv26_NUEVO_SET3\weights\best.pt"
    source = r"C:\Users\juanv\Desktop\Analisis mango\fruto"
    target = r"C:\Users\juanv\Desktop\Analisis mango\FRUTO_SEGMENTACION_FINAL"
    
    segmenter = MasterScientificSegmenter(source, target, pesos)
    segmenter.run_massive_analysis()
