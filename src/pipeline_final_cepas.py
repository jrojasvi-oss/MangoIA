import cv2
import numpy as np
import pandas as pd
import os
from ultralytics import YOLO, SAM

class CepasScientificSegmenter:
    def __init__(self, source_path, target_path, weights_path, sam_weights='sam_b.pt'):
        print(f"[*] Iniciando Segmentación Científica con YOLO+SAM en Cepas...")
        self.source = os.path.abspath(source_path)
        self.target = os.path.abspath(target_path)
        
        if not os.path.exists(self.target):
            os.makedirs(self.target)
            
        print(f"[*] Cargando modelo YOLO: {weights_path}")
        self.yolo = YOLO(weights_path)
        print(f"[*] Cargando modelo SAM: {sam_weights}")
        self.sam = SAM(sam_weights)
        
    def apply_sam_segmentation(self, img_bgr, bbox):
        """ Utiliza SAM para aislar el fruto perfectamente basado en el BBox de YOLO """
        results = self.sam.predict(img_bgr, bboxes=[bbox], verbose=False)
        if not results or results[0].masks is None:
            return None
        mask = results[0].masks.data[0].cpu().numpy().astype(np.uint8)
        h, w = img_bgr.shape[:2]
        mask = cv2.resize(mask, (w, h))
        return mask

    def apply_scientific_segmentation(self, img_rgb, roi_box, sam_mask):
        """ Aplica malla 64x64 con teñido binario usando la máscara de SAM """
        x1, y1, x2, y2 = roi_box
        roi = img_rgb[y1:y2, x1:x2]
        if roi.size == 0 or sam_mask is None: return 0, 0, None
        
        mask_fruto_roi = sam_mask[y1:y2, x1:x2]
        h, w = roi.shape[:2]
        cell_h, cell_w = h / 64, w / 64
        
        # Análisis NGRDI para detectar tejido oscuro (Necrosis)
        R = roi[:,:,0].astype(float)
        G = roi[:,:,1].astype(float)
        ngrdi = (G - R) / (G + R + 1e-5)
        
        # Necrosis estricta: Píxeles con NGRDI bajo dentro del área de SAM
        mask_necrosis = (ngrdi < -0.075).astype(np.uint8) * 255
        mask_real_damage = cv2.bitwise_and(mask_necrosis, mask_fruto_roi * 255)
        
        visual_roi = roi.copy()
        total_celdas_validas = 0
        total_severidad_grid = 0
        
        for r in range(64):
            for c in range(64):
                cy1, cy2 = int(r * cell_h), int((r + 1) * cell_h)
                cx1, cx2 = int(c * cell_w), int((c + 1) * cell_w)
                
                # Evitar desbordamiento
                cy2 = min(cy2, h); cx2 = min(cx2, w)
                if cy2 <= cy1 or cx2 <= cx1: continue
                
                cell_mask = mask_real_damage[cy1:cy2, cx1:cx2]
                cell_fruto = mask_fruto_roi[cy1:cy2, cx1:cx2]
                
                area_f = np.sum(cell_fruto > 0)
                if area_f > 0: # Solo evaluar celdas que contienen fruto según SAM
                    area_n = np.sum(cell_mask > 0)
                    sev = (area_n / area_f) * 100
                    
                    total_celdas_validas += 1
                    total_severidad_grid += (area_n / area_f)
                    
                    # Lógica de Color Binario
                    if sev > 4: # Umbral de detección de mancha
                        color = [255, 0, 0] # ROJO: Afectado
                        alpha = 0.5
                    else:
                        color = [0, 255, 0] # VERDE: Sano
                        alpha = 0.2
                        
                    sub = visual_roi[cy1:cy2, cx1:cx2]
                    colored = cv2.addWeighted(sub, 1-alpha, np.full(sub.shape, color, dtype=np.uint8), alpha, 0)
                    visual_roi[cy1:cy2, cx1:cx2] = colored
        
        # Área total usando la máscara global
        total_area_fruto = np.sum(mask_fruto_roi > 0)
        total_area_necrosis = np.sum(mask_real_damage > 0)
        
        if total_area_fruto == 0:
            return 0, 0, visual_roi
            
        total_afe = round((total_area_necrosis / total_area_fruto) * 100, 2)
        return 100 - total_afe, total_afe, visual_roi

    def run_massive_analysis(self):
        master_results = []
        
        print(f"[*] Escaneando directorio origen: {self.source}")
        for root, dirs, files in os.walk(self.source):
            if any(x in root for x in ["FRUTO_CLON", "FRUTO_ANALISIS", "ANALISIS_ROBOFLOW", "RESULTADOS", "FINAL"]): 
                continue
            
            rel_path = os.path.relpath(root, self.source)
            target_dir = os.path.join(self.target, rel_path)
            if not os.path.exists(target_dir): 
                os.makedirs(target_dir)
            
            images = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not images: 
                continue
            
            print(f"[*] Procesando {len(images)} imágenes en: {rel_path}")

            for img_name in images:
                img_path = os.path.join(root, img_name)
                img = cv2.imread(img_path)
                if img is None: continue
                
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_final = img_rgb.copy()
                
                # Predicción con YOLO para localizar el fruto
                results = self.yolo.predict(img, conf=0.45, iou=0.5, verbose=False)
                
                detected = False
                for r in results:
                    for box in r.boxes:
                        detected = True
                        bbox = box.xyxy[0].tolist()
                        x1, y1, x2, y2 = map(int, bbox)
                        
                        # Obtener segmentación precisa con SAM
                        sam_mask = self.apply_sam_segmentation(img, bbox)
                        
                        sano, afe, visual_roi = self.apply_scientific_segmentation(img_rgb, (x1, y1, x2, y2), sam_mask)
                        
                        if visual_roi is not None:
                            img_final[y1:y2, x1:x2] = visual_roi
                            cv2.rectangle(img_final, (x1, y1), (x2, y2), (255, 255, 255), 2)
                            label = f"Sano: {sano}% | Afectado: {afe}%"
                            cv2.putText(img_final, label, (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                            
                            master_results.append({
                                "Imagen": img_name,
                                "Carpeta": rel_path,
                                "Salud_%": sano,
                                "Afectado_%": afe,
                                "Status": "Detectado"
                            })
                
                if not detected:
                    master_results.append({
                        "Imagen": img_name,
                        "Carpeta": rel_path,
                        "Salud_%": 0,
                        "Afectado_%": 0,
                        "Status": "No Detectado"
                    })

                output_filename = f"ANALISIS_{img_name}"
                cv2.imwrite(os.path.join(target_dir, output_filename), cv2.cvtColor(img_final, cv2.COLOR_RGB2BGR))

        csv_path = os.path.join(self.target, "CONSOLIDADO_CEPAS_PROCESADAS.csv")
        pd.DataFrame(master_results).to_csv(csv_path, index=False)
        
        # Guardar también en la carpeta runs de la raíz del proyecto
        proyecto_root = os.path.dirname(os.path.dirname(self.source)) # C:\Users\juanv\Desktop\Analisis mango
        runs_dir = os.path.join(proyecto_root, 'runs')
        if not os.path.exists(runs_dir):
            os.makedirs(runs_dir)
        csv_runs_path = os.path.join(runs_dir, "NUEVO_CONSOLIDADO_SAM_YOLO.csv")
        pd.DataFrame(master_results).to_csv(csv_runs_path, index=False)
        
        print(f"\n[!] ANÁLISIS FINALIZADO EXITOSAMENTE.")
        print(f"[*] Carpeta de salida principal: {self.target}")
        print(f"[*] Consolidado principal: {csv_path}")
        print(f"[*] Nuevo archivo en runs: {csv_runs_path}")

if __name__ == "__main__":
    PESOS_YOLO = r"C:\Users\juanv\Desktop\Analisis mango\runs\detect\Entrenamiento_Cientifico\Mango_YOLOv26_NUEVO_SET3\weights\best.pt"
    SOURCE_DIR = r"C:\Users\juanv\Desktop\Analisis mango\fruto\fruto"
    TARGET_DIR = r"C:\Users\juanv\Desktop\Analisis mango\FRUTO_CEPAS_CIENTIFICO_FINAL"
    
    segmenter = CepasScientificSegmenter(SOURCE_DIR, TARGET_DIR, PESOS_YOLO)
    segmenter.run_massive_analysis()

