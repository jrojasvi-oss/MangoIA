import cv2
import numpy as np
import os
from ultralytics import YOLO, SAM

class RealisticScientificSAM:
    def __init__(self, source_path, target_path, yolo_weights, sam_weights='sam_b.pt'):
        print(f"[*] Iniciando Segmentación Científica Realista (Sin bordes, alta sutileza)...")
        self.source = os.path.abspath(source_path)
        self.target = os.path.abspath(target_path)
        
        if not os.path.exists(self.target):
            os.makedirs(self.target)
            
        print(f"[*] Cargando YOLO: {yolo_weights}")
        self.yolo = YOLO(yolo_weights)
        
        print(f"[*] Cargando SAM: {sam_weights}")
        self.sam = SAM(sam_weights)
        
        # Colores científicos sutiles
        self.colors = {
            'Mango-Antracnosis': (0, 0, 200),    # Rojo profundo
            'Mango-afectado': (0, 70, 200),      # Naranja/Rojo sutil
            'Mango-area-afectada': (50, 50, 220), # Rojo suave
            'Mango-sano': (0, 180, 0)            # Verde bosque
        }
        
        # Parámetros Espectrales
        self.threshold_ngrdi = -0.075

    def calculate_ngrdi(self, roi_rgb):
        R = roi_rgb[:,:,0].astype(float)
        G = roi_rgb[:,:,1].astype(float)
        return (G - R) / (G + R + 1e-6)

    def analyze_grid_64x64(self, img, mask):
        """ Análisis tipo Swin-Windowing para severidad real """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ngrdi = self.calculate_ngrdi(img_rgb)
        
        h, w = img.shape[:2]
        grid_size = 64
        ch, cw = h // grid_size, w // grid_size
        
        if ch == 0 or cw == 0: return 0
        
        total_valid = 0
        total_necrosis = 0
        
        for i in range(grid_size):
            for j in range(grid_size):
                y1, y2 = i*ch, (i+1)*ch
                x1, x2 = j*cw, (j+1)*cw
                
                cell_mask = mask[y1:y2, x1:x2]
                if np.mean(cell_mask) < 0.4: continue # Filtro de borde
                
                cell_ngrdi = ngrdi[y1:y2, x1:x2]
                # Filtro de ruido Swin (contexto local)
                if np.var(cell_ngrdi) > 0.04: continue 
                
                nec_area = (cell_ngrdi < self.threshold_ngrdi).sum()
                cell_area = (cell_mask > 0).sum()
                
                if cell_area > 0:
                    total_necrosis += (nec_area / cell_area)
                    total_valid += 1
                    
        return (total_necrosis / total_valid * 100) if total_valid > 0 else 0

    def process_all(self):
        print(f"[*] Procesando recursivamente: {self.source}")
        
        for root, dirs, files in os.walk(self.source):
            # Omitir carpetas de resultados previos
            if any(x in root.upper() for x in ["RESULTADOS", "FINAL", "SAM", "YOLO"]): continue
            
            rel_path = os.path.relpath(root, self.source)
            target_dir = os.path.join(self.target, rel_path)
            
            images = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not images: continue
            
            if not os.path.exists(target_dir): os.makedirs(target_dir)
            print(f"[*] Segmentando {len(images)} imágenes en: {rel_path}")

            for img_name in images:
                img_path = os.path.join(root, img_name)
                img = cv2.imread(img_path)
                if img is None: continue
                
                h, w = img.shape[:2]
                results_yolo = self.yolo.predict(img, conf=0.30, iou=0.45, verbose=False)
                
                # Fondo negro puro
                final_canvas = np.zeros_like(img)
                
                bboxes = []
                class_names = []
                for r in results_yolo:
                    for box in r.boxes:
                        bboxes.append(box.xyxy[0].tolist())
                        class_names.append(r.names[int(box.cls[0])])
                
                if bboxes:
                    results_sam = self.sam.predict(img, bboxes=bboxes, verbose=False)
                    
                    for i, r_sam in enumerate(results_sam):
                        if r_sam.masks is not None:
                            # Máscara precisa con SAM
                            mask = r_sam.masks.data[0].cpu().numpy().astype(np.uint8)
                            mask = cv2.resize(mask, (w, h))
                            mask_indices = mask > 0.5
                            
                            class_name = class_names[i]
                            color = self.colors.get(class_name, (0, 150, 0))
                            
                            # Realismo: Mezcla sutil sin bordes de caja
                            # Alpha bajo para "sutileza"
                            alpha = 0.45 if 'sano' in class_name.lower() else 0.65
                            
                            # Capa de color
                            color_layer = np.zeros_like(img)
                            color_layer[mask_indices] = color
                            
                            # Cálculo de Severidad Real (Grid 64x64)
                            real_severity = self.analyze_grid_64x64(img, mask)
                            
                            # Mezcla con textura original
                            roi_texture = img.copy()
                            blended = cv2.addWeighted(roi_texture, 1-alpha, color_layer, alpha, 0)
                            
                            # Añadir texto de severidad
                            cv2.putText(blended, f"Sev: {real_severity:.1f}%", (bboxes[i][0].astype(int), bboxes[i][1].astype(int)-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            
                            # Transferir al canvas final (fondo negro conservado fuera de la máscara)
                            final_canvas[mask_indices] = blended[mask_indices]

                # Guardar resultado científico
                cv2.imwrite(os.path.join(target_dir, f"REALISTIC_SAM_{img_name}"), final_canvas)

        print(f"\n[!] SEGMENTACIÓN REALISTA COMPLETADA EN TODA LA RAÍZ.")

if __name__ == "__main__":
    YOLO_W = r"C:\Users\juanv\Desktop\Analisis mango\runs\detect\Entrenamiento_Cientifico\Mango_YOLOv26_NUEVO_SET3\weights\best.pt"
    SAM_W = "sam_b.pt"
    # Raíz de datos 'fruto'
    SOURCE = r"C:\Users\juanv\Desktop\Analisis mango\fruto\fruto"
    TARGET = r"C:\Users\juanv\Desktop\Analisis mango\FRUTO_SEGMENTACION_REALISTA_FINAL"
    
    processor = RealisticScientificSAM(SOURCE, TARGET, YOLO_W, SAM_W)
    processor.process_all()
