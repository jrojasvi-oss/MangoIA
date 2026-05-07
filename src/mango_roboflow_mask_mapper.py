import cv2
import numpy as np
import pandas as pd
import os
import glob

class RoboflowCoordinateSegmenter:
    def __init__(self, dataset_path, output_path):
        print("[*] Iniciando Segmentación Técnica por Coordenadas de Roboflow...")
        self.dataset_path = dataset_path
        self.output_path = output_path
        # Clases según tu data.yaml:
        # 0: Mango-Antracnosis, 1: Mango-afectado, 2: Mango-area-afectada, 3: Mango-sano
        self.affected_classes = [0, 1, 2]
        self.healthy_classes = [3]
        if not os.path.exists(output_path): os.makedirs(output_path)

    def parse_all_layers(self, label_path, img_w, img_h):
        """ Extrae capas de salud y daño basadas en coordenadas exactas """
        layers = {"afectado": [], "sano": []}
        if not os.path.exists(label_path): return layers
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                if len(parts) < 5: continue
                
                cls_id = int(parts[0])
                poly = np.array(parts[1:]).reshape(-1, 2)
                poly[:, 0] *= img_w
                poly[:, 1] *= img_h
                poly_coords = poly.astype(np.int32)
                
                if cls_id in self.affected_classes:
                    layers["afectado"].append(poly_coords)
                elif cls_id in self.healthy_classes:
                    layers["sano"].append(poly_coords)
        return layers

    def segment_by_coordinates(self, limit=30):
        img_paths = glob.glob(os.path.join(self.dataset_path, "train", "images", "*.jpg"))
        
        for img_path in img_paths[:limit]:
            img = cv2.imread(img_path)
            if img is None: continue
            h, w = img.shape[:2]
            
            label_path = img_path.replace("images", "labels").replace(".jpg", ".txt")
            layers = self.parse_all_layers(label_path, w, h)
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            final_viz = img_rgb.copy()
            
            # 1. Crear máscaras maestras desde coordenadas
            mask_sano = np.zeros((h, w), dtype=np.uint8)
            mask_afe = np.zeros((h, w), dtype=np.uint8)
            
            for p in layers["sano"]: cv2.fillPoly(mask_sano, [p], 255)
            for p in layers["afectado"]: cv2.fillPoly(mask_afe, [p], 255)
            
            # 2. Aplicar Malla 64x64 sobre la unión de ambas (El Mango Completo)
            full_mango_mask = cv2.bitwise_or(mask_sano, mask_afe)
            x, y, mw, mh = cv2.boundingRect(full_mango_mask)
            
            if mw == 0 or mh == 0: continue
            
            cell_h, cell_w = mh / 64, mw / 64
            for r in range(64):
                for c in range(64):
                    cy1, cy2 = int(y + r * cell_h), int(y + (r+1) * cell_h)
                    cx1, cx2 = int(x + c * cell_w), int(x + (c+1) * cell_w)
                    
                    # Decisión por Coordenada Exacta
                    if np.any(mask_afe[cy1:cy2, cx1:cx2] > 0):
                        color = [255, 0, 0] # ROJO: Coordenada de Daño
                        alpha = 0.5
                    elif np.any(mask_sano[cy1:cy2, cx1:cx2] > 0):
                        color = [0, 255, 0] # VERDE: Coordenada de Salud
                        alpha = 0.3
                    else:
                        continue
                    
                    sub = final_viz[cy1:cy2, cx1:cx2]
                    if sub.size > 0:
                        colored = cv2.addWeighted(sub, 1-alpha, np.full(sub.shape, color, dtype=np.uint8), alpha, 0)
                        final_viz[cy1:cy2, cx1:cx2] = colored

            # Guardar visualización de segmentación por coordenadas
            out_name = f"COORD_SEGMENT_{os.path.basename(img_path)}"
            cv2.imwrite(os.path.join(self.output_path, out_name), cv2.cvtColor(final_viz, cv2.COLOR_RGB2BGR))

        print(f"[!] SEGMENTACIÓN POR COORDENADAS COMPLETADA en {self.output_path}")

if __name__ == "__main__":
    path_rf = r"C:\Users\juanv\Desktop\Analisis mango\Train\Mango-Antracnosis.v1i.yolo26"
    output = r"C:\Users\juanv\Desktop\Analisis mango\SEGMENTACION_COORDENADAS_PURA"
    segmenter = RoboflowCoordinateSegmenter(path_rf, output)
    segmenter.segment_by_coordinates(limit=30)
