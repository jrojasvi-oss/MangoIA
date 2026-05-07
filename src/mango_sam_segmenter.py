import cv2
import numpy as np
import os
from ultralytics import YOLO, SAM

class MangoSAMSegmenter:
    def __init__(self, source_path, target_path, yolo_weights, sam_weights='sam_b.pt'):
        print(f"[*] Iniciando Segmentación Avanzada SAM + YOLO...")
        self.source = os.path.abspath(source_path)
        self.target = os.path.abspath(target_path)
        
        if not os.path.exists(self.target):
            os.makedirs(self.target)
            
        print(f"[*] Cargando modelo YOLO: {yolo_weights}")
        self.yolo = YOLO(yolo_weights)
        
        print(f"[*] Cargando modelo SAM: {sam_weights}")
        # SAM de Ultralytics es muy potente para segmentación por prompts de caja
        self.sam = SAM(sam_weights)
        
        # Colores solicitados: Verdes para sano, Rojos para afectados
        self.colors = {
            'Mango-Antracnosis': (0, 0, 255),    # ROJO
            'Mango-afectado': (0, 100, 255),     # NARANJA/ROJO
            'Mango-area-afectada': (50, 50, 255), # ROJO CLARO
            'Mango-sano': (0, 255, 0)            # VERDE
        }

    def apply_mask(self, image, mask, color, alpha=0.5):
        """ Aplica una máscara coloreada sobre la imagen """
        mask_indices = mask > 0
        image[mask_indices] = image[mask_indices] * (1 - alpha) + np.array(color) * alpha
        return image

    def process_images(self):
        print(f"[*] Escaneando: {self.source}")
        
        for root, dirs, files in os.walk(self.source):
            rel_path = os.path.relpath(root, self.source)
            target_dir = os.path.join(self.target, rel_path)
            
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            
            images = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not images: continue
            
            print(f"[*] Segmentando {len(images)} imágenes en: {rel_path}")

            for img_name in images:
                img_path = os.path.join(root, img_name)
                img = cv2.imread(img_path)
                if img is None: continue
                
                h, w = img.shape[:2]
                
                # 1. Identificación inicial con YOLO
                results_yolo = self.yolo.predict(img, conf=0.35, iou=0.45, verbose=False)
                
                # Crear lienzo negro para omitir el fondo
                final_output = np.zeros_like(img)
                
                # Agrupar detecciones para pasarlas a SAM
                bboxes = []
                class_names = []
                
                for r in results_yolo:
                    names = r.names
                    for box in r.boxes:
                        bboxes.append(box.xyxy[0].tolist())
                        class_names.append(names[int(box.cls[0])])
                
                if bboxes:
                    # 2. Segmentación con SAM usando las cajas de YOLO como prompt
                    # Esto asegura que SAM se centre "específicamente" en el objeto
                    results_sam = self.sam.predict(img, bboxes=bboxes, verbose=False)
                    
                    for i, r_sam in enumerate(results_sam):
                        if r_sam.masks is not None:
                            # Obtener la máscara binaria y convertir a uint8 para resize
                            mask = r_sam.masks.data[0].cpu().numpy().astype(np.uint8)
                            mask = cv2.resize(mask, (w, h))
                            
                            class_name = class_names[i]
                            color = self.colors.get(class_name, (0, 255, 0))
                            
                            # Si es un fruto (sano o afectado), lo usamos para rescatar la textura original
                            # pero con el tinte del color solicitado
                            if 'Mango' in class_name:
                                # Aplicar color a la zona segmentada sobre fondo negro
                                mask_indices = mask > 0.5
                                
                                # Para el color sano (Verde) y afectado (Rojo)
                                alpha = 0.6 if 'sano' in class_name.lower() else 0.8
                                
                                # Extraer la textura original para que no sea solo un manchón
                                texture = img.copy()
                                colored_mask = np.zeros_like(img)
                                colored_mask[mask_indices] = color
                                
                                # Mezclar textura original con el color
                                blended = cv2.addWeighted(texture, 1-alpha, colored_mask, alpha, 0)
                                
                                # Pegar en el resultado final
                                final_output[mask_indices] = blended[mask_indices]

                # Guardar resultado con fondo omitido
                cv2.imwrite(os.path.join(target_dir, f"SAM_{img_name}"), final_output)

        print(f"\n[!] SEGMENTACIÓN SAM COMPLETADA.")
        print(f"[*] Resultados en: {self.target}")

if __name__ == "__main__":
    YOLO_W = r"C:\Users\juanv\Desktop\Analisis mango\runs\detect\Entrenamiento_Cientifico\Mango_YOLOv26_NUEVO_SET3\weights\best.pt"
    # Nota: sam_b.pt es el modelo base balanceado. 
    # Si no existe, Ultralytics lo descargará automáticamente.
    SAM_W = "sam_b.pt" 
    SOURCE = r"C:\Users\juanv\Desktop\Analisis mango\fruto\fruto"
    TARGET = r"C:\Users\juanv\Desktop\Analisis mango\FRUTO_SAM_SCIENTIFIC"
    
    segmenter = MangoSAMSegmenter(SOURCE, TARGET, YOLO_W, SAM_W)
    segmenter.process_images()
