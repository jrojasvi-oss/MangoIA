import cv2
import numpy as np
import os
from ultralytics import YOLO

class PureYOLODetector:
    def __init__(self, source_path, target_path, weights_path):
        print(f"[*] Iniciando Detección YOLO Pura (Sin cálculos adicionales)...")
        self.source = os.path.abspath(source_path)
        self.target = os.path.abspath(target_path)
        
        if not os.path.exists(self.target):
            os.makedirs(self.target)
            
        print(f"[*] Cargando modelo YOLO Científico: {weights_path}")
        self.model = YOLO(weights_path)
        
        # Colores para las clases (Cyan/Blue para mango, Blanco/Rojo para afectaciones)
        self.colors = {
            'Mango-Antracnosis': (0, 0, 255),    # Rojo
            'Mango-afectado': (0, 165, 255),    # Naranja
            'Mango-area-afectada': (255, 255, 255), # Blanco
            'Mango-sano': (255, 255, 0)         # Cyan/Azul claro
        }

    def draw_professional_box(self, img, box, label, confidence, class_name):
        x1, y1, x2, y2 = map(int, box)
        color = self.colors.get(class_name, (255, 255, 0))
        
        # Dibujar rectángulo principal (línea fina como en el screenshot)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Preparar etiqueta
        full_label = f"{class_name} {confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        # Tamaño del texto para el fondo
        (w, h), baseline = cv2.getTextSize(full_label, font, font_scale, thickness)
        
        # Dibujar fondo de la etiqueta (un poco más grande que el texto)
        cv2.rectangle(img, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
        
        # Dibujar texto en negro o blanco dependiendo del fondo
        text_color = (0, 0, 0) if np.mean(color) > 127 else (255, 255, 255)
        cv2.putText(img, full_label, (x1, y1 - 5), font, font_scale, text_color, thickness, cv2.LINE_AA)

    def run_detection(self):
        print(f"[*] Escaneando: {self.source}")
        
        for root, dirs, files in os.walk(self.source):
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
                
                # Inferencia pura sin cálculos adicionales
                # Se usa un umbral de confianza bajo para no perder nada (como pidió el usuario)
                results = self.model.predict(img, conf=0.30, iou=0.45, verbose=False)
                
                output_img = img.copy()
                
                for r in results:
                    # Acceder a las clases del modelo para mapear IDs a nombres
                    names = r.names
                    for box in r.boxes:
                        coords = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        class_name = names[cls_id]
                        
                        # Dibujar con la estética profesional solicitada
                        self.draw_professional_box(output_img, coords, class_name, conf, class_name)

                # Guardar resultado
                cv2.imwrite(os.path.join(target_dir, img_name), output_img)

        print(f"\n[!] DETECCIÓN COMPLETADA EXITOSAMENTE.")
        print(f"[*] Resultados guardados en: {self.target}")

if __name__ == "__main__":
    # Configuración de rutas
    PESOS = r"C:\Users\juanv\Desktop\Analisis mango\runs\detect\Entrenamiento_Cientifico\Mango_YOLOv26_NUEVO_SET3\weights\best.pt"
    SOURCE = r"C:\Users\juanv\Desktop\Analisis mango\fruto\fruto"
    TARGET = r"C:\Users\juanv\Desktop\Analisis mango\RESULTADOS_YOLO_PURO_PRO"
    
    detector = PureYOLODetector(SOURCE, TARGET, PESOS)
    detector.run_detection()
