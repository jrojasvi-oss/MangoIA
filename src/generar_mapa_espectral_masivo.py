import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from ultralytics import YOLO, SAM

def aplicar_mapa_calor_masivo(source_dir, output_dir, yolo_weights, sam_weights='sam_b.pt'):
    """
    Recorre el dataset completo, aísla el fruto usando YOLO+SAM (etiquetado previo)
    y genera el diagnóstico visual con capas espectrales.
    """
    print("[*] Iniciando proceso Masivo de Mapas Espectrales (YOLO + SAM + Heatmaps)...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("[*] Cargando modelo YOLO y SAM...")
    yolo_model = YOLO(yolo_weights)
    sam_model = SAM(sam_weights)

    # Buscar todas las imágenes
    image_paths = []
    for root, _, files in os.walk(source_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, f))
                
    print(f"[*] Se han encontrado {len(image_paths)} imágenes para mapeo espectral.")

    for img_path in tqdm(image_paths, desc="Generando Mapas Espectrales"):
        img = cv2.imread(img_path)
        if img is None: continue

        # YOLO Detection
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results_yolo = yolo_model.predict(img, conf=0.45, verbose=False)
        
        detected_bbox = None
        for r in results_yolo:
            if len(r.boxes) > 0:
                detected_bbox = r.boxes[0].xyxy[0].tolist()
                break
                
        if not detected_bbox:
            continue # Si no hay mango, saltar

        # SAM Segmentation
        x1, y1, x2, y2 = map(int, detected_bbox)
        results_sam = sam_model.predict(img, bboxes=[detected_bbox], verbose=False)
        
        if not results_sam or results_sam[0].masks is None:
            continue
            
        mask_sam = results_sam[0].masks.data[0].cpu().numpy().astype(np.uint8)
        h, w = img.shape[:2]
        mask_sam = cv2.resize(mask_sam, (w, h))

        # Aislar el mango de la imagen (Fondo negro)
        fruto_aislado = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_sam)
        
        # Aislar las capas
        R = fruto_aislado[:, :, 0].astype(float)
        G = fruto_aislado[:, :, 1].astype(float)
        B = fruto_aislado[:, :, 2].astype(float)

        # NGRDI solo en los píxeles del mango
        ngrdi = np.zeros((h, w), dtype=float)
        divisor = G + R
        valid_pixels = divisor > 0
        ngrdi[valid_pixels] = (G[valid_pixels] - R[valid_pixels]) / (divisor[valid_pixels] + 1e-6)

        # Generar máscara de Necrosis y Normalizar
        mask_necrosis = ((ngrdi < -0.075) & (mask_sam > 0)).astype(np.uint8) * 255
        
        # Heatmap centrado solo en el rango útil
        ngrdi_norm = cv2.normalize(ngrdi, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        ngrdi_invertido = cv2.bitwise_and(255 - ngrdi_norm, 255 - ngrdi_norm, mask=mask_sam)
        heatmap = cv2.applyColorMap(ngrdi_invertido, cv2.COLORMAP_JET)
        heatmap = cv2.bitwise_and(heatmap, heatmap, mask=mask_sam)

        # Añadir BBox YOLO en la original para la visualización
        img_visual = img_rgb.copy()
        cv2.rectangle(img_visual, (x1, y1), (x2, y2), (255, 255, 0), 3)
        cv2.putText(img_visual, "YOLO+SAM", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Configurar Figura (Matplotlib)
        fig = plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.imshow(img_visual)
        plt.title("Etiquetado Original (YOLO)")
        plt.axis("off")

        plt.subplot(2, 3, 2)
        plt.imshow(mask_sam, cmap="gray")
        plt.title("Aislamiento Exacto (SAM)")
        plt.axis("off")

        plt.subplot(2, 3, 3)
        plt.imshow(G, cmap="Greens")
        plt.title("Capa Verde (Tejido Sano)")
        plt.axis("off")

        plt.subplot(2, 3, 4)
        plt.imshow(R, cmap="Reds")
        plt.title("Capa Roja (Afectación)")
        plt.axis("off")

        plt.subplot(2, 3, 5)
        plt.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
        plt.title("Heatmap Espectral NGRDI")
        plt.axis("off")

        plt.subplot(2, 3, 6)
        plt.imshow(mask_necrosis, cmap="hot")
        plt.title("Severidad Estricta (Necrosis)")
        plt.axis("off")

        # Replicar estructura de carpetas
        rel_path = os.path.relpath(os.path.dirname(img_path), source_dir)
        target_subfolder = os.path.join(output_dir, rel_path)
        if not os.path.exists(target_subfolder):
            os.makedirs(target_subfolder)

        out_path = os.path.join(target_subfolder, f"MAPA_{os.path.basename(img_path)}")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

if __name__ == "__main__":
    SOURCE = r"C:\Users\juanv\Desktop\Analisis mango\fruto\fruto"
    TARGET = r"C:\Users\juanv\Desktop\Analisis mango\MAPAS_ESPECTRALES_MASIVOS"
    YOLO_W = r"C:\Users\juanv\Desktop\Analisis mango\runs\detect\Entrenamiento_Cientifico\Mango_YOLOv26_NUEVO_SET3\weights\best.pt"
    
    aplicar_mapa_calor_masivo(SOURCE, TARGET, YOLO_W)
