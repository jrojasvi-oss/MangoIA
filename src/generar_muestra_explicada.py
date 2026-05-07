import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from ultralytics import YOLO, SAM

def clasificar_frutos_al_azar(source_dir, yolo_model, num_muestras=20):
    """
    Busca al azar en el dataset imágenes y usa YOLO preliminarmente para 
    separar fotos con frutos que parezcan 'Sanos' y fotos con frutos 'Enfermos'.
    Devuelve listas con las rutas de las imágenes.
    """
    todas_las_fotos = []
    for root, _, files in os.walk(source_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                todas_las_fotos.append(os.path.join(root, f))
    
    random.shuffle(todas_las_fotos)
    
    sanas = []
    enfermas = []
    
    print("[*] Seleccionando 20 buenas y 20 afectadas al azar...")
    for img_path in todas_las_fotos:
        if len(sanas) >= num_muestras and len(enfermas) >= num_muestras:
            break
            
        img = cv2.imread(img_path)
        if img is None: continue
        
        # Evaluar rápidamente si es sana o enferma usando NGRDI básico sobre el bounding box de YOLO
        results = yolo_model.predict(img, conf=0.45, verbose=False)
        if not results or len(results[0].boxes) == 0:
            continue
            
        bbox = results[0].boxes[0].xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, bbox)
        roi = img[y1:y2, x1:x2]
        
        R = roi[:,:,2].astype(float) # OpenCV es BGR
        G = roi[:,:,1].astype(float)
        ngrdi = (G - R) / (G + R + 1e-6)
        
        severidad = np.mean(ngrdi < -0.075) # Porcentaje de necrosis
        
        if severidad > 0.05 and len(enfermas) < num_muestras:
            enfermas.append(img_path)
        elif severidad <= 0.01 and len(sanas) < num_muestras:
            sanas.append(img_path)
            
    return sanas, enfermas

def generar_paneles_explicados(rutas_imagenes, output_dir, yolo_model, sam_model, etiqueta):
    """
    Procesa las fotos seleccionadas, y agrega explicaciones visuales del porqué (lógica RGB/NGRDI)
    y especifica los nombres de los modelos en la propia imagen.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for i, img_path in enumerate(tqdm(rutas_imagenes, desc=f"Procesando {etiqueta}s")):
        img = cv2.imread(img_path)
        if img is None: continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results_yolo = yolo_model.predict(img, conf=0.45, verbose=False)
        
        if not results_yolo or len(results_yolo[0].boxes) == 0:
            continue
            
        detected_bbox = results_yolo[0].boxes[0].xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, detected_bbox)
        
        results_sam = sam_model.predict(img, bboxes=[detected_bbox], verbose=False)
        if not results_sam or results_sam[0].masks is None:
            continue
            
        mask_sam = results_sam[0].masks.data[0].cpu().numpy().astype(np.uint8)
        h, w = img.shape[:2]
        mask_sam = cv2.resize(mask_sam, (w, h))

        fruto_aislado = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_sam)
        R = fruto_aislado[:, :, 0].astype(float)
        G = fruto_aislado[:, :, 1].astype(float)
        
        ngrdi = np.zeros((h, w), dtype=float)
        valid_pixels = (G + R) > 0
        ngrdi[valid_pixels] = (G[valid_pixels] - R[valid_pixels]) / ((G[valid_pixels] + R[valid_pixels]) + 1e-6)

        mask_necrosis = ((ngrdi < -0.075) & (mask_sam > 0)).astype(np.uint8) * 255
        sev_pct = np.sum(mask_necrosis > 0) / (np.sum(mask_sam > 0) + 1e-6) * 100

        # Crear figura
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(f"Análisis RGB-NGRDI ({etiqueta.upper()}) | Modelos: YOLOv26 + SAM 2.1\nSeveridad: {sev_pct:.2f}%", 
                     fontsize=16, fontweight='bold', color='darkblue')
        
        # Panel 1: Original + Info
        ax1 = plt.subplot(2, 3, 1)
        ax1.imshow(img_rgb)
        ax1.set_title("1. Original\n(Se ve el mango sin procesar)", fontsize=10)
        ax1.axis("off")

        # Panel 2: YOLO + SAM
        ax2 = plt.subplot(2, 3, 2)
        img_bbox = img_rgb.copy()
        cv2.rectangle(img_bbox, (x1, y1), (x2, y2), (255, 255, 0), 3)
        img_bbox[mask_sam == 0] = img_bbox[mask_sam == 0] // 3 # Oscurecer fondo
        ax2.imshow(img_bbox)
        ax2.set_title("2. Localizado (YOLOv26)\n& Aislado (SAM 2.1)", fontsize=10)
        ax2.axis("off")

        # Panel 3: Lógica RGB (Explicación textual de por qué se usa)
        ax3 = plt.subplot(2, 3, 3)
        ax3.imshow(G, cmap="Greens")
        texto_rgb = "LÓGICA DEL ESPECTRO RGB:\n\n- La clorofila (SANO) refleja mucha luz Verde (G).\n- La necrosis absorbe Verde y refleja más Rojo (R).\n\nPor tanto, aislando el Rojo y el Verde podemos\nmatemáticamente saber qué es hongo y qué es tejido."
        ax3.text(0.5, 0.5, texto_rgb, transform=ax3.transAxes, ha="center", va="center", 
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'), fontsize=9)
        ax3.set_title("3. Espectro Verde Fuerte = Sano", fontsize=10)
        ax3.axis("off")

        # Panel 4: Capa NGRDI Explicada
        ax4 = plt.subplot(2, 3, 4)
        ax4.imshow(R, cmap="Reds")
        ax4.set_title("4. Espectro Rojo (Se marca la antracnosis)", fontsize=10)
        ax4.axis("off")

        # Panel 5: Mapa de Calor
        ax5 = plt.subplot(2, 3, 5)
        ngrdi_norm = cv2.normalize(ngrdi, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        ngrdi_invertido = cv2.bitwise_and(255 - ngrdi_norm, 255 - ngrdi_norm, mask=mask_sam)
        heatmap = cv2.applyColorMap(ngrdi_invertido, cv2.COLORMAP_JET)
        heatmap = cv2.bitwise_and(heatmap, heatmap, mask=mask_sam)
        ax5.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
        ax5.set_title("5. Heatmap NGRDI (Ecuación: (G-R)/(G+R))\nZonas cálidas = Necrosis crítica", fontsize=10)
        ax5.axis("off")

        # Panel 6: Segmentación Estricta
        ax6 = plt.subplot(2, 3, 6)
        ax6.imshow(mask_necrosis, cmap="hot")
        texto_sev = f"Severidad:\n{sev_pct:.2f}%"
        ax6.text(10, 30, texto_sev, color="white", fontsize=12, fontweight='bold')
        ax6.set_title("6. Segmentación Matemática del Hongo", fontsize=10)
        ax6.axis("off")

        out_path = os.path.join(output_dir, f"{etiqueta}_{i+1}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

if __name__ == "__main__":
    SOURCE = r"C:\Users\juanv\Desktop\Analisis mango\fruto\fruto"
    TARGET = r"C:\Users\juanv\Desktop\Analisis mango\MUESTRA_EXPLICATIVA_20"
    YOLO_W = r"C:\Users\juanv\Desktop\Analisis mango\runs\detect\Entrenamiento_Cientifico\Mango_YOLOv26_NUEVO_SET3\weights\best.pt"
    
    print("[*] Iniciando Sistema de Muestreo Inteligente...")
    yolo = YOLO(YOLO_W)
    sam = SAM('sam_b.pt')
    
    sanas, enfermas = clasificar_frutos_al_azar(SOURCE, yolo, num_muestras=20)
    
    print(f"[*] Generando paneles explicados para {len(sanas)} sanas y {len(enfermas)} enfermas...")
    generar_paneles_explicados(sanas, TARGET, yolo, sam, "Sana")
    generar_paneles_explicados(enfermas, TARGET, yolo, sam, "Enferma")
    
    print(f"\n[+] Proceso finalizado. Puedes ver las infografías en: {TARGET}")
