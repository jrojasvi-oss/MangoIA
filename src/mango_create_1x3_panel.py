import cv2
import numpy as np
import os
import glob

def create_final_1x3_panel(base_path, output_name):
    print("[*] Generando Panel 1x3 de Alta Fidelidad...")
    
    # Categorías principales
    categories = {
        "TESTIGO": os.path.join(base_path, "TESTIGO"),
        "SERENADE": os.path.join(base_path, "SERENADE"),
        "TRICHODERMA": os.path.join(base_path, "TRICHODERMA")
    }
    
    images = []
    
    for label, folder in categories.items():
        # Buscar recursivamente la primera imagen segmentada
        search_pattern = os.path.join(folder, "**", "SEGMENT_*.png")
        found_files = glob.glob(search_pattern, recursive=True)
        
        # También buscar en la raíz de la categoría por si acaso
        found_files += glob.glob(os.path.join(folder, "SEGMENT_*.png"))
        
        if not found_files:
            print(f"[!] No se encontró ninguna imagen para {label}")
            continue
            
        # Tomar la primera imagen encontrada
        img = cv2.imread(found_files[0])
        img = cv2.resize(img, (1000, 1200)) # Tamaño estándar para el panel
        
        # Añadir encabezado técnico
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (1000, 120), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
        
        cv2.putText(img, label, (50, 80), cv2.FONT_HERSHEY_DUPLEX, 2.5, (255, 255, 255), 4)
        images.append(img)

    if len(images) == 3:
        # Combinar en 1x3
        panel = np.hstack(images)
        # Añadir borde blanco entre imágenes
        cv2.imwrite(output_name, panel)
        print(f"[!] PANEL 1x3 GENERADO EXITOSAMENTE: {output_name}")
    else:
        print(f"[!] Error: Solo se recolectaron {len(images)} imágenes de 3.")

if __name__ == "__main__":
    base = r"C:\Users\juanv\Desktop\Analisis mango\FRUTO_SEGMENTACION_FINAL\fruto"
    out = r"C:\Users\juanv\Desktop\Analisis mango\COMPARATIVA_CIENTIFICA_1x3.jpg"
    create_final_1x3_panel(base, out)
