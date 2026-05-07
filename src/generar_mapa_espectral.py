import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def generar_mapa_espectral(imagen_path, output_dir):
    """
    Lee una imagen, aísla las capas RGB y calcula el índice NGRDI 
    para generar un mapa de calor (heatmap) que resalta visualmente 
    las áreas afectadas por la enfermedad (antracnosis).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img = cv2.imread(imagen_path)
    if img is None:
        print(f"Error: No se pudo leer la imagen {imagen_path}")
        return

    # Convertir a RGB para matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Aislar las capas
    R = img_rgb[:, :, 0].astype(float)
    G = img_rgb[:, :, 1].astype(float)
    B = img_rgb[:, :, 2].astype(float)

    # Calcular NGRDI (Normalized Green Red Difference Index)
    # NGRDI < -0.075 es típicamente tejido necrótico (enfermedad)
    ngrdi = (G - R) / (G + R + 1e-6)
    
    # Normalizar NGRDI entre 0 y 255 para visualización de mapa de calor
    ngrdi_norm = cv2.normalize(ngrdi, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Aplicar un mapa de calor falso (JET: Rojo indica valores altos/diferentes, Azul valores bajos)
    # Para necrosis (valores bajos de NGRDI), invertiremos el NGRDI para que la necrosis brille en rojo
    ngrdi_invertido = 255 - ngrdi_norm
    heatmap = cv2.applyColorMap(ngrdi_invertido, cv2.COLORMAP_JET)

    # Crear una figura con subplots para visualizar las capas de la enfermedad
    plt.figure(figsize=(15, 10))

    # Imagen original
    plt.subplot(2, 3, 1)
    plt.imshow(img_rgb)
    plt.title("Imagen Original")
    plt.axis("off")

    # Capa Roja (Suele absorber más en tejido sano y reflejar en tejido muerto)
    plt.subplot(2, 3, 2)
    plt.imshow(R, cmap="Reds")
    plt.title("Capa Roja (R)")
    plt.axis("off")

    # Capa Verde (Asociada a clorofila y tejido sano)
    plt.subplot(2, 3, 3)
    plt.imshow(G, cmap="Greens")
    plt.title("Capa Verde (G)")
    plt.axis("off")

    # Capa NGRDI Cruda (Blanco y negro)
    plt.subplot(2, 3, 4)
    plt.imshow(ngrdi, cmap="gray")
    plt.title("Índice NGRDI (Blanco = Sano, Negro = Necrosis)")
    plt.axis("off")

    # Heatmap de Severidad
    plt.subplot(2, 3, 5)
    plt.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
    plt.title("Mapa de Calor de la Enfermedad")
    plt.axis("off")
    
    # Máscara estricta de enfermedad
    mask_necrosis = (ngrdi < -0.075).astype(np.uint8) * 255
    plt.subplot(2, 3, 6)
    plt.imshow(mask_necrosis, cmap="hot")
    plt.title("Área Estricta de Infección")
    plt.axis("off")

    # Guardar visualización
    nombre_base = os.path.basename(imagen_path)
    output_path = os.path.join(output_dir, f"ESPECTRAL_{nombre_base}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"[+] Mapa espectral generado en: {output_path}")

if __name__ == "__main__":
    # Puedes cambiar esta ruta para probar con cualquier foto de tu dataset
    IMAGEN_PRUEBA = r"C:\Users\juanv\Desktop\Analisis mango\fruto\fruto\PROTERRA\CON PODA Y SIN PODA\Con poda\Jul\r1.png"
    CARPETA_SALIDA = r"C:\Users\juanv\Desktop\Analisis mango\VISUALIZACION_ESPECTRAL"
    
    generar_mapa_espectral(IMAGEN_PRUEBA, CARPETA_SALIDA)
