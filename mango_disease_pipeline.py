import os
import glob
import pandas as pd
from tqdm import tqdm
from antracnosis_spectral_analyzer import AnthracnoseSpectralAnalyzer

def run_mass_pipeline(input_root, output_root, model_path):
    """
    Orquestador Maestro para el procesamiento masivo de imágenes de mango.
    Recorre todas las carpetas (PROTERRA, SERENADE, etc.) y genera reportes.
    """
    # 1. Inicializar el Analizador
    analyzer = AnthracnoseSpectralAnalyzer(model_path)
    
    # 2. Configurar estructura de salida
    if not os.path.exists(output_root):
        os.makedirs(output_root)
        
    all_data = []
    
    # 3. Buscar todas las imágenes (.jpg, .png) recursivamente
    image_extensions = ('/**/*.jpg', '/**/*.png', '/**/*.jpeg')
    images = []
    for ext in image_extensions:
        images.extend(glob.glob(input_root + ext, recursive=True))
        
    print(f"[*] Se encontraron {len(images)} imágenes para procesar.")
    
    # 4. Loop de Procesamiento
    for img_path in tqdm(images, desc="Procesando Dataset"):
        # Extraer metadatos del path (Cepa, Tratamiento, etc)
        path_parts = os.path.normpath(img_path).split(os.sep)
        # Asumiendo estructura: Root/Tratamiento/Imagen.jpg
        # O Root/Cepa/Tratamiento/Imagen.jpg
        cepa = path_parts[-2] if len(path_parts) > 1 else "Unknown"
        filename = os.path.basename(img_path)
        
        # Ejecutar análisis
        results = analyzer.generate_full_report(img_path, output_root)
        
        if results:
            for res in results:
                res.update({
                    "Archivo": filename,
                    "Cepa/Tratamiento": cepa,
                    "Ruta_Original": img_path
                })
                all_data.append(res)
                
    # 5. Consolidar y Exportar Resultados
    if all_data:
        df = pd.DataFrame(all_data)
        csv_path = os.path.join(output_root, "CONSOLIDADO_ANT_SPECTRAL.csv")
        df.to_csv(csv_path, index=False)
        
        # Generar resumen por Cepa
        summary = df.groupby("Cepa/Tratamiento")["Severidad_%"].agg(['mean', 'max', 'count']).reset_index()
        summary.to_csv(os.path.join(output_root, "RESUMEN_ESTADISTICO_CEPAS.csv"), index=False)
        
        print(f"\n[+] Proceso completado exitosamente.")
        print(f"[+] Reporte consolidado en: {csv_path}")
        print(f"[+] Resumen estadístico generado.")
    else:
        print("[!] No se generaron datos. Verifique las detecciones YOLO.")

if __name__ == "__main__":
    # CONFIGURACIÓN DE RUTAS
    INPUT = r"c:\Users\juanv\Desktop\Analisis mango\RESULTADOS_MANGO_VAL_FINAL"
    OUTPUT = r"c:\Users\juanv\Desktop\Analisis mango\RUNS_SPECTRAL_FINAL"
    MODEL = r"c:\Users\juanv\Desktop\Analisis mango\runs\detect\Entrenamiento_Cientifico\Mango_YOLOv26_NUEVO_SET3\weights\best.pt"
    
    run_mass_pipeline(INPUT, OUTPUT, MODEL)
