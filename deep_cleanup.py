import os
import shutil
import glob

def deep_cleanup():
    # 1. Definir Estructura Maestra
    master_structure = {
        'src': [], # Para scripts de python
        'scripts_r': [], # Para scripts de R
        'docs': [], # Documentación
        'data': [], # Datasets, imágenes crudas, archivos zip
        'results': [], # Salidas, reportes, imágenes procesadas
        'models': [], # Pesos .pt, carpetas de entrenamiento 'runs'
        'media': []  # Imágenes/Videos para visualización
    }
    
    for folder in master_structure.keys():
        os.makedirs(folder, exist_ok=True)

    # 2. Clasificación de remanentes en la raíz
    to_move = {
        'data': [
            'Train', 'YOLO_Antracnosis_Dataset', 'fruto', 'fruto.zip', 
            'datos_severidad_exacta.csv', 'data_samples'
        ],
        'results': [
            'Carpeta_Especial_Mascaras_Escala', 'IMAGENES_SUELTAS_PROCESADAS', 
            'RESULTADOS_MANGO_VAL_FINAL', 'temp_docx', 'REPORTE_ESTADISTICO_MANGO_ANTRACO.xlsx'
        ],
        'models': [
            'runs'
        ],
        'src': [
            'final_cleanup.py', 'organize_for_github.py'
        ]
    }

    for target_folder, items in to_move.items():
        for item in items:
            if os.path.exists(item):
                try:
                    # Si el destino ya existe (como data_samples -> data), movemos el contenido o la carpeta
                    dest = os.path.join(target_folder, item)
                    if os.path.exists(dest):
                        # Si es una carpeta, intentamos fusionar o renombrar
                        dest = os.path.join(target_folder, item + "_root_backup")
                    
                    shutil.move(item, dest)
                    print(f"[OK] {item} -> {target_folder}/")
                except Exception as e:
                    print(f"[ERR] No se pudo mover {item}: {e}")

    # 3. Limpieza de archivos temporales / basura
    if os.path.exists('__pycache__'):
        shutil.rmtree('__pycache__', ignore_errors=True)
        print("[OK] Eliminado __pycache__")

    print("\n[!] LIMPIEZA PROFUNDA FINALIZADA.")
    print("La raíz ahora solo contiene las carpetas maestras y los archivos de configuración.")

if __name__ == "__main__":
    deep_cleanup()
