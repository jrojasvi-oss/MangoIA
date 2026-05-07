import os
from ultralytics import YOLO
from pathlib import Path

# 1. Configuración de Rutas (Proyecto Mango - Lógica Antracnosis)
mango_model_weights = r"C:\Users\juanv\Desktop\Analisis mango\runs\detect\Entrenamiento_Cientifico\Mango_YOLOv26_NUEVO_SET3\weights\best.pt"
base_dir = Path(r"C:\Users\juanv\Desktop\Analisis mango")
output_folder = base_dir / "RESULTADOS_MANGO_VAL_FINAL"
output_folder.mkdir(parents=True, exist_ok=True)

# 2. Cargar el modelo de Mango (Antracnosis)
print(f"[*] Usando Lógica de Mango (Antracnosis): {mango_model_weights}")
model = YOLO(mango_model_weights)

# 3. Procesar todas las categorías de Mango
categories = ["TESTIGO", "SERENADE", "TRICHODERMA", "PROTERRA"]
source_base = base_dir / "fruto" / "fruto"

print(f"[*] Iniciando procesamiento masivo de categorías de Mango...")

for cat in categories:
    cat_path = source_base / cat
    if not cat_path.exists():
        continue
        
    print(f"\n[+] Procesando Categoría: {cat}")
    
    # Procesar imágenes de la categoría
    results = model.predict(
        source=str(cat_path),
        save=True,
        project=str(output_folder),
        name=cat,
        exist_ok=True,
        conf=0.25
    )

print(f"\n[!] PROCESAMIENTO DE MANGO COMPLETADO.")
print(f"Ruta de resultados: {output_folder}")
