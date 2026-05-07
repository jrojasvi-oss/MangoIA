from ultralytics import YOLO
import os

print("""
============================================================
              YOLOv8 - ENTRENAMIENTO (50 EPOCHS)            
============================================================
Iniciando carga de hiperparámetros y tensores...
""")

ruta_yaml = r"C:\Users\juanv\Desktop\Analisis mango\YOLO_Antracnosis_Dataset\dataset.yaml"

if not os.path.exists(ruta_yaml):
    print("Error: No encuentro el dataset.yaml Asegúrate de correr crear_dataset_yolo.py primero.")
else:
    # 1. Instanciar YOLO base
    model = YOLO("yolov8n.pt")  # Modelo Base (Nano para que corras sin sobrecargar la memoria)
    
    # 2. Iniciar el Entrenamiento de 50 Épocas
    print("[*] Iniciando Backpropagation en 50 épocas...")
    resultados = model.train(
        data=ruta_yaml,
        epochs=50,          # Exigencia directa: 50 epochs
        imgsz=640,
        batch=4,            # Batch bajo para computadores portátiles / de escritorio normales
        device='cpu',       # Si tienes tarjeta gráfica NVidia en el PC, cámbialo a device=0
        name="Antracnosis_Investigacion_Modelo",
        project=r"C:\Users\juanv\Desktop\Analisis mango\YOLO_Resultados"
    )

    print("\n[✔] Entrenamiento completado exitosamente.")
    print("Los pesos neuronales de tu inteligencia artificial se han guardado en YOLO_Resultados")

    # 3. Predicción con Tu Modelo Recién Creado:
    # Tomamos 1 foto cualquiera para medir
    foto_prueba = r"C:\Users\juanv\Desktop\Analisis mango\YOLO_Antracnosis_Dataset\images\train\rr3.png" # Cambia esto si no existe
    
    if os.path.exists(foto_prueba):
        pred_resultados = model.predict(foto_prueba, save=True)
        # Calcular áreas
        for r in pred_resultados:
            # R.boxes tiene las manchas detectadas
            area_lesiones = 0
            for box in r.boxes:
                # box.xywh <- CentroX, CentroY, Ancho, Alto
                w, h = box.xywh[0][2], box.xywh[0][3]
                area_lesiones += (w * h).item()
            
            print(f"El Agente YOLO detectó {len(r.boxes)} lesiones de Antracnosis en esta prueba.")
            print(f"Área absoluta infectada (Píxeles al cuadrado): {int(area_lesiones)} px²")
