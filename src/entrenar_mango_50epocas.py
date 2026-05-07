from ultralytics import YOLO
import os

def iniciar_entrenamiento_actualizado():
    print("[*] Iniciando entrenamiento con el NUEVO DATASET: Mango-Antracnosis.yolo26")
    # Cargamos el modelo
    model = YOLO('yolov8n.pt') 

    # NUEVA RUTA DEL DATASET
    data_path = r"C:\Users\juanv\Desktop\Analisis mango\Train\Mango-Antracnosis.yolo26\data.yaml"
    
    if not os.path.exists(data_path):
        print(f"Error crítico: No se encuentra {data_path}")
        return

    # Entrenamiento por 50 épocas con el nuevo set
    model.train(
        data=data_path,
        epochs=50,
        imgsz=640,
        batch=8,
        name='Mango_YOLOv26_NUEVO_SET',
        project='Entrenamiento_Cientifico',
        device='cpu' # Cambiar a 0 si hay GPU
    )
    
    print("\n[!] Entrenamiento con el nuevo set completado.")
    print("Resultados en: Entrenamiento_Cientifico/Mango_YOLOv26_NUEVO_SET")

if __name__ == "__main__":
    iniciar_entrenamiento_actualizado()
