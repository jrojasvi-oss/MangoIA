from ultralytics import YOLO
import os

def run_autonomous_training():
    print("[*] Iniciando Entrenamiento Autónomo de Refinamiento Científico...")
    
    # Cargar el modelo actual como base para fine-tuning
    weights_path = r"C:\Users\juanv\Desktop\Analisis mango\runs\detect\Entrenamiento_Cientifico\Mango_YOLOv26_NUEVO_SET3\weights\best.pt"
    model = YOLO(weights_path)
    
    # Configuración de datos
    data_yaml = r"C:\Users\juanv\Desktop\Analisis mango\YOLO_Antracnosis_Dataset\dataset.yaml"
    
    # Entrenamiento autónomo de alta intensidad
    # 50 épocas en 50 ciclos = 2500 épocas (con paciencia para optimizar)
    results = model.train(
        data=data_yaml,
        epochs=2500,
        patience=50,
        imgsz=640,
        batch=16,
        name='Refinamiento_Cientifico_Final',
        project='runs/detect',
        exist_ok=True,
        # Aumentos para realismo
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1
    )
    
    print("[!] ENTRENAMIENTO FINALIZADO. Nuevo modelo en runs/detect/Refinamiento_Cientifico_Final")

if __name__ == "__main__":
    run_autonomous_training()
