import gradio as gr
import cv2
import numpy as np
import pandas as pd
import os
import sys

# Agregar el directorio actual al path para importar el motor
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.autonomous_anthracnose_pipeline import AutonomousAnthracnoseModel

# Configuración de Rutas para Hugging Face (Donde los archivos están en la raíz o /models)
PESOS_YOLO = "models/best.pt"
PESOS_SAM = "models/sam_b.pt"

# Intentar cargar el modelo
try:
    model = AutonomousAnthracnoseModel(yolo_weights=PESOS_YOLO, sam_weights=PESOS_SAM)
except Exception as e:
    print(f"Esperando a que los modelos se carguen en HF: {e}")
    model = None

def process_single_image(input_img):
    if model is None:
        return None, "Error: Los modelos no se han cargado correctamente en el Space."
    
    temp_path = "temp_hf_input.jpg"
    cv2.imwrite(temp_path, cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR))
    
    processed_img, fruit_data = model.process_image(temp_path)
    
    if processed_img is None:
        return None, "No se detectaron frutos en la imagen."
    
    if fruit_data:
        df = pd.DataFrame(fruit_data)
        avg_sev = df["Severidad_%"].mean()
        incidencia = (df["Incidencia"].sum() / len(df)) * 100
        report = f"### 📊 Reporte Epidemiológico (MangoIA)\n"
        report += f"- **Mangos Analizados:** {len(df)}\n"
        report += f"- **Incidencia Lote:** {incidencia:.2f}%\n"
        report += f"- **Severidad Promedio:** {avg_sev:.2f}%\n"
        return cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), report
    
    return cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), "Análisis completado."

# --- INTERFAZ GRADIO ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="green", neutral_hue="slate")) as demo:
    gr.Markdown("""
    # 🥭 MangoIA: Diagnóstico de Antracnosis
    ### Fitopatología de Precisión con IA y Análisis Espectral (RGB-Only)
    Sube una foto de tus mangos para obtener un diagnóstico instantáneo de severidad e incidencia.
    """)
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Subir Foto de Mango", type="numpy")
            btn = gr.Button("🔍 Iniciar Análisis", variant="primary")
        
        with gr.Column():
            output_image = gr.Image(label="Resultado (Malla 64x64)")
            output_text = gr.Markdown()

    gr.Markdown("""
    ---
    **Proyecto MangoIA** | Conocimiento Libre | Desarrollado por Juan V.
    """)

if __name__ == "__main__":
    demo.launch()
