import gradio as gr
import cv2
import numpy as np
import pandas as pd
import os
import sys

# Asegurar que podemos importar desde src
sys.path.append(os.path.join(os.getcwd(), 'src'))
from autonomous_anthracnose_pipeline import AutonomousAnthracnoseModel

# Configuración de Rutas
PESOS_YOLO = r"models/runs/detect/Entrenamiento_Cientifico/Mango_YOLOv26_NUEVO_SET3/weights/best.pt"
# Si no existe en la ruta de arriba, buscamos en la raíz de models por si acaso
if not os.path.exists(PESOS_YOLO):
    PESOS_YOLO = r"models/best.pt"

# Inicializar Modelo (Singleton para la App)
try:
    model = AutonomousAnthracnoseModel(yolo_weights=PESOS_YOLO)
except Exception as e:
    print(f"Error cargando modelo: {e}")
    model = None

def process_single_image(input_img):
    """ Función para procesar una imagen desde la interfaz """
    if model is None:
        return None, "Error: Modelo no cargado. Verifica las rutas en models/"
    
    # Gradio entrega RGB, nuestro modelo usa BGR internamente (vía OpenCV) o RGB según la lógica
    # El modelo espera una ruta o podríamos adaptarlo para recibir el array directamente
    # Para simplificar y no cambiar la lógica original, guardamos temporalmente
    temp_path = "temp_gui_input.jpg"
    cv2.imwrite(temp_path, cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR))
    
    processed_img, fruit_data = model.process_image(temp_path)
    
    if processed_img is None:
        return None, "No se detectaron frutos."
    
    # Preparar reporte de texto
    if fruit_data:
        df = pd.DataFrame(fruit_data)
        avg_sev = df["Severidad_%"].mean()
        incidencia = (df["Incidencia"].sum() / len(df)) * 100
        report = f"### 📊 Resultado del Análisis\n"
        report += f"- **Frutos Detectados:** {len(df)}\n"
        report += f"- **Incidencia:** {incidencia:.2f}%\n"
        report += f"- **Severidad Promedio:** {avg_sev:.2f}%\n"
        return cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), report
    
    return cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), "Detección finalizada sin métricas."

# --- INTERFAZ GRAFICA (Gradio) ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="green", neutral_hue="slate")) as demo:
    gr.Markdown("""
    # 🥭 MangoIA: Diagnóstico Autónomo de Antracnosis
    ### Interfaz Científica de Alta Precisión (RGB-Only)
    Arrastra una imagen o selecciona un archivo para analizar la salud del fruto en segundos.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Imagen Original", type="numpy")
            btn_process = gr.Button("🚀 Iniciar Diagnóstico", variant="primary")
        
        with gr.Column(scale=1):
            output_image = gr.Image(label="Diagnóstico (Malla 64x64)")
            output_text = gr.Markdown(label="Métricas Epidemiológicas")

    with gr.Accordion("Configuración Avanzada", open=False):
        gr.Info("Aquí podrás ajustar umbrales de detección en futuras versiones.")
        conf_slider = gr.Slider(0.1, 1.0, value=0.45, label="Confianza YOLO")

    btn_process.click(
        fn=process_single_image,
        inputs=[input_image],
        outputs=[output_image, output_text]
    )

    gr.Markdown("""
    ---
    **Nota:** El conocimiento es libre, así como el código debe ser accesible, gratuito y comprensible.
    """)

if __name__ == "__main__":
    demo.launch(share=True)
