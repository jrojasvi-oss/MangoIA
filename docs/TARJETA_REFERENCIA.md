# ⚡ Tarjeta de Referencia Rápida (Cheat Sheet)

## 🛠️ Comandos de Ejecución

| Tarea | Comando |
| :--- | :--- |
| **Ejecutar Pipeline Completo** | `python mango_disease_pipeline.py` |
| **Re-entrenar YOLOv26** | `python entrenamiento_autonomo.py` |
| **Instalar Dependencias** | `pip install -r requirements.txt` |

## 📏 Parámetros del Analizador
*(Ubicados en `antracnosis_spectral_analyzer.py`)*

*   **NGRDI Threshold (-0.075):** Valores por debajo de esto se consideran necrosis (tejido muerto/oscuro).
*   **GLI Threshold (0.1):** Valores por encima de esto indican tejido vegetal sano y vigoroso.
*   **Confianza YOLO (0.45):** Filtro para detectar solo frutos con alta probabilidad.

## 📁 Estructura de Salida
*   `RUNS_SPECTRAL_FINAL/`: Contiene todas las imágenes procesadas.
*   `CONSOLIDADO_ANT_SPECTRAL.csv`: La columna `Severidad_%` es la métrica principal para tu tesis/reporte.

## 🧪 Clasificación de Severidad
*   **Leve:** < 5% del área afectada.
*   **Moderada:** 5% - 15% del área afectada.
*   **Severa:** > 15% del área afectada.
