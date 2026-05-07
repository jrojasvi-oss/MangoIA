# 🥭 Mango Anthracnose AI Diagnosis Pipeline

Este repositorio contiene un ecosistema avanzado de Inteligencia Artificial diseñado para la detección, segmentación y análisis epidemiológico de la antracnosis en frutos de mango utilizando imágenes RGB.

## 🚀 Descripción del Proyecto
El proyecto integra modelos de **Deep Learning (YOLO v26 + SAM)** con análisis de **Índices Espectrales (NGRDI, RGBVI, GLI)** para proporcionar una herramienta autónoma de diagnóstico para investigadores y productores (APLIMAN).

## 📂 Estructura del Repositorio

| Carpeta / Archivo | Descripción |
| --- | --- |
| `src/` | Núcleo del software. Contiene el pipeline autónomo y analizadores espectrales. |
| `scripts_r/` | Scripts en R para análisis estadístico avanzado y uso de `pliman`. |
| `models/` | (No incluido en GitHub por peso) Almacena `best.pt` (YOLO) y `sam_b.pt`. |
| `docs/` | Manuales de uso, metodologías científicas y guías de prompts. |
| `results/` | Directorio por defecto para reportes, CSVs e imágenes diagnosticadas. |

## 🛠️ Tecnologías Utilizadas
*   **Computer Vision**: YOLO (Ultralytics), SAM (Segment Anything Model).
*   **Análisis Espectral**: Índices de vegetación calculados sobre espacio de color RGB.
*   **Lenguajes**: Python 3.9+, R 4.x.
*   **Librerías Clave**: OpenCV, Pandas, Matplotlib, Seaborn, Pliman (R).

## 📋 Requisitos e Instalación

1.  **Clonar el repositorio**:
    ```bash
    git clone https://github.com/tu-usuario/mango-anthracnose-ai.git
    cd mango-anthracnose-ai
    ```
2.  **Instalar dependencias de Python**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Descargar Pesos de Modelos**: Asegúrate de colocar `best.pt` y `sam_b.pt` en la carpeta raíz o configurar la ruta en los scripts.

## 🏃 Cómo ejecutar el Pipeline

El punto de entrada principal es el **Modelo Autónomo**:

```bash
python autonomous_anthracnose_pipeline.py
```

Este script procesará las imágenes en la carpeta configurada, generará máscaras de severidad y exportará un consolidado estadístico (`REPORTE_EPIDEMIOLOGICO_APLIMAN.csv`).

### 🖥️ Interfaz Gráfica (GUI)
Si prefieres una experiencia visual con **arrastrar y soltar (Drag & Drop)**:
```bash
python app_gui.py
```
Esto abrirá una interfaz en tu navegador donde puedes cargar fotos y ver los resultados instantáneamente.

## 🧪 Metodología Científica
El pipeline sigue un flujo de 4 etapas:
1.  **Detección**: Localización del fruto.
2.  **Segmentación**: Aislamiento del fruto del fondo.
3.  **Cuantificación**: Cálculo de necrosis mediante malla 64x64 y umbrales NGRDI/RGBVI.
4.  **Reporte**: Generación de métricas de incidencia y severidad.

---
**Desarrollado para la convocatoria HERMES - Proyecto Mango.**  
Colaboradores: Juan V. y la comunidad de código abierto.

> [!IMPORTANT]
> El conocimiento es libre, así como el código debe ser accesible, gratuito y comprensible.
