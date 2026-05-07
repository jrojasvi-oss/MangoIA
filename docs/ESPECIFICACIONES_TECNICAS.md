# 🧩 Especificaciones del Modelo e Investigación

Este documento detalla la lógica técnica y los parámetros de los modelos de IA integrados en el pipeline de MangoIA.

## 1. Detección (YOLO v26)
*   **Arquitectura:** Ultralytics YOLOv8/v9 (configurado como v26 para el proyecto).
*   **Pesos:** `models/best.pt`.
*   **Clases:** `0: Mango`.
*   **Confianza Recomendada:** `0.45`.
*   **Uso:** Localización espacial del fruto para aislarlo de hojas y ramas.

## 2. Segmentación (SAM 2.1)
*   **Modelo:** Segment Anything Model (SAM) - `sam_b.pt`.
*   **Lógica:** Segmentación por instancia disparada por el BBox de YOLO.
*   **Propósito:** Generar una máscara binaria de alta precisión que defina el área real del fruto (píxeles de interés).

## 3. Análisis Espectral (RGB-Only)
El sistema utiliza el espacio de color RGB para derivar índices de vegetación sin necesidad de sensores multiespectrales:

### NGRDI (Normalized Green Red Difference Index)
`NGRDI = (Green - Red) / (Green + Red)`
*   **Rango:** -1 a 1.
*   **Umbral Necrosis:** `-0.075`. Valores inferiores indican tejido muerto o severamente estresado.

### RGBVI (RGB Vegetation Index)
`RGBVI = (Green^2 - Red * Blue) / (Green^2 + Red * Blue)`
*   **Uso:** Mejor discriminación de salud foliar/fruto en condiciones de iluminación variable.

## 4. Cuantificación de Severidad
*   **Técnica:** Rejilla espacial de 64x64 celdas.
*   **Cálculo:** `(Área Necrosada / Área Total del Fruto) * 100`.
*   **Incidencia:** Un fruto se considera enfermo si su severidad supera el `2.0%`.

---
*Este proyecto busca democratizar el acceso a la fitopatología de precisión.*
