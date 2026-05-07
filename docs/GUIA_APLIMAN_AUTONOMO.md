# 🚀 Modelo Autónomo de Diagnóstico: APLIMAN v3.0

Este modelo consolidado está diseñado como una solución autónoma que mide incidencia y severidad sin intervención manual, facilitando la investigación abierta.

## 🛠️ Componentes del Pipeline

1.  **Detección Localizada (YOLO v26)**: Localiza cada fruto individualmente en la imagen RGB.
2.  **Segmentación de Ultra-Precisión (SAM)**: Crea una máscara perfecta del fruto, eliminando el ruido del fondo.
3.  **Análisis Multiespectral RGB-Only**:
    *   **NGRDI**: Para detección base de necrosis.
    *   **RGBVI**: Implementado según la fórmula: `(G² - R*B) / (G² + R*B)`.
4.  **Malla Científica 64x64**: Divide cada fruto en una rejilla para un análisis espacial detallado.
5.  **Métricas Epidemiológicas**:
    *   **Incidencia**: Porcentaje de frutos con lesiones significativas (>2%).
    *   **Severidad**: Porcentaje de área afectada por fruto.

## 📁 Estructura de Salida (`results/RESULTADOS_AUTONOMOS_APLIMAN`)

*   **`DIAGNOSTICO_*.png`**: Imágenes procesadas con la rejilla de severidad (Rojo: Afectado, Verde: Sano).
*   **`REPORTE_EPIDEMIOLOGICO_APLIMAN.csv`**: Base de datos completa con métricas por fruto e imagen.
*   **`RESUMEN_GRAFICO_APLIMAN.png`**: Visualización estadística de la salud del lote analizado.

## 📝 Cómo ejecutar el modelo

El script `src/autonomous_anthracnose_pipeline.py` es el motor principal. Solo necesitas configurar las rutas al final del archivo.

```python
if __name__ == "__main__":
    PESOS_YOLO = "models/best.pt"
    INPUT_DIR = "data/fotos_originales"
    
    pipeline = AutonomousAnthracnoseModel(yolo_weights=PESOS_YOLO)
    pipeline.run_batch(INPUT_DIR)
```

---
> [!IMPORTANT]
> **El conocimiento es libre, así como el código debe ser accesible, gratuito y comprensible.**
