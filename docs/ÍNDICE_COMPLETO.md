# 🗂️ Índice Completo del Sistema de Análisis de Mango

Este documento sirve como mapa para navegar por todos los componentes del pipeline de Antracnosis.

## 🚀 Núcleo del Procesamiento (Python)
1.  [`antracnosis_spectral_analyzer.py`](file:///c:/Users/juanv/Desktop/Analisis%20mango/antracnosis_spectral_analyzer.py)
    *   **Función:** Clase maestra que integra YOLO, SAM e índices espectrales.
2.  [`mango_disease_pipeline.py`](file:///c:/Users/juanv/Desktop/Analisis%20mango/mango_disease_pipeline.py)
    *   **Función:** Orquestador masivo. Procesa carpetas enteras y genera CSVs.
3.  [`entrenamiento_autonomo.py`](file:///c:/Users/juanv/Desktop/Analisis%20mango/entrenamiento_autonomo.py)
    *   **Función:** Script para re-entrenar el modelo YOLO con 2500 épocas.

## 📝 Documentación y Guías
4.  [`README.md`](file:///c:/Users/juanv/Desktop/Analisis%20mango/README.md)
    *   Visión general del proyecto y estado actual.
5.  [`INSTALACION.md`](file:///c:/Users/juanv/Desktop/Analisis%20mango/INSTALACION.md)
    *   Pasos para configurar el entorno virtual y dependencias.
6.  [`TARJETA_REFERENCIA.md`](file:///c:/Users/juanv/Desktop/Analisis%20mango/TARJETA_REFERENCIA.md)
    *   Guía rápida de comandos y parámetros clave.

## 🧪 Investigación y Estrategia
7.  [`PROMPT_INVESTIGACION_AMPLÍA.md`](file:///c:/Users/juanv/Desktop/Analisis%20mango/PROMPT_INVESTIGACION_AMPL%C3%8DA.md)
    *   Prompt para Claude Sonnet sobre ciencia de vanguardia.
8.  [`REFERENCIA_RÁPIDA.md`](file:///c:/Users/juanv/Desktop/Analisis%20mango/REFERENCIA_RÁPIDA.md)
    *   Tablas de valores RGB/HSV y teoría de índices espectrales.

## 📊 Salidas y Resultados
*   **Carpeta `RUNS_SPECTRAL_FINAL`** (Se crea al ejecutar el pipeline)
    *   `DIAGNOSTICO_*.png`: Imágenes con máscaras de severidad.
    *   `CONSOLIDADO_ANT_SPECTRAL.csv`: Base de datos de todos los frutos.
    *   `RESUMEN_ESTADISTICO_CEPAS.csv`: Promedios por tratamiento.
