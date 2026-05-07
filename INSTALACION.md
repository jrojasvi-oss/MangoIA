# ⚙️ Guía de Instalación y Configuración

Sigue estos pasos para preparar tu entorno de trabajo.

## 1. Requisitos Previos
*   **Python 3.10+** (Recomendado 3.11).
*   **Git** instalado.
*   **NVIDIA GPU** con drivers actualizados (Opcional, pero recomendado para SAM).

## 2. Crear Entorno Virtual (Recomendado)
```bash
python -m venv venv
# Activar en Windows:
.\venv\Scripts\activate
```

## 3. Instalar Dependencias
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Descargar Pesos de SAM
El pipeline descargará automáticamente `sam_b.pt` en la primera ejecución, pero asegúrate de tener conexión a internet.

## 5. Verificación
Ejecuta el siguiente comando para probar la instalación:
```bash
python -c "import ultralytics; ultralytics.checks()"
```

Si todo está en verde, ¡estás listo para procesar tus imágenes!
