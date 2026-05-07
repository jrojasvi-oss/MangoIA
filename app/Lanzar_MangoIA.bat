@echo off
title MangoIA - Diagnostico Autonomo (App Mode)
echo [*] Iniciando Interfaz Grafica desde carpeta /app...
cd /d "%~dp0"
python app_gui.py
pause
