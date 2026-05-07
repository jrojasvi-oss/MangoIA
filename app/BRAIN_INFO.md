# 🧠 El Cerebro de MangoIA (Modelos)

Este documento explica cómo gestionar los pesos de los modelos para que el repositorio no sea pesado en GitHub pero funcione a la perfección.

## 📁 Ubicación de los Pesos
El "Cerebro" consta de dos archivos principales que deben estar en la raíz o en una carpeta `models/`:
1.  **`best.pt`**: El modelo YOLO entrenado para detectar mangos.
2.  **`sam_b.pt`**: El modelo Segment Anything para la segmentación ultra-precisa.

## ☁️ Estrategia para GitHub (Sin peso excesivo)
Para subir el proyecto a GitHub sin que el archivo `sam_b.pt` (375MB) bloquee la subida:

### Opción A: Git LFS (Recomendado)
Si tienes instalado **Git LFS**, puedes subir los archivos grandes así:
```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes
git add models/best.pt models/sam_b.pt
git commit -m "Agregado el Cerebro (Pesos de IA) vía LFS"
git push origin main
```

### Opción B: Enlace Externo (Ligero)
Si prefieres que el repo sea liviano, puedes subir solo el código y poner un enlace de descarga en el `README` para los pesos.

---
> [!TIP]
> Mantener los modelos separados del código es una buena práctica de ingeniería de IA para facilitar el control de versiones.
