# 🌈 Referencia Rápida: Espectros y Valores

Este documento detalla la base científica de los índices utilizados en el pipeline.

## 🟢 1. Espacios de Color Clave

| Canal | Descripción en Mango | Uso en el Pipeline |
| :--- | :--- | :--- |
| **G (Verde)** | Clorofila, tejido sano. | Base para NGRDI y GLI. |
| **R (Rojo)** | Absorción por necrosis/antracnosis. | Contraste para detectar daño. |
| **H (Hue)** | El "tono". Café/Negro = Necrosis. | Refinamiento de máscaras. |

## 📊 2. Índices Espectrales Implementados

### **NGRDI (Normalized Green Red Difference Index)**
*   **Fórmula:** `(G - R) / (G + R)`
*   **Uso:** Es el estándar para detectar estrés en plantas. 
*   **Interpretación:**
    *   `> 0.1`: Tejido muy sano.
    *   `-0.05 a 0.05`: Estrés leve o transición.
    *   ` < -0.075`: **Necrosis confirmada (Antracnosis).**

### **GLI (Green Leaf Index)**
*   **Fórmula:** `(2*G - R - B) / (2*G + R + B)`
*   **Uso:** Excelente para separar el fruto verde del fondo ruidoso.

## 🔬 3. Escala de Severidad (Visual)

| % Área | Grado | Acción Sugerida |
| :--- | :--- | :--- |
| **0-2%** | Sano | Monitoreo regular. |
| **2-10%** | Inicial | Aplicación preventiva de SERENADE/PROTERRA. |
| **>15%** | Crítico | Cuarentena del lote o tratamiento de choque. |
