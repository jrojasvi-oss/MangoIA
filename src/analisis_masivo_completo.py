import os
import cv2
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def procesar_experimento_completo(ruta_raiz):
    print("Iniciando escaneo masivo de imágenes...")
    archivos = glob.glob(os.path.join(ruta_raiz, "**", "*.png"), recursive=True)
    
    # Filtramos la posible basura y nos quedamos con las imágenes reales
    archivos = [a for a in archivos if "Captura de pantalla" not in os.path.basename(a)]
    print(f"Total de imágenes de frutos detectadas: {len(archivos)}")
    
    resultados = []
    
    # Mapeo de meses a números para graficar correctamente el tiempo
    map_mes = {
        'jul': 1, 'julio': 1,
        'ago': 2, 'agosto': 2, 'agos': 2,
        'sep': 3, 'septiembre': 3,
        'oct': 4, 'octubre': 4
    }
    
    for arc in archivos:
        partes = arc.split(os.sep)
        # Asumiendo estructura típica: .../TRICHODERMA/Sinpoda/septiembre/rrr3.png
        tratamiento = "Desconocido"
        poda = "Desconocido"
        mes = "Desconocido"
        
        for i, p in enumerate(partes):
            if p.upper() in ["PROTERRA", "SERENADE", "TESTIGO", "TRICHODERMA"]:
                tratamiento = p.upper()
                if i + 1 < len(partes):
                    poda = partes[i+1].replace(" ", "").lower()
                if i + 2 < len(partes):
                    mes = partes[i+2].lower()
                break
        
        # Procesamiento de NGRDI (Pliman math proxy)
        img = cv2.imread(arc)
        if img is None: continue
        
        img_rz = cv2.resize(img, (224, 224))
        img_float = img_rz.astype(np.float32)
        B, G, R = cv2.split(img_float)
        
        den = (G + R)
        den[den == 0] = 0.0001
        ngrdi = (G - R) / den
        
        ngrdi_norm = cv2.normalize(ngrdi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, umbral_fondo = cv2.threshold(ngrdi_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        fruto_mask = (umbral_fondo == 255)
        # Ajuste de severidad: valores de NGRDI menores a -0.02 representan las manchas necróticas de antracnosis
        lesion_mask = (ngrdi < -0.02) & fruto_mask
        
        pix_totales = np.sum(fruto_mask)
        if pix_totales > 0:
            sev = (np.sum(lesion_mask) / pix_totales) * 100
        else:
            sev = 0
            
        nombre = os.path.basename(arc)
        
        mes_orden = 0
        for k, v in map_mes.items():
            if k in mes:
                mes_orden = v
                break
                
        resultados.append({
            "Archivo": nombre,
            "Tratamiento": tratamiento,
            "Poda": poda,
            "Mes_Tag": mes,
            "Mes_Orden": mes_orden,
            "Severidad_%": round(sev, 2)
        })

    # 1. Crear el CSV Consolidado (Excel-ready)
    df = pd.DataFrame(resultados)
    df.to_csv("Base_Datos_Severidad_Mangos.csv", index=False)
    print("\n-> Creado Excel (CSV) Mestro: 'Base_Datos_Severidad_Mangos.csv'")
    
    # 2. Generar Gráficas (R4PDE y ggplot proxies en Python)
    print("-> Generando gráficas científicas integradas...")
    sns.set_theme(style="whitegrid")
    
    # Gráfica 1: Curva de Progreso Epidemiológico
    plt.figure(figsize=(10, 6))
    df_progreso = df[df['Mes_Orden'] > 0] # Solo graficamos los que entendimos el mes
    if not df_progreso.empty:
        sns.lineplot(data=df_progreso, x="Mes_Orden", y="Severidad_%", hue="Tratamiento", 
                     style="Poda", markers=True, dashes=False, err_style="bars")
        plt.title("Curva de Progreso de Antracnosis en Mangos (Proxy temporal)")
        plt.xlabel("Trancurso de Meses (1=Jul, 4=Oct)")
        plt.ylabel("Severidad Promedio (%)")
        plt.xticks([1,2,3,4], ['Julio', 'Agosto', 'Septiembre', 'Octubre'])
        plt.tight_layout()
        plt.savefig("Curva_Epidemiologica_Progresion.png", dpi=300)
    plt.close()
    
    # Gráfica 2: Boxplot General (Comportamiento de los fungicidas)
    plt.figure(figsize=(9, 6))
    sns.boxplot(data=df, x="Tratamiento", y="Severidad_%", hue="Poda", palette="Set2")
    plt.title("Boxplot Acumulado: Efectividad Poda vs. Tratamiento")
    plt.ylabel("Severidad (%)")
    plt.tight_layout()
    plt.savefig("Boxplot_Tratamientos_Poda.png", dpi=300)
    plt.close()

    print("\n[ÉXITO] Pipeline Finalizado. Gráficas y Excel listos en tu directorio raíz.")

if __name__ == "__main__":
    import yaml # test
    ruta_dataset = r"C:\Users\juanv\Desktop\Analisis mango\fruto_ext"
    procesar_experimento_completo(ruta_dataset)
