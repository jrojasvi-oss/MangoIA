import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

ruta_csv = r"C:\Users\juanv\Desktop\Analisis mango\Resultados_Analiticos\Base_Datos_Severidad_Mangos.csv"
ruta_graficacion = r"C:\Users\juanv\Desktop\Analisis mango\Informe_Graficacion"

os.makedirs(ruta_graficacion, exist_ok=True)
print("Iniciando compilación del Informe y Gráficas...")

try:
    df = pd.read_csv(ruta_csv)
    
    # Preparamos el exportador de PDF
    pdf_path = os.path.join(ruta_graficacion, "Reporte_Epi_Antracnosis.pdf")
    pdf_pages = PdfPages(pdf_path)
    sns.set_theme(style="whitegrid", context="talk")

    # 1. Curva de Progreso (Evolución de la enfermedad)
    fig1 = plt.figure(figsize=(10, 6))
    sns.lineplot(data=df[df['Mes_Orden'] > 0], x="Mes_Orden", y="Severidad_%", hue="Tratamiento", 
                 style="Poda", markers=True, dashes=False, err_style="bars", linewidth=2.5)
    plt.title("Curva de Progreso Epidemiológico: Supresión de Antracnosis")
    plt.xlabel("Línea de Tiempo (1=Julio, 4=Octubre)")
    plt.ylabel("Área Severamente Necrosada (%)")
    plt.xticks([1, 2, 3, 4], ['Jul', 'Ago', 'Sep', 'Oct'])
    plt.tight_layout()
    # Guardar PNG en la nueva carpeta
    plt.savefig(os.path.join(ruta_graficacion, "01_Curva_Epidemiologica.png"), dpi=300)
    pdf_pages.savefig(fig1)
    
    # 2. Boxplot Acumulado
    fig2 = plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="Tratamiento", y="Severidad_%", hue="Poda", palette="YlGnBu", showmeans=True)
    plt.title("Comparación Paramétrica Efecto/Tratamiento")
    plt.ylabel("Severidad(%) Relativa")
    plt.tight_layout()
    plt.savefig(os.path.join(ruta_graficacion, "02_Boxplot_Efecto.png"), dpi=300)
    pdf_pages.savefig(fig2)

    # 3. Mapa de Calor (Interacciones)
    if 'Mes_Orden' in df.columns:
        pivot = df.pivot_table(index="Tratamiento", columns="Mes_Orden", values="Severidad_%", aggfunc=np.mean)
        fig3 = plt.figure(figsize=(8, 6))
        sns.heatmap(pivot, annot=True, cmap="Reds", fmt=".1f", linewidths=.5)
        plt.title("Zonas Críticas de Proliferación")
        plt.xlabel("Mes (Numérico)")
        plt.tight_layout()
        plt.savefig(os.path.join(ruta_graficacion, "03_Heatmap_Interaccion.png"), dpi=300)
        pdf_pages.savefig(fig3)

    # 4. Tabla de Severidad por Foto (Archivo/Etiqueta)
    print("Generando tablas de datos para el PDF...")
    df_table = df[['Archivo', 'Tratamiento', 'Severidad_%']].copy()
    
    # Ordenar por severidad para mayor claridad
    df_table = df_table.sort_values(by='Severidad_%', ascending=False)
    
    # Agrupar en lotes de 30 filas por página para que quepan bien
    rows_per_page = 30
    num_pages = int(np.ceil(len(df_table) / rows_per_page))
    
    for i in range(num_pages):
        fig_tbl = plt.figure(figsize=(10, 8))
        ax = fig_tbl.add_subplot(111)
        ax.axis('tight')
        ax.axis('off')
        
        chunk = df_table.iloc[i * rows_per_page : (i + 1) * rows_per_page]
        
        if i == 0:
            plt.title("Tabla de Severidad por Etiqueta de Foto", fontsize=14, fontweight='bold', pad=20)
        else:
            plt.title(f"Tabla de Severidad por Etiqueta de Foto (Pág {i+1})", fontsize=12, pad=20)
            
        table = ax.table(cellText=chunk.values, colLabels=chunk.columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        pdf_pages.savefig(fig_tbl)

    # Cerrar PDF master
    pdf_pages.close()
    plt.close('all')

    print("\n[✔] GRÁFICOS Y ARCHIVOS EXPORTADOS A:")
    print(ruta_graficacion)
    print("Reporte PDF generado exitosamente con tablas de datos.")
    
except Exception as e:
    print(f"Error procesando la graficación automatizada: {e}")
