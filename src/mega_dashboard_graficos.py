import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Rutas
ruta_csv = r"C:\Users\juanv\Desktop\Analisis mango\Resultados_Analiticos\Base_Datos_Severidad_Mangos.csv"
ruta_graficos = r"C:\Users\juanv\Desktop\Analisis mango\Resultados_Analiticos\Mega_Dashboard"

print("Cargando la base de datos maestra para despliegue masivo de gráficos...")
os.makedirs(ruta_graficos, exist_ok=True)

try:
    df = pd.read_csv(ruta_csv)
    # Limpieza breve
    df = df[df['Severidad_%'] >= 0]
    
    # 1. Violin Plot con Swarm (Distribución profunda)
    print("1/5 Generando Mapa de Violín (Distribución anatómica)...")
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x="Tratamiento", y="Severidad_%", hue="Poda", split=True, inner="quart", palette="muted")
    plt.title("Biometría de Infección: Distribuciones (Violin Plot)")
    plt.tight_layout()
    plt.savefig(os.path.join(ruta_graficos, "01_ViolinPlot_Biometria.png"), dpi=300)
    plt.close()

    # 2. Mapa de Calor (Heatmap) Tratamientos vs Meses
    print("2/5 Construyendo Matriz de Calor Epidémica (Heatmap)...")
    if 'Mes_Orden' in df.columns:
        pivot = df.pivot_table(index="Tratamiento", columns="Mes_Orden", values="Severidad_%", aggfunc=np.mean)
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot, annot=True, cmap="YlOrRd", fmt=".1f", linewidths=.5)
        plt.title("Mapa de Calor Térmico: Focos de Extrema Severidad (%)")
        plt.xlabel("Línea de Tiempo (Meses 1=Jul a 4=Oct)")
        plt.tight_layout()
        plt.savefig(os.path.join(ruta_graficos, "02_Heatmap_Severidad.png"), dpi=300)
        plt.close()

    # 3. Gráfica de Densidad Kernel (KDE) - Análisis de Olas
    print("3/5 Trazando Ondas Gaussianas (KDE)...")
    plt.figure(figsize=(9, 5))
    sns.kdeplot(data=df, x="Severidad_%", hue="Tratamiento", fill=True, common_norm=False, palette="crest", alpha=.5, linewidth=0)
    plt.title("Ondas de Densidad de Probabilidad (KDE)")
    plt.tight_layout()
    plt.savefig(os.path.join(ruta_graficos, "03_Ondas_KDE_Densidad.png"), dpi=300)
    plt.close()
    
    # 4. Barplot Estadístico de Barras de Error Analítico
    print("4/5 Modelando Barras con Intervalos de Confianza...")
    plt.figure(figsize=(9, 6))
    sns.barplot(data=df, x="Mes_Tag", y="Severidad_%", hue="Tratamiento", capsize=.1, palette="pastel")
    plt.title("Margen de Severidad y Barras de Estimación (95% CI)")
    plt.tight_layout()
    plt.savefig(os.path.join(ruta_graficos, "04_Barras_Error_Estadisticos.png"), dpi=300)
    plt.close()

    # 5. Scatter Plot Grid Cruzado (FacetGrid Disperso)
    print("5/5 Generando Grid Multivariable...")
    g = sns.FacetGrid(df, col="Poda", hue="Tratamiento", palette="dark", height=5)
    g.map_dataframe(sns.scatterplot, x="Mes_Orden", y="Severidad_%", s=80, alpha=0.7)
    g.add_legend()
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle("Dispersión Atómica de Cada Singular Fruto Muestreado")
    plt.savefig(os.path.join(ruta_graficos, "05_Scatter_Atomico.png"), dpi=300)
    plt.close()

    print("\n[✔] ¡Arsenal Gráfico Completado con Éxito Extremo!")
    print(f"Directorio de Destino: {ruta_graficos}")

except Exception as e:
    print(f"Hubo un error cargando el CSV o procesando: {e}")
