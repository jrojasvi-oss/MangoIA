import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def generar_visualizaciones(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: No se encuentra el archivo {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    # Limpiar datos N/A para la gráfica
    df = df[df['Tratamiento'] != 'N/A']
    
    sns.set_theme(style="whitegrid")
    
    # 1. Gráfica de Barras (Media de Severidad)
    plt.figure(figsize=(10, 6))
    order = df.groupby('Tratamiento')['Severidad_%'].mean().sort_values().index
    sns.barplot(x='Tratamiento', y='Severidad_%', data=df, palette='magma', order=order, errorbar='sd')
    plt.title('Eficacia de Tratamientos: Severidad Media de Antracnosis', fontsize=14)
    plt.ylabel('Severidad (%)')
    plt.xlabel('Tratamiento Aplicado')
    plt.tight_layout()
    plt.savefig('GRAFICA_BARRAS_TRATAMIENTOS.png')
    plt.close()
    
    # 2. Boxplot (Distribución y Outliers)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Tratamiento', y='Severidad_%', data=df, palette='Spectral', order=order)
    plt.title('Análisis de Variabilidad: Distribución de Severidad por Grupo', fontsize=14)
    plt.ylabel('Severidad (%)')
    plt.xlabel('Tratamiento Aplicado')
    plt.tight_layout()
    plt.savefig('GRAFICA_BOXPLOT_TRATAMIENTOS.png')
    plt.close()
    
    # 3. Métricas Estadísticas para el reporte
    stats = df.groupby('Tratamiento').agg({
        'Severidad_%': ['mean', 'std', 'max'],
        'Lesiones_Contadas': ['sum', 'mean']
    }).round(2)
    stats.to_csv("TABLA_METRICAS_ESTADISTICAS.csv")
    
    print("[!] Gráficas y tablas métricas generadas con éxito.")

if __name__ == "__main__":
    path = r"C:\Users\juanv\Desktop\Analisis mango\Resultados_Mapeo_Cientifico\DATOS_CSV\Mapeo_Cientifico_Total_Fruto.csv"
    generar_visualizaciones(path)
