import pandas as pd
import os

def generate_excel_report(csv_path, output_path):
    df = pd.read_csv(csv_path)
    
    # 1. Resumen por Cepa y Mes
    summary = df.groupby(['Cepa', 'Mes']).agg({
        'Imagen': 'count',
        'Salud_%': 'mean',
        'Afectado_%': 'mean'
    }).rename(columns={'Imagen': 'Total_Frutos', 'Salud_%': 'Salud_Promedio', 'Afectado_%': 'Severidad_Media'})
    
    # 2. Cálculo de Incidencia
    incidence = df.groupby(['Cepa', 'Mes']).apply(lambda x: (x['Afectado_%'] > 0.5).sum() / len(x) * 100).reset_index(name='Incidencia_%')
    
    # Unir
    final_summary = pd.merge(summary, incidence, on=['Cepa', 'Mes'])
    
    # 3. Resumen por Cepa Total
    cepa_summary = df.groupby('Cepa').agg({
        'Imagen': 'count',
        'Afectado_%': 'mean'
    }).rename(columns={'Imagen': 'Total_Frutos', 'Afectado_%': 'Severidad_Cepa'})
    cepa_incidence = df.groupby('Cepa').apply(lambda x: (x['Afectado_%'] > 0.5).sum() / len(x) * 100).reset_index(name='Incidencia_Total_%')
    cepa_final = pd.merge(cepa_summary, cepa_incidence, on='Cepa')
    
    # Guardar en Excel con múltiples hojas
    with pd.ExcelWriter(output_path) as writer:
        final_summary.to_excel(writer, sheet_name='Resumen_Mensual', index=False)
        cepa_final.to_excel(writer, sheet_name='Resumen_Tratamientos', index=False)
        df.to_excel(writer, sheet_name='Datos_Detallados', index=False)
        
    print(f"[*] Reporte Excel generado en: {output_path}")

if __name__ == "__main__":
    SOURCE = r"C:\Users\juanv\Desktop\Analisis mango\DATOS_LIMPIOS_PARA_R.csv"
    TARGET = r"C:\Users\juanv\Desktop\Analisis mango\REPORTE_ESTADISTICO_MANGO_ANTRACO.xlsx"
    generate_excel_report(SOURCE, TARGET)
