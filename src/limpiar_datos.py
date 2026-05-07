import pandas as pd
import os
import re

def clean_data(csv_path, output_path):
    df = pd.read_csv(csv_path)
    
    # Listas para las nuevas columnas
    cepas = []
    podas = []
    meses = []
    
    for folder in df['Carpeta']:
        folder = folder.upper()
        
        # Identificar Cepa
        if 'PROTERRA' in folder: cepa = 'PROTERRA'
        elif 'SERENADE' in folder: cepa = 'SERENADE'
        elif 'TESTIGO' in folder: cepa = 'TESTIGO'
        elif 'TRICHODERMA' in folder: cepa = 'TRICHODERMA'
        else: cepa = 'OTROS'
        
        # Identificar Poda
        if 'SIN' in folder: poda = 'SIN PODA'
        elif 'CON' in folder: poda = 'CON PODA'
        else: poda = 'GENERAL'
        
        # Identificar Mes (Aproximación por palabras clave)
        if 'JUL' in folder: mes = 'JULIO'
        elif 'AGO' in folder: mes = 'AGOSTO'
        elif 'SEP' in folder: mes = 'SEPTIEMBRE'
        elif 'OCT' in folder: mes = 'OCTUBRE'
        elif 'MAY' in folder: mes = 'MAYO'
        elif 'ABR' in folder: mes = 'ABRIL'
        else: mes = 'INICIAL'
        
        cepas.append(cepa)
        podas.append(poda)
        meses.append(mes)
        
    df['Cepa'] = cepas
    df['Poda'] = podas
    df['Mes'] = meses
    
    # Guardar CSV limpio para R
    df.to_csv(output_path, index=False)
    print(f"[*] Datos limpiados y guardados en: {output_path}")

if __name__ == "__main__":
    SOURCE_CSV = r"C:\Users\juanv\Desktop\Analisis mango\FRUTO_CEPAS_CIENTIFICO_FINAL\CONSOLIDADO_CEPAS_PROCESADAS.csv"
    OUTPUT_CSV = r"C:\Users\juanv\Desktop\Analisis mango\DATOS_LIMPIOS_PARA_R.csv"
    clean_data(SOURCE_CSV, OUTPUT_CSV)
