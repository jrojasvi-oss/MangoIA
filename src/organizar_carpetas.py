import os
import shutil
import glob

# Rutas principales
ruta_raiz = r"C:\Users\juanv\Desktop\Analisis mango"
ruta_origen = os.path.join(ruta_raiz, "fruto_ext")
ruta_dataset_limpio = os.path.join(ruta_raiz, "Fotos_Ordenadas")
ruta_analitica = os.path.join(ruta_raiz, "Resultados_Analiticos")

# Crear carpeta de reportes y mover Excels/Graficas allí
print("1. Creando carpeta de Resultados Analíticos...")
os.makedirs(ruta_analitica, exist_ok=True)

archivos_analiticos = [
    "Base_Datos_Severidad_Mangos.csv",
    "severidad_raw_frutos.csv",
    "severidad_analizada_python.csv",
    "Curva_Epidemiologica_Progresion.png",
    "Boxplot_Tratamientos_Poda.png",
    "rf_importancia_predictores.png",
    "pca_multivariado.png",
    "ggplot_curva_progreso.png",
    "ggplot_boxplot.png",
    "ejemplo_diagnostico_calibracion.png"
]

for arc in archivos_analiticos:
    ruta_arc = os.path.join(ruta_raiz, arc)
    if os.path.exists(ruta_arc):
        shutil.move(ruta_arc, os.path.join(ruta_analitica, arc))
        print(f"   -> Movido: {arc}")

# Organizar las fotos en una nueva carpeta limpia
print("\n2. Reestructurando y ordenando las fotos rigurosamente...")
os.makedirs(ruta_dataset_limpio, exist_ok=True)

archivos = glob.glob(os.path.join(ruta_origen, "**", "*.png"), recursive=True)
archivos_fotos = [a for a in archivos if "Captura de pantalla" not in os.path.basename(a)]

# Diccionario de equivalencia de meses para dejarlos bonitos
map_mes_nombre = {
    'jul': '01_Julio', 'julio': '01_Julio',
    'ago': '02_Agosto', 'agosto': '02_Agosto', 'agos': '02_Agosto',
    'sep': '03_Septiembre', 'septiembre': '03_Septiembre',
    'oct': '04_Octubre', 'octubre': '04_Octubre'
}

fotos_movidas = 0

for arc in archivos_fotos:
    partes = arc.split(os.sep)
    tratamiento = "Otros"
    poda = "Desconocido"
    mes_original = ""
    
    # Extraer las variables basadas en la ruta vieja
    for i, p in enumerate(partes):
        if p.upper() in ["PROTERRA", "SERENADE", "TESTIGO", "TRICHODERMA"]:
            tratamiento = p.upper()
            if i + 1 < len(partes):
                p_poda = partes[i+1].replace(" ", "").lower()
                poda = "Con_Poda" if "con" in p_poda else "Sin_Poda"
            if i + 2 < len(partes):
                mes_original = partes[i+2].lower()
            break
            
    # Asignar mes limpio
    mes_limpio = "00_Desconocido"
    for clave, valor in map_mes_nombre.items():
        if clave in mes_original:
            mes_limpio = valor
            break
            
    # Crear ruta de destino impecable
    ruta_dest_dir = os.path.join(ruta_dataset_limpio, tratamiento, poda, mes_limpio)
    os.makedirs(ruta_dest_dir, exist_ok=True)
    
    nombre_archivo = os.path.basename(arc)
    ruta_dest_final = os.path.join(ruta_dest_dir, nombre_archivo)
    
    # Copiar el archivo al nuevo layout
    shutil.copy2(arc, ruta_dest_final)
    fotos_movidas += 1

print(f"\n¡Éxito! {fotos_movidas} fotos ordenadas milimétricamente en '{ruta_dataset_limpio}'")
print(f"Todos los Excels y PNGs fueron aislados en '{ruta_analitica}'")
