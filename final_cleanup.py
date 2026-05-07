import os
import shutil
import glob

def cleanup():
    # Mover archivos de medios
    os.makedirs('media', exist_ok=True)
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.mp4']:
        for f in glob.glob(ext):
            shutil.move(f, os.path.join('media', f))
            print(f"Moved {f} to media/")

    # Mover carpetas de resultados
    os.makedirs('results', exist_ok=True)
    result_patterns = ['FRUTO_*', 'RESULTADOS_*', 'ANALISIS_*', 'SEGMENTACION_*', 'RUNS_*', 
                       'VISUALIZACION_*', 'MAPAS_*', 'REPORTES_*', 'Estudio_*', 'Fotos_*', 
                       'Informe_*', 'Resultados_*', 'MUESTRA_*', 'TEST_*', 'ENTREGA_*']
    for pattern in result_patterns:
        for d in glob.glob(pattern):
            if os.path.isdir(d) and d != 'results':
                try:
                    shutil.move(d, os.path.join('results', os.path.basename(d)))
                    print(f"Moved directory {d} to results/")
                except Exception as e:
                    print(f"Error moving {d}: {e}")

    # Renombrar README
    if os.path.exists('README_GITHUB.md'):
        if os.path.exists('README.md'):
            os.remove('README.md')
        os.rename('README_GITHUB.md', 'README.md')
        print("Updated README.md")

if __name__ == "__main__":
    cleanup()
