import os
import shutil

def organize():
    # Definir carpetas destino
    folders = ['src', 'scripts_r', 'docs', 'models', 'data_samples', 'results']
    
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"[*] Creada carpeta: {folder}")

    # Clasificación de archivos
    mapping = {
        'src': [
            'autonomous_anthracnose_pipeline.py', 'pipeline_final_cepas.py', 
            'antracnosis_spectral_analyzer.py', 'generar_informe.py', 
            'deteccion_yolo_pura.py', 'mango_sam_segmenter.py', 
            'analisis_masivo_completo.py', 'crear_dataset_yolo.py', 
            'entrenamiento_autonomo.py', 'entrenar_mango_50epocas.py', 
            'entrenar_yolo.py', 'escala_y_mascaras.py', 
            'generar_diagnosticos.py', 'generar_excel.py', 
            'generar_graficas_finales.py', 'generar_mapa_espectral.py', 
            'generar_mapa_espectral_masivo.py', 'generar_muestra_explicada.py', 
            'integracion_paso_a_paso.py', 'integracion_robusta.py', 
            'limpiar_datos.py', 'mango_advanced_3cards_report.py', 
            'mango_create_1x3_panel.py', 'mango_disease_pipeline.py', 
            'mango_final_scientific_organization.py', 'mango_full_scientific_mapping.py', 
            'mango_massive_batch_analysis.py', 'mango_master_scientific_cloner.py', 
            'mango_roboflow_mask_mapper.py', 'mega_dashboard_graficos.py', 
            'organizar_carpetas.py', 'procesar_todo_mango.py', 
            'segmentacion_realista.py', 'vit_pliman_pipeline.py', 
            'yolo_mango.py'
        ],
        'scripts_r': [
            'analisis_cientifico.R', 'analisis_pliman_definitivo.R', 
            'analisis_pliman_foto_x_foto.R', 'analisis_pliman_post.R', 
            'app_pliman_interactiva.R', 'pipeline_avanzado.R'
        ],
        'docs': [
            'COMO_USAR_PROMPT.md', 'INSTALACION.md', 
            'PROMPT_INVESTIGACION_AMPLÍA.md', 'REFERENCIA_RÁPIDA.md', 
            'TARJETA_REFERENCIA.md', 'ÍNDICE_COMPLETO.md',
            'pipeline_antracnosis_mango_pliman.docx', 
            'pipeline_pliman_R_antracnosis_PROTERRA.docx'
        ],
        'models': [
            'yolov8n.pt', 'sam_b.pt'
        ]
    }

    # Mover archivos
    for folder, files in mapping.items():
        for file in files:
            if os.path.exists(file):
                try:
                    shutil.move(file, os.path.join(folder, file))
                    print(f"    [OK] {file} -> {folder}/")
                except Exception as e:
                    print(f"    [ERR] No se pudo mover {file}: {e}")

    # Mover archivos de datos específicos si existen
    data_files = ['DATOS_LIMPIOS_PARA_R.csv', 'REPORTE_MASIVO_ANTRACO_MANGO.csv', 'TABLA_METRICAS_ESTADISTICAS.csv']
    for df in data_files:
        if os.path.exists(df):
            shutil.move(df, os.path.join('data_samples', df))
            print(f"    [OK] {df} -> data_samples/")

    print("\n[!] PROCESO DE ORGANIZACIÓN FINALIZADO.")
    print("[*] Tu repositorio ahora está listo para GitHub.")

if __name__ == "__main__":
    organize()
