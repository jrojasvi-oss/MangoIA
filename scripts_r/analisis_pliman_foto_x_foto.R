# =========================================================================
# ANÁLISIS ESTRICTO FOTO POR FOTO: SOLO 'PLIMAN' NATIVO
# Procesando Áreas Sanas vs. Áreas Afectadas directamente con R
# =========================================================================

# Cargar las librerías oficiales de botánica y fenotipado
library(pliman)

# 1. Definir la carpeta maestra donde ordenamos las fotos previamente
ruta_fotos <- "C:/Users/juanv/Desktop/Analisis mango/Fotos_Ordenadas"
ruta_reportes <- "C:/Users/juanv/Desktop/Analisis mango/Resultados_Analiticos/Pliman_Visual"

if(!dir.exists(ruta_reportes)) dir.create(ruta_reportes, recursive = TRUE)

# Buscar todas las fotos (recursivamente)
archivos_png <- list.files(path = ruta_fotos, pattern = "\\.png$", full.names = TRUE, recursive = TRUE)

message(sprintf("*** INICIANDO ESCANEO PLIMAN (%d frutos) ***", length(archivos_png)))

# Crear un contenedor para la tabla que me pediste
tabla_foto_x_foto <- data.frame(
  Fruto = character(),
  Tratamiento = character(),
  Area_Total = numeric(),
  Area_Sana = numeric(),
  Area_Afectada = numeric(),
  Indice_Afectada_Sobre_Sana = numeric(),
  Indice_Afectada_Sobre_Total = numeric(), # La severidad clásica
  stringsAsFactors = FALSE
)

# 2. Análisis Bucle Foto por Foto (Exigencia del usuario)
for (i in seq_along(archivos_png)) {
  archivo_actual <- archivos_png[i]
  nombre_foto <- basename(archivo_actual)
  
  # Extraer info para el reporte
  rutas_split <- unlist(strsplit(archivo_actual, "/|\\\\"))
  tratamiento <- rutas_split[length(rutas_split)-3]
  
  # Importar imagen en formato oficial de pliman (ImageMatrix)
  img <- image_import(archivo_actual)
  
  # === LA MATEMÁTICA PURA DE PLIMAN ===
  # index_lb = "B" -> Extrae el fruto separándolo del fondo
  # index_dh = "NGRDI" -> Distingue los tonos vivos de la oscuridad necrótica (Antracnosis)
  
  analisis <- tryCatch(
    measure_disease(
      img = img, 
      index_lb = "B",
      index_dh = "NGRDI",
      threshold = c("Otsu", 0), # Otsu para el fondo, umbral estricto para diferenciar
      show_features = FALSE,
      plot = FALSE # Apagamos el plot para que no reviente la memoria al procesar muchas
    ),
    error = function(e) { NULL }
  )
  
  # Si la función logra encontrar el fruto y procesarlo
  if (!is.null(analisis)) {
    # Extraer pixeles numéricos para la fórmula exigida
    pix_sanos <- analisis$severity$healthy_pix
    pix_enfermos <- analisis$severity$symptomatic_pix
    pix_totales <- pix_sanos + pix_enfermos
    
    # Índice especificado (Afectada / Sana)
    indice_personalizado <- ifelse(pix_sanos > 0, pix_enfermos / pix_sanos, NA)
    # Severity clásica (Afectada / Total %)
    severidad_clasica <- analisis$severity$symptomatic
    
    # 3. Guardar el comprobante visual "Foto por Foto" 
    # (Sólo guardaremos unas pocas (ej. primeras 5) para que no demore horas en tu PC local, puedes cambiarlo)
    if (i <= 5) {
      ruta_visual = file.path(ruta_reportes, paste0("Pliman_Mask_", nombre_foto))
      png(ruta_visual, width = 800, height = 800)
      # Función nativa de plot de pliman que pinta (Sano verde / Afectado azul/rojo)
      plot(analisis$shape, main = paste("Severidad:", round(severidad_clasica, 2), "%\n", tratamiento))
      dev.off()
    }
    
    # Agregar los datos a la tabla maestra
    tabla_foto_x_foto <- rbind(tabla_foto_x_foto, data.frame(
      Fruto = nombre_foto,
      Tratamiento = tratamiento,
      Area_Total = pix_totales,
      Area_Sana = pix_sanos,
      Area_Afectada = pix_enfermos,
      Indice_Afectada_Sobre_Sana = indice_personalizado,
      Indice_Afectada_Sobre_Total = severidad_clasica
    ))
  }
  
  # Imprimir progreso en RStudio
  if (i %% 20 == 0) message(paste(">> Procesados:", i, "mangos..."))
}

# 4. Guardar archivo final
ruta_csv <- file.path(ruta_reportes, "Pliman_Reporte_FotoXFoto_Estricto.csv")
write.csv(tabla_foto_x_foto, ruta_csv, row.names = FALSE)

message("\n[✔] ANÁLISIS FINALIZADO CON PLIMAN.")
message(sprintf("La base de datos se guardó exitosamente en:\n %s", ruta_csv))
