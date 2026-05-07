# =========================================================================# =========================================================================
# APLICACIÓN INTERACTIVA (SHINY APP) PARA ANTRACNOSIS CON PLIMAN
# Interfaz Gráfica para ver, seleccionar y analizar imágenes científicamente.
# =========================================================================

# Instalar librerías necesarias para la interfaz si no las tienes
paquetes <- c("shiny", "bslib", "pliman", "tidyverse", "shinyWidgets", "gridExtra")
nuevos <- paquetes[!(paquetes %in% installed.packages()[,"Package"])]
if (length(nuevos)) install.packages(nuevos)

library(shiny)
library(bslib)
library(pliman)
library(tidyverse)
library(shinyWidgets)
library(gridExtra)

# =========================================================================
# 1. INTERFAZ DE USUARIO (UI) - Bien Explicada y Científica
# =========================================================================
ui <- page_sidebar(
  title = "MangoIA: Análisis Interactivo de Antracnosis (Pliman)",
  theme = bs_theme(version = 5, bootswatch = "cosmo", primary = "#1A5C2E"),
  
  # Barra lateral para subir fotos
  sidebar = sidebar(
    width = 320,
    h4("1. Sube tu imagen"),
    fileInput("imagen_fruto", "Selecciona una foto de mango (.png, .jpg)",
              accept = c("image/png", "image/jpeg", "image/jpg")
    ),
    
    hr(),
    h4("2. Segmentación Automática"),
    helpText("Pliman utilizará el índice de color (NGRDI) para detectar lesiones fúngicas automáticamente."),
    
    # Parámetros científicos
    selectInput("indice_fondo", "Índice de eliminación de fondo:", choices = c("B", "NGRDI", "R", "G"), selected = "NGRDI"),
    selectInput("indice_enfermedad", "Índice de lesión (Antracnosis):", choices = c("NGRDI", "GLI", "R, G, B"), selected = "NGRDI"),
    
    actionBttn("analizar_btn", "Calcular Área con Pliman", style = "jelly", color = "success", size = "md"),
    
    hr(),
    h5("Resultados de Severidad"),
    uiOutput("resultados_ciencia")
  ),
  
  # Panel principal para mostrar las imágenes y evidencias
  card(
    card_header("Evidencia Científica y Contrastes Visuales"),
    p("Aquí podrás visualizar qué está entendiendo el modelo cuando mira la foto del fruto. Revisa si la segmentación es precisa.", style="color: #666;"),
    
    fluidRow(
      column(6, 
             h5("Imagen Original del Fruto", align = "center"),
             plotOutput("plot_original", height = "500px") %>% 
               shinycssloaders::withSpinner(color="#1A5C2E")
      ),
      column(6, 
             h5("Área Calculada por Pliman (Diagnóstico)", align = "center"),
             plotOutput("plot_segmentada", height = "500px") %>% 
               shinycssloaders::withSpinner(color="#C65911")
      )
    )
  )
)

# =========================================================================
# 2. LÓGICA DEL SERVIDOR (BACKEND)
# =========================================================================
server <- function(input, output, session) {
  
  # Variable reactiva para almacenar la imagen cargada
  imagen_cargada <- reactive({
    req(input$imagen_fruto)
    img <- image_import(input$imagen_fruto$datapath)
    return(img)
  })
  
  # Mostrar la imagen original apenas se suba
  output$plot_original <- renderPlot({
    req(imagen_cargada())
    plot(imagen_cargada())
  })
  
  # Evento: Al hacer clic en el botón de Analizar
  analisis_reactivo <- eventReactive(input$analizar_btn, {
    req(imagen_cargada())
    
    # Pliman corre la medición de enfermedad con el índice elegido
    # Esto es lo más parecido al método automático de tu DOCX
    res <- measure_disease(
      img = imagen_cargada(),
      index_lb = input$indice_fondo,       # Para separar el mango del fondo negro/blanco
      index_dh = input$indice_enfermedad,  # Para extraer la mancha oscura del mango verde
      threshold = "Otsu",                  # Segmentación por umbral de Otsu estandar
      show_image = FALSE,
      verbose = FALSE
    )
    
    return(res)
  })
  
  # Mostrar la imagen resultante con los pixeles clasificados (sanos vs enfermos)
  output$plot_segmentada <- renderPlot({
    req(analisis_reactivo())
    
    # Pliman guarda el plot de diagnóstico con 'plot(res)' o accediendo a la imagen de salida
    # 'image_index' si quisiéramos mostrar el mapa de color, pero measure_disease 
    # nativamente arroja la salida cuando llamamos a plot.
    
    # Para poder hacer output, llamamos a la función genérica
    img_sintoma <- analisis_reactivo()$shape
    
    if(is.null(analisis_reactivo()$shape)) {
       plot(imagen_cargada())
       title("Aún se está procesando o no hubo detección", col.main="red")
    } else {
        # Como measure_disease no devuelve la imagen renderizada al objeto fácilmente,
        # lo corremos en vivo para el renderPlot mostrando las máscaras:
        measure_disease(
          img = imagen_cargada(),
          index_lb = input$indice_fondo,
          index_dh = input$indice_enfermedad,
          threshold = "Otsu",
          show_image = TRUE
        )
    }
  })
  
  # Imprimir texto y ciencia pura con los resultados de severidad en la barra
  output$resultados_ciencia <- renderUI({
    req(analisis_reactivo())
    data_severidad <- analisis_reactivo()$severity
    
    HTML(paste0(
      "<div style='padding: 10px; background: #e8f5e9; border-radius: 8px;'>",
      "<b>Mango Total Evaluado:</b> ", round(data_severidad$healthy + data_severidad$symptomatic, 2), "%<br/>",
      "<b>Tejido Sano (Limpio):</b> <span style='color:green;'>", round(data_severidad$healthy, 2), "%</span><br/>",
      "<b>Tejido Enfermo (Antracnosis):</b> <span style='color:red; font-size: 1.2em; font-weight: bold;'>", round(data_severidad$symptomatic, 2), "%</span>",
      "</div>"
    ))
  })
}

# Iniciar la interfaz gráfica
shinyApp(ui, server)
