# =========================================================================
# ANÁLISIS MAESTRO DE MANGO (ANTRACNOSIS)
# Herramientas: Pliman, RandomForest, PCA (Multivariado), ggplot2
# =========================================================================

# 1. Instalar y cargar pacotes requeridos
paquetes <- c("pliman", "tidyverse", "randomForest", "FactoMineR", "factoextra", "corrplot")
nuevos <- paquetes[!(paquetes %in% installed.packages()[,"Package"])]
if (length(nuevos)) install.packages(nuevos)

library(pliman)
library(tidyverse)
library(randomForest)
library(FactoMineR)
library(factoextra)
library(ggplot2)

# =========================================================================
# 2. CARGA DE DATOS DE SEVERIDAD
# =========================================================================
# Asegúrate de generar previamente tu archivo "severidad_raw_frutos.csv" con pliman
# Aquí hacemos un tryCatch para cargar o simular datos en caso no exista
archivo_datos <- "severidad_raw_frutos.csv"

if(file.exists(archivo_datos)) {
  datos <- read.csv(archivo_datos)
  print("Datos reales cargados con éxito!")
} else {
  print("Simulando datos para demostración. Recuerda procesar las imágenes con pliman primero.")
  datos <- data.frame(
    tratamiento = rep(c("PROTERRA", "SERENADE", "TESTIGO", "TRICHODERMA"), each = 24),
    poda = rep(c("Con_Poda", "Sin_Poda"), times = 48),
    mes_ord = rep(1:4, times = 24),
    sev_pct = runif(96, 5, 80) # Porcentajes aleatorios simulaods
  )
}

# =========================================================================
# 3. RANDOM FOREST (PREDICCIÓN/IMPORTANCIA)
# Objetivo: Identificar qué factores afectan más la severidad de la enfermedad
# =========================================================================
print("Ejecutando modelo de Random Forest...")
rf_modelo <- randomForest(
  as.factor(tratamiento) ~ sev_pct + poda + mes_ord, 
  data = datos, 
  ntree = 500, 
  importance = TRUE
)

# Guardar la gráfica de importancia de predictores
png("rf_importancia_predictores.png", width=800, height=600)
varImpPlot(rf_modelo, main = "Importancia de Predictores (Random Forest)")
dev.off()
print("Gráfica Random Forest guardada como rf_importancia_predictores.png")

# =========================================================================
# 4. ANÁLISIS MULTIVARIADO (PCA)
# Objetivo: Encontrar relaciones complejas entre poda, tratamiento y enfermedad
# =========================================================================
print("Generando Análisis de Componentes Principales (PCA)...")
datos_pca <- datos %>%
  group_by(tratamiento, poda, mes_ord) %>%
  summarise(
    Severidad_Media = mean(sev_pct, na.rm=TRUE),
    Desviacion = sd(sev_pct, na.rm=TRUE),
    .groups = 'drop'
  )

# Prevenimos sd NA en datos muy pequeños para PCA
datos_pca$Desviacion[is.na(datos_pca$Desviacion)] <- 0 

res.pca <- PCA(datos_pca[, c("Severidad_Media", "Desviacion")], graph = FALSE)

# Gráfico Multivariado estético usando factoextra y ggplot2
p_pca <- fviz_pca_biplot(res.pca, 
                         habillage = as.factor(datos_pca$tratamiento),
                         addEllipses = TRUE, 
                         palette = "jco",
                         title = "PCA Multivariado: Tratamiento vs Comportamiento de Enfermedad",
                         legend.title = "Tratamiento") + 
         theme_minimal()

ggsave("pca_multivariado.png", plot = p_pca, width = 8, height = 6)
print("Gráfica PCA guardada como pca_multivariado.png")

# =========================================================================
# 5. VISUALIZACIONES AVANZADAS (GGPLOT2)
# =========================================================================
print("Generando Curvas de Progreso usando ggplot2...")

# 5.1 Gráfico de Curva de Progreso (Evolución)
p_curva <- ggplot(datos, aes(x = mes_ord, y = sev_pct, color = tratamiento, group = tratamiento)) +
  stat_summary(fun = mean, geom = "line", linewidth = 1.2) +
  stat_summary(fun = mean, geom = "point", size = 3) +
  facet_wrap(~poda) +
  scale_color_brewer(palette = "Set1") +
  theme_light() +
  labs(
    title = "Evolución de la Antracnosis",
    subtitle = "Integrando Podas y Tratamientos",
    x = "Mes (N°)",
    y = "Severidad de Infección (%)"
  ) +
  theme(text = element_text(size = 14, family = "sans"))

ggsave("ggplot_curva_progreso.png", plot = p_curva, width = 8, height = 6)

# 5.2 Boxplot Comparativo General
p_box <- ggplot(datos, aes(x = tratamiento, y = sev_pct, fill = tratamiento)) +
  geom_boxplot(alpha=0.7, outlier.colour="red", outlier.shape=16) +
  theme_classic() +
  labs(title="Distribución de la Severidad por Tratamiento", y="% Daño", x="Tratamiento") +
  coord_flip() +
  scale_fill_brewer(palette = "Set2")

ggsave("ggplot_boxplot.png", plot = p_box, width = 8, height = 6)
print("Gráficas y pipeline finalizado exitosamente. Revisa tu carpeta para ver los PNGs generados.")
