# 🧪 Análisis Fitopatológico de Precisión con PLIMAN
# Autor: Antigravity AI
# Objetivo: Procesar el consolidado espectral y generar visualizaciones científicas.

library(pliman)
library(tidyverse)
library(ggplot2)
library(gridExtra)

# 1. CARGAR DATOS
# Asegúrate de haber ejecutado el pipeline de Python primero.
data_path <- "c:/Users/juanv/Desktop/Analisis mango/RUNS_SPECTRAL_FINAL/CONSOLIDADO_ANT_SPECTRAL.csv"

if (!file.exists(data_path)) {
  stop("¡Error! No se encuentra el archivo consolidado. Ejecuta mango_disease_pipeline.py primero.")
}

df <- read.csv(data_path)

# 2. LIMPIEZA Y CATEGORIZACIÓN
df_clean <- df %>%
  mutate(
    Cepa = as.factor(Cepa.Tratamiento),
    Nivel_Severidad = case_when(
      Severidad_.. < 5 ~ "Leve",
      Severidad_.. < 15 ~ "Moderado",
      TRUE ~ "Severo"
    )
  )

# 3. VISUALIZACIÓN CIENTÍFICA

# A. Boxplot de Severidad por Tratamiento
p1 <- ggplot(df_clean, aes(x = Cepa, y = Severidad_.., fill = Cepa)) +
  geom_boxplot(alpha = 0.7, outlier.colour = "red") +
  geom_jitter(width = 0.2, alpha = 0.3) +
  theme_minimal() +
  labs(title = "Distribución de Severidad por Tratamiento (NGRDI Grid 64x64)",
       subtitle = "Análisis basado en Swin-Windowing para eliminación de ruido",
       y = "Severidad Real (%)", x = "Tratamiento / Cepa") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# B. Gráfico de Barras: Proporción de Severidad
p2 <- ggplot(df_clean, aes(x = Cepa, fill = Nivel_Severidad)) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = scales::percent) +
  theme_minimal() +
  labs(title = "Proporción de Niveles de Infección",
       y = "Porcentaje del Total", x = "Tratamiento") +
  scale_fill_brewer(palette = "Reds")

# 4. EXPORTAR RESULTADOS
output_dir <- "c:/Users/juanv/Desktop/Analisis mango/RUNS_SPECTRAL_FINAL/REPORTES_PLIMAN"
if (!dir.exists(output_dir)) dir.create(output_dir)

ggsave(file.path(output_dir, "boxplot_severidad.png"), p1, width = 10, height = 6)
ggsave(file.path(output_dir, "proporcion_infeccion.png"), p2, width = 10, height = 6)

# 5. RESUMEN ESTADÍSTICO (Estilo Pliman)
summary_stats <- df_clean %>%
  group_by(Cepa) %>%
  summarise(
    Media = mean(Severidad_..),
    SD = sd(Severidad_..),
    Max = max(Severidad_..),
    N_Frutos = n()
  )

write.csv(summary_stats, file.path(output_dir, "resumen_estadistico_pliman.csv"), row.names = FALSE)

print("✅ Análisis con PLIMAN completado. Revisa la carpeta REPORTES_PLIMAN.")
