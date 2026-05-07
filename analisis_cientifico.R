# Script de Análisis Científico Robusto - Mango Anthracnose
library(ggplot2)
library(dplyr)
library(tidyr)
library(gridExtra)

# Cargar datos
df <- read.csv("C:/Users/juanv/Desktop/Analisis mango/DATOS_LIMPIOS_PARA_R.csv")

# Asegurar orden de meses
df$Mes <- factor(df$Mes, levels=c("INICIAL", "ABRIL", "MAYO", "JULIO", "AGOSTO", "SEPTIEMBRE", "OCTUBRE"))

# Directorio de salida para gráficas
output_dir <- "C:/Users/juanv/Desktop/Analisis mango/REPORTES_CIENTIFICOS"
if (!dir.exists(output_dir)) dir.create(output_dir)

# 1. INCIDENCIA POR CEPA
# Incidencia = (Frutos con afectación > 0 / Total frutos) * 100
incidencia_df <- df %>%
  group_by(Cepa) %>%
  summarise(
    Total = n(),
    Afectados = sum(Afectado_. > 0.5),
    Incidencia = (Afectados / Total) * 100
  )

g1 <- ggplot(incidencia_df, aes(x=Cepa, y=Incidencia, fill=Cepa)) +
  geom_bar(stat="identity", color="black") +
  theme_minimal() +
  labs(title="Incidencia de Antracnosis por Cepa", y="Incidencia (%)", x="Tratamiento") +
  scale_fill_brewer(palette="Set2")
ggsave(file.path(output_dir, "01_Incidencia_Cepa.png"), g1, width=8, height=6)

# 2. SEVERIDAD POR CEPA (BOXPLOT)
g2 <- ggplot(df, aes(x=Cepa, y=Afectado_., fill=Cepa)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title="Distribución de Severidad por Tratamiento", y="Área Afectada (%)", x="Cepa") +
  scale_fill_brewer(palette="Pastel1")
ggsave(file.path(output_dir, "02_Boxplot_Severidad.png"), g2, width=8, height=6)

# 3. EFECTO DE LA PODA
g3 <- ggplot(df, aes(x=Poda, y=Afectado_., fill=Poda)) +
  geom_violin(trim=FALSE) +
  geom_boxplot(width=0.1, fill="white") +
  theme_minimal() +
  labs(title="Impacto de la Poda en la Severidad", y="Área Afectada (%)") +
  scale_fill_manual(values=c("#E41A1C", "#377EB8", "#4DAF4A"))
ggsave(file.path(output_dir, "03_Efecto_Poda.png"), g3, width=8, height=6)

# 4. CURVA DE PROGRESO DE LA ENFERMEDAD (POR MES)
progreso_df <- df %>%
  group_by(Mes, Cepa) %>%
  summarise(Severidad_Media = mean(Afectado_.))

g4 <- ggplot(progreso_df, aes(x=Mes, y=Severidad_Media, group=Cepa, color=Cepa)) +
  geom_line(size=1.2) +
  geom_point(size=3) +
  theme_minimal() +
  labs(title="Progreso Epidemiológico de Antracnosis", y="Severidad Media (%)") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggsave(file.path(output_dir, "04_Curva_Progreso.png"), g4, width=10, height=6)

# 5. HISTOGRAMA DE DISTRIBUCIÓN
g5 <- ggplot(df, aes(x=Afectado_.)) +
  geom_histogram(binwidth=2, fill="#69b3a2", color="#e9ecef", alpha=0.9) +
  theme_minimal() +
  labs(title="Distribución de la Severidad en la Población", x="Área Afectada (%)", y="Frecuencia")
ggsave(file.path(output_dir, "05_Histograma_Severidad.png"), g5, width=8, height=6)

# 6. INTERACCIÓN CEPA X PODA
g6 <- ggplot(df, aes(x=Cepa, y=Afectado_., fill=Poda)) +
  geom_boxplot() +
  facet_wrap(~Poda) +
  theme_minimal() +
  labs(title="Interacción Cepa vs Poda", y="Severidad (%)")
ggsave(file.path(output_dir, "06_Interaccion_Cepa_Poda.png"), g6, width=12, height=6)

# 7. CORRELACIÓN SALUD VS AFECTADO
g7 <- ggplot(df, aes(x=Salud_., y=Afectado_.)) +
  geom_point(alpha=0.3, color="darkgreen") +
  geom_smooth(method="lm", color="red") +
  theme_minimal() +
  labs(title="Correlación Inversa: Salud vs Afectación", x="Tejido Sano (%)", y="Tejido Afectado (%)")
ggsave(file.path(output_dir, "07_Correlacion.png"), g7, width=8, height=6)

# 8. HEATMAP DE RIESGO (CEPA VS MES)
g8 <- ggplot(progreso_df, aes(x=Mes, y=Cepa, fill=Severidad_Media)) +
  geom_tile() +
  scale_fill_gradient(low="white", high="red") +
  theme_minimal() +
  labs(title="Mapa de Calor de Riesgo de Antracnosis", fill="Severidad (%)")
ggsave(file.path(output_dir, "08_Heatmap_Riesgo.png"), g8, width=10, height=6)

# 9. COMPARATIVA DE STATUS (DETECTADO VS NO DETECTADO)
status_df <- df %>% group_by(Status) %>% summarise(Count = n())
g9 <- ggplot(status_df, aes(x="", y=Count, fill=Status)) +
  geom_bar(stat="identity", width=1) +
  coord_polar("y", start=0) +
  theme_void() +
  labs(title="Efectividad de la Detección YOLO")
ggsave(file.path(output_dir, "09_Efectividad_YOLO.png"), g9, width=6, height=6)

# 10. DENSITY PLOT POR CEPA
g10 <- ggplot(df, aes(x=Afectado_., fill=Cepa)) +
  geom_density(alpha=0.4) +
  theme_minimal() +
  labs(title="Densidad de la Enfermedad por Tratamiento", x="Severidad (%)")
ggsave(file.path(output_dir, "10_Densidad_Cepa.png"), g10, width=8, height=6)

# 11. PANEL CONSOLIDADO
# (Solo para visualización rápida)
# grid.arrange(g1, g2, g4, g8, ncol=2)

print("[!] Análisis en R finalizado. Gráficas guardadas en REPORTES_CIENTIFICOS")
