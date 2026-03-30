library(tidyverse)
library(magrittr)
library(ggpubr)

source("Project1/A3/source_files_for_R.R")
df_raw = read_csv("resources/adni_pet_image_analysis/AMY/Diagnostics_merged_amy_6mm.csv")
amy_threshold = 1.10

cols_lower <- tolower(cols_to_convert)

df_raw %<>% # head() %>% 
  mutate(
    across(all_of(cols_to_convert), 
           ~ as.integer(. >= amy_threshold),
           .names ="{.col}_positivity"), 
    
    DIFF = as.integer(stringr::str_split_i(DATE_DIFF, " ", 1)), 
    DIAGNOSIS = factor(DIAGNOSIS_NAME, levels = c("CN", "MCI", "AD"))
  ) %>% 
  rename_with(tolower)


df_raw %>% 
  filter(
    tracer == "FBP", 
    qc_flag >=2, 
    diff < 180) -> df


########################## 
# Sanity Checks 

results <- expand.grid(
  region = cols_lower,
  comparison = c("CN_vs_MCI", "MCI_vs_AD", "CN_vs_AD"),
  stringsAsFactors = FALSE
) %>%
  rowwise() %>%
  mutate(p_value = {
    pair <- strsplit(comparison, "_vs_")[[1]]
    g1 <- df[[region]][df$diagnosis == pair[1]]
    g2 <- df[[region]][df$diagnosis == pair[2]]
    t.test(g1, g2)$p.value
  }) %>%
  ungroup() %>%
  pivot_wider(names_from = comparison, values_from = p_value)

results %>% 
  mutate(
    across(contains("_vs_"), ~as.character(. < 0.05), .names = "{col}_significance")
    ) %>% 
  select(region, contains("significance")) %>% 
  view()

############### 
### PLOST to PDF

# flatten it out
df_long <- df %>%
  select(diagnosis, all_of(cols_lower)) %>%
  pivot_longer(cols = all_of(cols_lower), names_to = "region", values_to = "value")


pdf("temp_db/brain_region_boxplots.pdf", width = 6, height = 5)

for (reg in cols_lower) {
  p <- df_long %>%
    filter(region == reg) %>%
    ggplot(aes(x = diagnosis, y = value, fill = diagnosis)) +
    geom_boxplot(outlier.size = 0.8, alpha = 0.8) +
    stat_compare_means(
      comparisons = comparisons,
      method = "t.test", # pairwise comparison for CN-MCI, MCI-AD, CN-AD       
      label = "p.signif" 
    ) +
    # stat_compare_means(
    #   method = "anova", # do we even need a overall ANOVA p at top
    #   label.y.npc = 0.95,
    #   label = "p.format"
    # ) +
    scale_fill_manual(values = c(CN = "#4CAF50", MCI = "#FF9800", AD = "#F44336")) +
    labs(title = reg, x = "Diagnosis", y = "Value") +
    theme_bw() +
    theme(legend.position = "none")
  
  print(p)
}

dev.off()


######################################
# density pltos for CN vs AD

comparison <- "CN_vs_AD"
pair <- strsplit(comparison, "_vs_")[[1]]

pdf("density_plots_per_region_CN_MCI.pdf", width = 8, height = 5)

for (region in cols_lower) {
  g1 <- df[[region]][df$diagnosis == pair[1]]
  g2 <- df[[region]][df$diagnosis == pair[2]]
  
  p_val <- t.test(g1, g2)$p.value
  d <- (mean(g2) - mean(g1)) / sqrt(((length(g1)-1)*var(g1) + (length(g2)-1)*var(g2)) / (length(g1)+length(g2)-2))
  
  plot_df <- data.frame(
    value = c(g1, g2),
    group = c(rep(pair[1], length(g1)), rep(pair[2], length(g2)))
  )
  
  p <- ggplot(plot_df, aes(x = value, fill = group)) +
    geom_density(alpha = 0.4) +
    labs(
      title  = region,
      subtitle = sprintf("p = %.2e  |  Cohen's d = %.3f", p_val, d),
      x = "SUVR", y = "Density"
    ) +
    theme_minimal()
  
  print(p)
}

dev.off()


