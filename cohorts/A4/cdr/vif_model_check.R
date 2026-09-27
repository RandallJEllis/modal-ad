library(arrow)
library(tidyverse)
library(pec)
library(timeROC)
library(pROC)
library(yardstick)
library(dplyr)
library(ggplot2)
library(survival)
library(survminer)
library(riskRegression)
library(this.path)
library(car)
library(polycor)

setwd(dirname(this.path()))

source("plot_figures.R")
source("metrics.R")

load_path <- "../../../tidy_data/A4/"

# Load results
models_list <- qs::qread(paste0(load_path, "fitted_models_id.qs"))
val_df_l <- qs::qread(paste0(load_path, "val_df_l_id.qs"))
train_df_l <- qs::qread(paste0(load_path, "train_df_l_id.qs"))
metrics_list <- qs::qread(paste0(load_path, "metrics_id.qs"))

# Calculate VIF for all 5 folds and summarize
model_name <- "ptau_centiloids_demographics_lancet"

vif_all_folds <- lapply(1:5, function(fold) {
  model <- models_list[[model_name]][[paste0("fold_", fold)]]
  vif_res <- car::vif(model)

  # Handle both simple VIF (vector) and GVIF (matrix with interactions)
  if (is.matrix(vif_res)) {
    # For models with interactions: use GVIF^(1/(2*Df)) which is comparable to VIF
    data.frame(
      variable = rownames(vif_res),
      vif = vif_res[, "GVIF^(1/(2*Df))"],
      fold = fold
    )
  } else {
    data.frame(
      variable = names(vif_res),
      vif = vif_res,
      fold = fold
    )
  }
})

vif_df <- do.call(rbind, vif_all_folds)

# Summarize across folds
vif_summary <- vif_df %>%
  group_by(variable) %>%
  summarise(
    mean_vif = mean(vif),
    sd_vif = sd(vif),
    min_vif = min(vif),
    max_vif = max(vif),
    .groups = "drop"
  ) %>%
  arrange(desc(mean_vif))

print(vif_summary)

# Flag any variables with VIF > 5 (or > 10, depending on your threshold)
high_vif <- vif_summary %>% filter(mean_vif > 5)
if (nrow(high_vif) > 0) {
  print("Variables with concerning VIF (>5):")
  print(high_vif)
}

vif_summary %>%
  kbl(digits = 3, caption = "VIF Summary") %>%
  kable_styling(bootstrap_options = c("striped", "hover")) %>%
  save_kable(paste0(load_path, "vif_summary_table.html"))
