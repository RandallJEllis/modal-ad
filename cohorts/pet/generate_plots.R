library(arrow)
library(ggplot2)
library(xtable)
library(tidyverse)
library(this.path)
library(tidyr)
library(dplyr)

setwd(dirname(this.path()))

source("../../survival/time2event/plot_figures.R")
source("../../survival/time2event/metrics.R")

# Load fonts
library(extrafont)
extrafont::loadfonts()
font_import()
loadfonts(device = "postscript")

base_path <- "../../pet_all_cohorts/tidy_data/"

# set parameters of experiment type to identify path
ad_outcome <- TRUE
if (ad_outcome) {
  ad_outcome_path <- "ad_outcome/"
} else {
  ad_outcome_path <- ""
}

age_cutoff <- 65
if (!is.null(age_cutoff)) {
  age_cutoff_path <- paste0("age_", age_cutoff, "_cutoff/")
} else {
  age_cutoff_path <- ""
}

cross_cohort_validation <- FALSE
if (cross_cohort_validation) {
  cross_cohort_validation_path <- "cross_cohort_validation/"
} else {
  cross_cohort_validation_path <- ""
}

exp_path <- paste0(base_path, age_cutoff_path, ad_outcome_path, cross_cohort_validation_path)

# for (ad_outcome in c(FALSE, TRUE)) {
#   for (age_cutoff in c(NULL,
#                       65)) {

#     if (ad_outcome) {
#       ad_outcome_path <- "ad_outcome/"
#     } else {
#       ad_outcome_path <- ""
#     }

#     if (!is.null(age_cutoff)) {
#       age_cutoff_path <- paste0("age_", age_cutoff, "_cutoff/")
#     } else {
#       age_cutoff_path <- ""
#     }

#     main_path <- paste0("../../pet_all_cohorts/tidy_data/", age_cutoff_path, ad_outcome_path)
#     print(main_path)

models_list <- qs::qread(paste0(exp_path, "fitted_models_all.qs"))
metrics_list <- qs::qread(paste0(exp_path, "metrics_all.qs"))
train_df_l <- qs::qread(paste0(exp_path, "train_df_all.qs"))
val_df_l <- qs::qread(paste0(exp_path, "val_df_all.qs"))

model_names <- c(
  "demographics",
  "centiloids_demographics",
  "centiloids",
  "demographics_no_apoe",
  "centiloids_demographics_no_apoe"
)
width <- 8
height <- 6
dpi <- 300
eval_times <- seq(2, 10)

save_all_figures(
  modality = "PET", model_names, models_list, metrics_list, train_df_l, val_df_l,
  eval_times, width, height, dpi, exp_path
)
#   }
# }

########################################################
# REVISION - Brier decomposition
brier_model_names <- c(
  "demographics",
  "centiloids_demographics",
  "centiloids",
  "demographics_no_apoe",
  "centiloids_demographics_no_apoe"
)

folds <- 1:5
eval_times <- seq(2, 10) # Adjust as needed

brier_decomp_df <- data.frame(
  model = character(),
  fold = integer(),
  time = numeric(),
  brier_score = numeric(),
  reliability = numeric(),
  resolution = numeric(),
  uncertainty = numeric(),
  stringsAsFactors = FALSE
)

for (model_name in brier_model_names) {
  for (fold in folds) {
    model <- models_list[[model_name]][[paste0("fold_", fold)]]
    val_data <- val_df_l[[paste0("fold_", fold)]]

    for (t in eval_times) {
      res <- tryCatch(
        {
          calculate_brier_at_time(model, val_data, t, decomp = TRUE)
        },
        error = function(e) {
          warning(sprintf(
            "Error for model %s, fold %d, time %f: %s",
            model_name, fold, t, e$message
          ))
          return(NULL)
        }
      )

      if (!is.null(res) && !is.na(res$brier)) {
        brier_decomp_df <- rbind(brier_decomp_df, data.frame(
          model = model_name,
          fold = fold,
          time = t,
          brier_score = res$brier,
          reliability = res$reliability,
          resolution = res$resolution,
          uncertainty = res$uncertainty
        ))
      }
    }
  }
}

print(brier_decomp_df)

brier_summary <- brier_decomp_df %>%
  group_by(model, time) %>%
  summarise(
    mean_metric = mean(brier_score, na.rm = TRUE),
    sd_metric = sd(brier_score, na.rm = TRUE),
    ymin = pmax(mean_metric - sd_metric, 0),
    ymax = pmin(mean_metric + sd_metric, 1),
    .groups = "drop"
  )
brier_summary$model <- factor(brier_summary$model, levels = brier_model_names)

reliability_summary <- brier_decomp_df %>%
  group_by(model, time) %>%
  summarise(
    mean_metric = mean(reliability, na.rm = TRUE),
    sd_metric = sd(reliability, na.rm = TRUE),
    ymin = pmax(mean_metric - sd_metric, 0),
    ymax = mean_metric + sd_metric,
    .groups = "drop"
  )
reliability_summary$model <- factor(reliability_summary$model, levels = brier_model_names)

resolution_summary <- brier_decomp_df %>%
  group_by(model, time) %>%
  summarise(
    mean_metric = mean(resolution, na.rm = TRUE),
    sd_metric = sd(resolution, na.rm = TRUE),
    ymin = pmax(mean_metric - sd_metric, 0),
    ymax = mean_metric + sd_metric,
    .groups = "drop"
  )
resolution_summary$model <- factor(resolution_summary$model, levels = brier_model_names)

uncertainty_summary <- brier_decomp_df %>%
  group_by(model, time) %>%
  summarise(
    mean_metric = mean(uncertainty, na.rm = TRUE),
    sd_metric = sd(uncertainty, na.rm = TRUE),
    ymin = pmax(mean_metric - sd_metric, 0),
    ymax = mean_metric + sd_metric,
    .groups = "drop"
  )
uncertainty_summary$model <- factor(uncertainty_summary$model, levels = brier_model_names)

if (!is.null(eval_times)) {
  brier_summary <- brier_summary %>% filter(time %in% eval_times)
  reliability_summary <- reliability_summary %>% filter(time %in% eval_times)
  resolution_summary <- resolution_summary %>% filter(time %in% eval_times)
  uncertainty_summary <- uncertainty_summary %>% filter(time %in% eval_times)
}

# Brier score plot
brier_plot <- td_plot(brier_summary,
  model_names = brier_model_names,
  metric = "brier",
  all_models = FALSE,
  # eval_times = eval_times
)

# Reliability plot (lower is better - measures calibration error)
reliability_plot <- td_plot(reliability_summary,
  model_names = brier_model_names,
  metric = "brier",
  all_models = FALSE,
  # eval_times = eval_times
) + labs(title = "Reliability Over Time", y = "Reliability")

# Resolution plot (higher is better - measures discrimination)
resolution_plot <- td_plot(resolution_summary,
  model_names = brier_model_names,
  metric = "brier",
  all_models = FALSE,
  # eval_times = eval_times
) + labs(title = "Resolution Over Time", y = "Resolution")

# Uncertainty plot (baseline variability)
uncertainty_plot <- td_plot(uncertainty_summary,
  model_names = brier_model_names,
  metric = "brier",
  all_models = FALSE,
  # eval_times = eval_times
) + labs(title = "Uncertainty Over Time", y = "Uncertainty")

# Combined decomposition plot
decomp_combined <- ((brier_plot | reliability_plot) / (resolution_plot | uncertainty_plot)) +
  plot_layout(guides = "collect", heights = c(1, 1)) +
  plot_annotation(tag_levels = "A")


print(decomp_combined)

ggsave("../../pet_all_cohorts/figures/brier_decomp.png", dpi = 400, width = 14, height = 8)

########################################################
# REVISION - Net Reclassification Improvement
folds <- 1:5
eval_times <- seq(2, 10) # Adjust as needed

nri_res <- list()
for (fold in folds) {
  nri_res[[paste0("fold_", fold)]] <- net_reclassification_improvement(
    models_list$demographics[[paste0("fold_", fold)]],
    models_list$centiloids[[paste0("fold_", fold)]],
    val_df_l[[paste0("fold_", fold)]],
    eval_times
  )
}

nri_df <- bind_rows(nri_res, .id = "fold")

# 1. Transform your results into long format and summarize across folds
plot_data <- nri_df %>%
  pivot_longer(
    cols = c(nri, nri_event, nri_nonevent),
    names_to = "Component",
    values_to = "Value"
  ) %>%
  # Average over folds within each time point
  group_by(time, Component) %>%
  summarise(
    mean_value = mean(Value, na.rm = TRUE),
    sd_value = sd(Value, na.rm = TRUE),
    se_value = sd(Value, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  ) %>%
  # Clean up component names for display
  mutate(Component = case_when(
    Component == "nri" ~ "NRI",
    Component == "nri_event" ~ "NRI+ (Cases)",
    Component == "nri_nonevent" ~ "NRI- (Controls)"
  ))

# 2. Create the plot with error ribbons
nri_over_time <- ggplot(plot_data, aes(x = time, y = mean_value, color = Component, fill = Component, group = Component)) +
  geom_ribbon(aes(ymin = mean_value - sd_value, ymax = mean_value + sd_value), alpha = 0.2, color = NA) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 3) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", alpha = 0.5) +
  labs(
    title = "Net Reclassification Improvement Over Time",
    subtitle = "Demographics vs. Amyloid-PET",
    x = "Follow-up Time (Years)",
    y = "NRI Estimate (Mean ± SD)",
    color = "Metric",
    fill = "Metric"
  ) +
  theme_bw(base_size = 16) +
  scale_color_brewer(palette = "Set1") +
  scale_fill_brewer(palette = "Set1") +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold", size = 22),
    plot.subtitle = element_text(size = 20),
    axis.title = element_text(face = "bold", size = 20),
    axis.text = element_text(size = 18),
    legend.title = element_text(size = 18),
    legend.text = element_text(size = 16),
    panel.background = element_rect(fill = "white"),
    plot.background = element_rect(fill = "white")
  )

print(nri_over_time)

ggsave("../../pet_all_cohorts/figures/nri_over_time.png", dpi = 400, width = 8, height = 8)

########################################################
# REVISION: Decision Curve Analysis
# Compare multiple models
# Calculate DCA across all folds
dca_all <- decision_curve_all_folds(
  models_list = models_list,
  model_names = c("demographics", "centiloids"),
  val_df_l = val_df_l,
  folds = 1:5,
  times = seq(2, 10)
)

# Plot all time points with SD ribbons
dca_data <- data_decision_curves_all_times(dca_all)

ref_data <- dca_data$ref_data
summary_df <- dca_data$summary_df

# Create faceted plot
p <- ggplot() +
  # Reference lines
  geom_line(
    data = ref_data,
    aes(x = threshold, y = net_benefit, linetype = strategy),
    color = "gray40", linewidth = 0.6
  ) +
  # Model lines with ribbon for SD
  geom_ribbon(
    data = summary_df,
    aes(
      x = threshold,
      ymin = mean_net_benefit - sd_net_benefit,
      ymax = mean_net_benefit + sd_net_benefit,
      fill = model
    ),
    alpha = 0.2
  ) +
  geom_line(
    data = summary_df,
    aes(x = threshold, y = mean_net_benefit, color = model),
    linewidth = 1
  ) +
  # Facet by time
  facet_wrap(~time,
    ncol = 3,
    labeller = labeller(time = function(x) paste("Year", x))
  ) +
  # Formatting
  coord_cartesian(ylim = c(
    -0.05,
    max(summary_df$mean_net_benefit, na.rm = TRUE) * 1.2
  )) +
  scale_x_continuous(labels = scales::label_number()) +
  labs(
    title = "Decision Curve Analysis Over Time",
    subtitle = "Demographics vs. Amyloid-PET",
    x = "Threshold Probability",
    y = "Net Benefit",
    color = "Model",
    fill = "Model",
    linetype = "Reference"
  ) +
  theme_bw(base_size = 18) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold", size = 26),
    plot.subtitle = element_text(size = 22),
    axis.title = element_text(face = "bold", size = 22),
    axis.text = element_text(size = 18),
    strip.text = element_text(face = "bold", size = 20),
    legend.title = element_text(size = 20),
    legend.text = element_text(size = 18)
  ) +
  guides(
    color = guide_legend(nrow = 2),
    fill = guide_legend(nrow = 2)
  )

if (!is.null(model_colors)) {
  p <- p +
    scale_color_manual(values = model_colors) +
    scale_fill_manual(values = model_colors)
}

print(p)
ggsave("../../pet_all_cohorts/figures/dca_all_times.png", dpi = 400, width = 14, height = 10)

########################################################
# Other stats and tables
age_cutoff <- 65
model_names <- c(
  "demographics",
  "centiloids_demographics",
  "centiloids",
  "demographics_no_apoe",
  "centiloids_demographics_no_apoe"
)

ad_outcome <- TRUE
if (ad_outcome) {
  ad_outcome_path <- "ad_outcome/"
} else {
  ad_outcome_path <- ""
}

if (!is.null(age_cutoff)) {
  age_cutoff_path <- paste0("age_", age_cutoff, "_cutoff/")
} else {
  age_cutoff_path <- ""
}
main_path <- paste0("../../pet_all_cohorts/tidy_data/", age_cutoff_path, ad_outcome_path)
models_list <- qs::qread(paste0(main_path, "fitted_models_all.qs"))
metrics_list <- qs::qread(paste0(main_path, "metrics_all.qs"))
train_df_l <- qs::qread(paste0(main_path, "train_df_all.qs"))
val_df_l <- qs::qread(paste0(main_path, "val_df_all.qs"))


options(pillar.width = Inf)
########################################################
# Sensitivity, Specificity, PPV, NPV
model1 <- "demographics"
model2 <- "centiloids_demographics"
range_sespppvnnv(model1, model2, main_path)

# work with the data
df_sespppvnpv <- read_parquet(paste0(main_path, "sespppvnpv_summary.parquet"))

# filter for the two models and years seq(3,8)
df_sespppvnpv <- df_sespppvnpv %>%
  filter(model %in% c(model1, model2)) %>%
  filter(time %in% seq(2, 10)) %>%
  print()

df_sespppvnpv

########################################################
# statistics of PET+demo+apoe
# print ranges of coefficients for centiloids_demographics
model_name <- "centiloids_demographics"
coef_list <- lapply(models_list[[model_name]], function(x) {
  coef_summary <- summary(x)$coefficients
  data.frame(
    term = rownames(coef_summary),
    estimate = coef_summary[, "coef"],
    std_error = coef_summary[, "se(coef)"],
    z_value = coef_summary[, "z"],
    p_value = coef_summary[, "Pr(>|z|)"]
  )
})
coef_df <- do.call(rbind, coef_list)

# compute exponentiated coefficients per fit
coef_df$exp_estimate <- exp(coef_df$estimate)

# For each predictor (term) produce min and max for requested stats and format as ranges
coef_summary <- coef_df %>%
  group_by(term) %>%
  summarise(
    min_p = min(p_value, na.rm = TRUE),
    max_p = max(p_value, na.rm = TRUE),
    min_coef = min(estimate, na.rm = TRUE),
    max_coef = max(estimate, na.rm = TRUE),
    min_exp_coef = min(exp_estimate, na.rm = TRUE),
    max_exp_coef = max(exp_estimate, na.rm = TRUE),
    min_se = min(std_error, na.rm = TRUE),
    max_se = max(std_error, na.rm = TRUE),
    min_z = min(z_value, na.rm = TRUE),
    max_z = max(z_value, na.rm = TRUE)
  ) %>%
  ungroup() %>%
  mutate(
    # format p-values with 3 significant digits, others with 3 decimal places
    `p-value` = paste0(formatC(min_p, format = "g", digits = 3), " - ", formatC(max_p, format = "g", digits = 3)),
    coefficient = paste0(formatC(min_coef, format = "f", digits = 3), " - ", formatC(max_coef, format = "f", digits = 3)),
    `exp(coefficient)` = paste0(formatC(min_exp_coef, format = "f", digits = 3), " - ", formatC(max_exp_coef, format = "f", digits = 3)),
    `se(coefficient)` = paste0(formatC(min_se, format = "f", digits = 3), " - ", formatC(max_se, format = "f", digits = 3)),
    `z-value` = paste0(formatC(min_z, format = "f", digits = 3), " - ", formatC(max_z, format = "f", digits = 3))
  ) %>%
  arrange(min_p) %>%
  select(term, `p-value`, coefficient, `exp(coefficient)`, `se(coefficient)`, `z-value`) %>%
  print()

print(coef_summary)
print(xtable(coef_summary), type = "latex", sanitize.text.function = function(x) x)

########################################################
# Bayes Information Criterion
for (model_name in names(models_list)) {
  for (fold in 1:5) {
    model <- models_list[[model_name]][[paste0("fold_", fold)]]
    print(paste0("Model: ", model_name, " Fold: ", fold, " BIC: ", BIC(model)))
  }
}

########################################################
# Fig S5 and Table S7
# Calculate p-values comparing AUCs between two models at each time point
# First combine timeROC objects from each fold for each model
demo_trocs <- pull_trocs(metrics_list, "demographics")
centiloids_demo_trocs <- pull_trocs(metrics_list, "centiloids_demographics")

# demo vs centiloids+demo
pvals_compare_trocs <- compare_tvaurocs(
  demo_trocs,
  centiloids_demo_trocs
)
# Table S1 - pivot table of p-values where each row is a fold and each column is a time point
print_pvalue_latex_table(pvals_compare_trocs$all_results)

# Fig S1 - Histogram of p-values, bin size 0.05
ggsave(paste0(main_path, "pvalue_histogram_PET_Demo_vs_Demo.pdf"),
  plot = histogram_pvals(pvals_compare_trocs$all_results),
  width = 8,
  height = 6,
  dpi = 300
)
# how many p-values are less than 0.05? out of how many total p-values?
sum(pvals_compare_trocs$all_results$p_value < 0.05) /
  length(pvals_compare_trocs$all_results$p_value)

centiloids_stats <- print_model_stats(models_list, "centiloids", return_table = TRUE)
# print centiloids_stats as latex table
print(xtable(centiloids_stats), type = "latex", sanitize.text.function = function(x) x)

# print out all p-values
print("Summary of AUC differences and p-values by time point:")
print(range(pvals_compare_trocs$all_results$p_value))
print(mean(pvals_compare_trocs$all_results$p_value))
print(sd(pvals_compare_trocs$all_results$p_value))
print(median(pvals_compare_trocs$all_results$p_value))

# Create detailed results table
results_table <- pvals_compare_trocs$all_results
write.csv(results_table, paste0(main_path, "auc_comparison_results_demo_lancet_vs_ptau_demo_lancet.csv"),
  row.names = FALSE
)

# # Boxplots at each time point of AUC differences for pTau217+Demographics+Lancet vs Demographics+Lancet
# library(ggplot2)
# auc_plot <- ggplot(pvals_compare_trocs$all_results,
#        aes(x = factor(time), y = auc_diff)) +
#   geom_boxplot(fill = "#009292", alpha = 0.8) +
#   geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
#   labs(
#     title = "AUC Differences Between Models\n(pTau-217+Demographics+Lancet vs Demographics+Lancet)",
#     x = "Time (years)",
#     y = "AUC Difference"
#   ) +
#   theme_minimal() +
#   theme(
#     plot.title = element_text(hjust = 0.5, size = 12),
#     axis.text = element_text(size = 10),
#     axis.title = element_text(size = 11)
#   )

# print(auc_plot)
# ggsave(
#   "../../tidy_data/A4/auc_differences_boxplot.pdf",
#   plot = auc_plot,
#   width = 8,
#   height = 6,
#   dpi = 300
# )

########################################################
# Table S3 - reshape auc_summary to wide format
auc_summary <- read_parquet(paste0(main_path, "auc_summary.parquet"))
print_auc_latex_table(auc_summary)

rr <- pull_roc_summary(model_names, seq(3, 7))
pauc_res <- calc_pAUC(rr, model_names, threshold = 0.25)
pauc_res$Model <- model_labels[pauc_res$Model]
pauc_res

# latex table of results_table_wide with 4 decimal places
xtable_obj <- xtable(pauc_res)
digits(xtable_obj) <- c(0, rep(4, ncol(pauc_res))) # Set digits for each column
print(xtable_obj, type = "latex", sanitize.text.function = function(x) x) # Don't escape LaTeX commands


########################################################
# Generate and save results for each metric
metrics_to_collect <- c("auc", "brier", "concordance")

for (metric in metrics_to_collect) {
  results <- collate_metric(metrics_list, metric)
  write_csv(
    results,
    paste0(main_path, "results_", metric, "_all_models.csv")
  )
}

# average brier score across folds within each model and timepoint
brier_scores <- read_csv(paste0(main_path, "results_brier_all_models.csv"))
brier_scores_avg <- brier_scores %>%
  group_by(model, time) %>%
  summarise(mean_brier = mean(metric))
# print differences between centiloids+demo and demo at each timepoint
brier_scores_avg %>%
  filter(model == "centiloids_demographics" | model == "demographics") %>%
  select(time, mean_brier) %>%
  pivot_wider(names_from = model, values_from = mean_brier) %>%
  mutate(diff = `centiloids_demographics` - `demographics`) %>%
  print()

########################################################
# Calculate calibration data
cal_data_avg <- calculate_calibration_data(models_list, val_df_l)

# Create calibration plots
plots <- calibration_plots(cal_data_avg, seq(3, 8), model_colors)
print(plots)

# Save plots
ggsave("../../tidy_data/A4/final_calibration_plots.pdf",
  plot = plots,
  width = 8,
  height = 6,
  dpi = 300
)

########################################################
# Decision curve analysis
# Collect predictions and create DCA data
dca_data_all <- list()
models_to_analyze <- c(
  "demographics",
  "demographics_no_apoe",
  "demographics_lancet",
  "ptau",
  "ptau_demographics_lancet"
)

# Collect predictions for each time point and model
for (t in seq(3, 8)) {
  for (fold in 0:4) {
    for (model_name in models_to_analyze) {
      model <- overwrite_na_coef_to_zero(
        models_list[[model_name]][[paste0("fold_", fold + 1)]]
      )

      pred_probs <- 1 - pec::predictSurvProb(
        model,
        newdata = val_df_l[[paste0("fold_", fold + 1, "_", model_name)]],
        times = t
      )

      val_data <- val_df_l[[paste0("fold_", fold + 1, "_", model_name)]]
      dca_data_all[[paste0(
        "t", t, "_fold",
        fold, "_", model_name
      )]] <- data.frame(
        fold = fold,
        time = t,
        model = model_name,
        tstop = val_data$tstop,
        event = val_data$event,
        pred_prob = pred_probs
      )
    }
  }
}

# Combine all DCA data
all_dca_data <- do.call(rbind, dca_data_all)

# Create and save DCA plots
dca_plots <- dca_plots(all_dca_data)
print(dca_plots)

ggsave("../../tidy_data/A4/final_DCA_Over_Time.pdf",
  plot = dca_plots,
  width = 8,
  height = 6,
  dpi = 300
)


########################################################
# Clinical risk reclassification
library(nricens) # For NRI calculations with survival data

find_events_within_horizon <- function(data, horizon, newdata) {
  # Create a mapping from ID to event status within horizon
  event_summary <- data %>%
    group_by(id) %>%
    summarize(
      event_occurred = any(event == 1),
      event_time = ifelse(event_occurred, min(tstop[event == 1]), Inf),
      within_horizon = event_occurred & event_time <= horizon
    )

  # Match the event status to the IDs in newdata
  event_status <- numeric(nrow(newdata))
  for (i in 1:nrow(newdata)) {
    id_match <- which(event_summary$id == newdata$id[i])
    if (length(id_match) > 0) {
      event_status[i] <- as.numeric(event_summary$within_horizon[id_match])
    }
  }

  return(event_status)
}

model1 <- models_list$demographics_lancet$fold_1
model2 <- models_list$ptau_demographics_lancet$fold_1
df <- train_df_l$fold_1_demographics_lancet
all.equal(
  train_df_l$fold_1_demographics_lancet,
  train_df_l$fold_1_ptau_demographics_lancet
) # must be TRUE

# Define prediction time horizon
horizon <- 5 # years

# Create a baseline dataset for prediction (use data at a specific time point)
newdata <- df[df$tstart < 5, ] # or another relevant baseline

# Get predicted survival probabilities
pred_surv1 <- 1 - summary(survfit(model1, newdata = newdata),
  times = horizon
)$surv
pred_surv2 <- 1 - summary(survfit(model2, newdata = newdata),
  times = horizon
)$surv

# Define risk categories (modify based on your clinical context)
risk_cats <- c(0, 0.05, 0.10, 0.20, 1)
risk_labels <- c("0-5%", "5-10%", "10-20%", ">20%")

# Categorize predicted risks
risk_cat1 <- cut(pred_surv1, breaks = risk_cats, labels = risk_labels, include.lowest = TRUE)
risk_cat2 <- cut(pred_surv2, breaks = risk_cats, labels = risk_labels, include.lowest = TRUE)

# Create reclassification table
reclass_table <- table(risk_cat1, risk_cat2)
print(reclass_table)

# Calculate percentage reclassified in each risk category
percent_reclass <- numeric(length(risk_labels))
names(percent_reclass) <- risk_labels

for (i in 1:length(risk_labels)) {
  cat <- risk_labels[i]
  n_total <- sum(reclass_table[i, ])
  n_reclass <- n_total - reclass_table[i, i]
  percent_reclass[i] <- 100 * n_reclass / n_total
}
print(percent_reclass)

# Create a dataframe with predicted risks and actual outcomes
reclass_df <- data.frame(
  id = newdata$id, # Use actual IDs from newdata
  risk_cat1 = risk_cat1,
  risk_cat2 = risk_cat2,
  pred_risk1 = pred_surv1,
  pred_risk2 = pred_surv2
)

# Add outcome information - pass newdata to ensure proper ID matching
event_within_horizon <- find_events_within_horizon(df, horizon, newdata)
reclass_df$event <- event_within_horizon

# Extract relevant time information from your original dataset
# We need to find the actual observed time (either event time or censoring time)
time_info <- df %>%
  group_by(id) %>%
  summarize(
    max_time = max(tstop),
    event_time = ifelse(any(event == 1), min(tstop[event == 1]), max_time),
    time = pmin(event_time, horizon) # Censor at horizon for NRI analysis
  )

# Join this time information to your reclass_df
reclass_df <- left_join(reclass_df, select(time_info, id, time), by = "id")

# Now check that we have the time column
head(reclass_df)

# Check dimensions for nricens
print(paste("Length of pred_surv1:", length(pred_surv1)))
print(paste("Length of pred_surv2:", length(pred_surv2)))
print(paste("Length of reclass_df$time:", length(reclass_df$time)))
print(paste("Length of reclass_df$event:", length(reclass_df$event)))

# Now call nricens with the corrected data
nri_result <- nricens(
  time = reclass_df$time, # Time to event or censoring
  event = reclass_df$event, # Event indicator (1=event, 0=censored)
  p.std = pred_surv1, # Predicted risks from standard model
  p.new = pred_surv2, # Predicted risks from new model
  cut = risk_cats[-1], # Cut points for risk categories
  t0 = horizon # Time horizon for prediction
)

# Calculate observed event rates within each cell of the reclassification table
observed_rates <- aggregate(event ~ risk_cat1 + risk_cat2,
  data = reclass_df, FUN = mean
)

# Format as a matrix similar to Table 3 in the paper
observed_matrix <- matrix(NA,
  nrow = length(risk_labels),
  ncol = length(risk_labels)
)
rownames(observed_matrix) <- colnames(observed_matrix) <- risk_labels

for (i in 1:nrow(observed_rates)) {
  r <- which(risk_labels == observed_rates$risk_cat1[i])
  c <- which(risk_labels == observed_rates$risk_cat2[i])
  observed_matrix[r, c] <- round(100 * observed_rates$event[i], 1)
}
print(observed_matrix)


# Make sure all vectors have the same length before calling nricens
# Check dimensions of inputs
print(paste("Length of pred_surv1:", length(pred_surv1)))
print(paste("Length of pred_surv2:", length(pred_surv2)))
print(paste("Length of reclass_df$time:", length(reclass_df$time)))
print(paste("Length of reclass_df$event:", length(reclass_df$event)))

# Make sure we're using the same individuals for all vectors
# Create a complete data frame with all required variables
nri_data <- data.frame(
  id = newdata$id,
  pred_surv1 = pred_surv1[1, ],
  pred_surv2 = pred_surv2[1, ],
  time = reclass_df$time,
  event = reclass_df$event
)

# Remove any rows with NA values
nri_data <- nri_data[complete.cases(nri_data), ]

# Check the structure of our data
str(nri_data)

# Make sure time and event are properly formatted
# time should be numeric and event should be 0/1
nri_data$time <- as.numeric(nri_data$time)
nri_data$event <- as.numeric(nri_data$event)

# Try using the function with minimal parameters first
nri_result <- nricens(
  time = reclass_df$time, # Time to event or censoring
  event = reclass_df$event, # Event indicator (1=event, 0=censored)
  p.std = pred_surv1, # Predicted risks from standard model
  p.new = pred_surv2, # Predicted risks from new model
  cut = risk_cats[-1], # Cut points for risk categories
  t0 = horizon # Time horizon for prediction
)

print(summary(nri_result))
print(nri_result$nri)
# Overall reclassification table
print(nri_result$rtab)

# Reclassification table for cases (events)
print(nri_result$rtab.case)

# Reclassification table for controls (non-events)
print(nri_result$rtab.ctrl)

# Plots
### Sankey Plot
library(networkD3)

# Prepare data for Sankey diagram
links <- data.frame(
  source = character(),
  target = character(),
  value = numeric(),
  group = character()
)

# Get risk categories
risk_cats_labels <- c("< 5%", "5-10%", "10-20%", "20-100%", "≥ 100%")

# Create links from the reclassification tables
for (i in 1:nrow(nri_result$rtab)) {
  for (j in 1:ncol(nri_result$rtab)) {
    if (nri_result$rtab[i, j] > 0) {
      # For all individuals
      links <- rbind(links, data.frame(
        source = paste("Old:", risk_cats_labels[i]),
        target = paste("New:", risk_cats_labels[j]),
        value = nri_result$rtab[i, j],
        group = "All"
      ))
    }
  }
}

# Create Sankey diagram
nodes <- data.frame(
  name = unique(c(links$source, links$target))
)

links$source <- match(links$source, nodes$name) - 1
links$target <- match(links$target, nodes$name) - 1

sankeyNetwork(
  Links = links, Nodes = nodes,
  Source = "source", Target = "target",
  Value = "value", NodeID = "name",
  LinkGroup = "group", fontSize = 12
)

### Risk Shift Plot

# Create a proper data frame
plot_df <- data.frame(
  Model1 = nri_result$p.std[1, ],
  Model2 = nri_result$p.new[1, ],
  Event = ifelse(rep(1:2, length.out = length(nri_result$p.std)) == 1, "Case", "Control")
)

# Check the structure
head(plot_df)

# Create the plot and explicitly print it
rsp <- ggplot(plot_df, aes(x = Model1, y = Model2, color = Event)) +
  geom_point(alpha = 0.3) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  scale_color_manual(values = c("Case" = "red", "Control" = "blue")) +
  labs(
    x = "Risk from Model 1", y = "Risk from Model 2",
    title = "Change in Predicted Risk Between Models"
  ) +
  theme_minimal() +
  coord_fixed(ratio = 1)

# Explicitly print the plot
print(rsp)

# Also save the plot to ensure it's being generated
ggsave("../../tidy_data/A4/risk_shift_plot.pdf",
  plot = rsp,
  width = 8,
  height = 6,
  dpi = 300
)

### NRI components
nri_components <- data.frame(
  Component = c("Overall NRI", "Events (NRI+)", "Non-events (NRI-)"),
  Estimate = c(nri_result$nri[1, 1], nri_result$nri[2, 1], nri_result$nri[3, 1]),
  Lower = c(nri_result$nri[1, 2], nri_result$nri[2, 2], nri_result$nri[3, 2]),
  Upper = c(nri_result$nri[1, 3], nri_result$nri[2, 3], nri_result$nri[3, 3])
)

# Create bar plot with error bars
nri_components_plot <- ggplot(nri_components, aes(x = Component, y = Estimate, fill = Component)) +
  geom_bar(stat = "identity", width = 0.6) +
  geom_errorbar(aes(ymin = Lower, ymax = Upper), width = 0.2) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(
    title = "Net Reclassification Improvement Components",
    y = "NRI Value", x = ""
  ) +
  theme_minimal() +
  theme(legend.position = "none")

# Explicitly print the plot
print(nri_components_plot)

# Also save the plot to ensure it's being generated
ggsave("../../tidy_data/A4/nri_components_plot.pdf",
  plot = nri_components_plot,
  width = 8,
  height = 6,
  dpi = 300
)


### Heatmap of reclassification table
# Convert reclassification table to data frame
reclass_df <- as.data.frame(as.table(nri_result$rtab))
names(reclass_df) <- c("Old", "New", "Count")

# Calculate percentages for text labels
total_count <- sum(reclass_df$Count)
reclass_df$Percent <- round(100 * reclass_df$Count / total_count, 1)
reclass_df$Label <- paste0(reclass_df$Count, "\n(", reclass_df$Percent, "%)")

# Create publication-quality heat map
heatmap_reclass_df <- ggplot(reclass_df, aes(x = New, y = Old, fill = Count)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = Label),
    # Adjust text color based on background brightness for better contrast
    color = ifelse(reclass_df$Count > mean(reclass_df$Count) * 1.5, "white", "black"),
    fontface = "bold", size = 4
  ) + # Increased text size
  scale_fill_viridis_c(
    option = "mako", # Changed to "mako" for better contrast
    trans = "log",
    name = "Number of\nPatients",
    guide = guide_colorbar(
      title.position = "top",
      barwidth = 10,
      barheight = 0.5
    )
  ) +
  labs(
    title = "Risk Reclassification Matrix",
    subtitle = "Demographics + Lancet vs. pTau-217 + Demographics + Lancet",
    x = "Risk Category with pTau-217 Model",
    y = "Risk Category with Demographics Model",
    caption = "Numbers show count and percentage of total patients"
  ) +
  scale_x_discrete(position = "top") +
  theme_minimal(base_family = "Helvetica") +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5, margin = margin(b = 15)),
    plot.caption = element_text(size = 9, hjust = 1, margin = margin(t = 10)),
    axis.title = element_text(face = "bold", size = 12),
    axis.text = element_text(size = 10, face = "bold"),
    legend.position = "right",
    legend.title = element_text(size = 10, face = "bold"),
    legend.text = element_text(size = 9),
    panel.grid = element_blank(),
    panel.border = element_rect(fill = NA, color = "gray30", linewidth = 0.5),
    plot.margin = margin(20, 20, 20, 20)
  ) +
  # Add diagonal line highlighting to show unchanged classifications
  geom_tile(
    data = subset(reclass_df, Old == New),
    aes(x = New, y = Old),
    fill = NA, color = "black", linewidth = 1.2
  )

# Explicitly print the plot
print(heatmap_reclass_df)

# Also save the plot to ensure it's being generated
ggsave("../../tidy_data/A4/risk_reclassification_heatmap.pdf",
  plot = heatmap_reclass_df,
  width = 8, height = 7, dpi = 300
)

### Observed Event Rate Plot
# Create data frame with observed event rates
event_rates <- data.frame(
  Old = character(),
  New = character(),
  Count = numeric(),
  EventRate = numeric()
)

risk_levels <- colnames(nri_result$rtab)

for (i in 1:length(risk_levels)) {
  for (j in 1:length(risk_levels)) {
    total <- nri_result$rtab[i, j]
    if (total > 0) {
      events <- nri_result$rtab.case[i, j]
      event_rates <- rbind(event_rates, data.frame(
        Old = risk_levels[i],
        New = risk_levels[j],
        Count = total,
        EventRate = events / total * 100
      ))
    }
  }
}

# Plot event rates
event_rate_plot <- ggplot(event_rates, aes(x = New, y = Old, fill = EventRate)) +
  geom_tile() +
  geom_text(aes(label = sprintf("%.1f%%", EventRate)),
    color = ifelse(event_rates$EventRate > 50, "white", "black")
  ) +
  scale_fill_gradient(low = "white", high = "red") +
  labs(
    title = "Observed Event Rate by Risk Reclassification",
    x = "New Model Risk Category",
    y = "Original Model Risk Category",
    fill = "Event Rate (%)"
  ) +
  theme_minimal()

# Explicitly print the plot
print(event_rate_plot)

# Also save the plot to ensure it's being generated
ggsave("../../tidy_data/A4/event_rate_plot.pdf",
  plot = event_rate_plot,
  width = 8,
  height = 6,
  dpi = 300
)

### Decision Curve Analysis
library(rmda)

# Create decision curve analysis dataframe
dca_df <- data.frame(
  # event = reclass_df$event,
  event = ifelse(rep(1:2, length.out = length(nri_result$p.std)) == 1, 0, 1),
  p.std = nri_result$p.std[1, ], # Extract first row of matrix
  p.new = nri_result$p.new[1, ] # Extract first row of matrix
)
# dca_df$event <- as.factor(dca_df$event)
# Verify the data is properly formatted
print("Data structure:")
str(dca_df)
print(paste("Number of events:", sum(dca_df$event)))
print(paste("Range of p.std:", min(dca_df$p.std), "to", max(dca_df$p.std)))
print(paste("Range of p.new:", min(dca_df$p.new), "to", max(dca_df$p.new)))


# Create decision curve
decision_curve <- decision_curve(
  formula = event ~ p.std + p.new,
  data = dca_df,
  thresholds = seq(0, 0.5, by = 0.01)
)


# Plot decision curve
decision_curve_plot <- plot_decision_curve(decision_curve,
  curve.names = c("Standard Model", "New Model"),
  xlab = "Threshold Probability (%)",
  ylab = "Net Benefit",
  cost.benefit.axis = TRUE,
  col = c("blue", "red"),
  confidence.intervals = FALSE
)

# Explicitly print the plot
print(decision_curve_plot)

# Also save the plot to ensure it's being generated
ggsave("../../tidy_data/A4/decision_curve_plot.pdf",
  plot = decision_curve_plot,
  width = 8,
  height = 6,
  dpi = 300
)
