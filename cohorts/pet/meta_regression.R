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
library(broom)
library(metafor)

setwd(dirname(this.path()))

source("../time2event/plot_figures.R")
source("../time2event/metrics.R")

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

cross_cohort_validation <- TRUE
if (cross_cohort_validation) {
  cross_cohort_validation_path <- "cross_cohort_validation/"
} else {
  cross_cohort_validation_path <- ""
}

exp_path <- paste0(base_path, age_cutoff_path, ad_outcome_path, cross_cohort_validation_path)

# tmerge data for all models
format_df <- function(df, lancet = FALSE) {
  df$sex <- factor(df$sex)
  df$apoe <- factor(df$apoe)
  df <- within(df, apoe <- relevel(apoe, ref = "33"))

  base <- df[!duplicated(df$id), c(
    "id", "time_to_event", "event",
    "age_centered", "age_centered_squared",
    "sex", "apoe", "education_z"
  )]
  tv_covar <- df[, c("id", "time", "centiloids")]
  colnames(base) <- c(
    "id", "time", "event", "age", "age2",
    "sex", "apoe", "education"
  )
  colnames(tv_covar) <- c("id", "time", "centiloids")

  if (lancet) {
    habits <- habits[habits$BID %in% df$BID, c(
      "BID",
      "COLLECTION_DATE_DAYS_CONSENT",
      "SMOKE", "ALCOHOL", "SUBUSE",
      "AEROBIC", "WALKING"
    )]
    psychwell <- psychwell[psychwell$BID %in% df$BID, c(
      "BID",
      "COLLECTION_DATE_DAYS_CONSENT",
      "GDTOTAL", "STAITOTAL"
    )]
    vitals <- vitals[vitals$BID %in% df$BID, c(
      "BID",
      "COLLECTION_DATE_DAYS_CONSENT",
      "VSBPSYS", "VSBPDIA"
    )]
    colnames(habits) <- c(
      "id", "time", "smoke", "alcohol", "subuse",
      "aerobic", "walking"
    )
    colnames(psychwell) <- c("id", "time", "gdtotal", "staital")
    colnames(vitals) <- c("id", "time", "vsbsys", "vsdia")

    habits$time <- habits$time / 365.25
    psychwell$time <- psychwell$time / 365.25
    vitals$time <- vitals$time / 365.25
  }

  # Create initial time-dependent data
  td_data <- tmerge(
    data1 = base,
    data2 = base,
    id = id,
    tstart = 0,
    tstop = time
  )

  # Add the event column
  td_data <- tmerge(
    td_data,
    base,
    id = id,
    event = event(time, event)
  )

  td_data <- tmerge(
    td_data,
    tv_covar,
    id = id,
    centiloids = tdc(time, centiloids)
  )

  if (lancet) {
    td_data <- tmerge(
      td_data,
      habits,
      id = id,
      smoke = tdc(time, smoke),
      alcohol = tdc(time, alcohol),
      subuse = tdc(time, subuse),
      aerobic = tdc(time, aerobic),
      walking = tdc(time, walking)
    )

    td_data <- tmerge(
      td_data,
      psychwell,
      id = id,
      gdtotal = tdc(time, gdtotal),
      staital = tdc(time, staital)
    )

    td_data <- tmerge(
      td_data,
      vitals,
      id = id,
      vsbsys = tdc(time, vsbsys),
      vsdia = tdc(time, vsdia)
    )
  }

  # First, let's store the baseline age for each person
  baseline_ages <- td_data %>%
    group_by(id) %>%
    slice_min(tstart) %>%
    select(id, baseline_age = age)

  # Now update the age column to reflect actual age at each timepoint
  td_data <- td_data %>%
    left_join(baseline_ages, by = "id") %>%
    mutate(
      # Convert tstart from days to years and add to baseline age
      age = baseline_age + (tstart)
    ) %>%
    select(-baseline_age) # Remove the temporary baseline_age column

  td_data <- td_data[order(td_data$id), ]
  # Perform last observation carried forward (LOCF) within each subject
  td_data <- td_data %>%
    group_by(id) %>%
    fill(everything(), .direction = "down") %>%
    # Also carry first value backward for any remaining NAs
    fill(everything(), .direction = "up") %>%
    ungroup()

  # print(dim(td_data))
  td_data <- td_data[complete.cases(td_data), ]
  # print(dim(td_data))
  # update age2
  td_data$age2 <- td_data$age^2

  # update age3
  td_data$age3 <- td_data$age^3

  # if (lancet) {
  #   td_data_updated <- cut_time_data(td_data_updated)
  # }

  return(td_data)
}

# Define model formulas
get_model_formula <- function(model_type, lancet = FALSE) {
  base_formulas <- list(
    "centiloids_demographics" = Surv(tstart, tstop, event) ~ centiloids +
      age + age2 + education +
      sex + apoe + age * apoe + age2 * apoe,
    "demographics" = Surv(tstart, tstop, event) ~ age + age2 + education +
      sex + apoe + age * apoe + age2 * apoe,
    "centiloids" = Surv(tstart, tstop, event) ~ centiloids,
    "demographics_no_apoe" = Surv(tstart, tstop, event) ~ age + age2 +
      sex + education,
    "centiloids_demographics_no_apoe" = Surv(
      tstart,
      tstop,
      event
    ) ~ centiloids +
      age + age2 + education +
      sex
  )

  formula <- base_formulas[[model_type]]

  # if (lancet) {
  #   formula <- update(formula, . ~ . +
  #                       smoke + alcohol + subuse +
  #                       aerobic + walking +
  #                       gdtotal + staital +
  #                       vsbsys + vsdia)
  # }

  return(formula)
}

# Initialize lists to store results for all models
models_list <- list(
  "centiloids_demographics" = list()
  # "demographics" = list(),
  # "centiloids" = list(),
  # "demographics_no_apoe" = list(),
  # "centiloids_demographics_no_apoe" = list()
)

# Initialize lists to store results for all models
metrics_list <- list()
val_df_l <- list()
train_df_l <- list()

eval_times <- seq(2, 10)

# iterate over folds and run experiments
for (fold in seq(0, 4)) {
  print(paste0("Fold ", fold + 1))

  # Read and format data
  train_df_raw <- read_parquet(paste0(
    exp_path, "train_", fold, ".parquet"
  ))
  val_df_raw <- read_parquet(paste0(
    exp_path, "val_", fold, ".parquet"
  ))

  df <- format_df(train_df_raw, lancet = FALSE)
  val_df <- format_df(val_df_raw, lancet = FALSE)
  df <- df[df$time < 100, ]
  val_df <- val_df[val_df$time < 100, ]
  val_df_l[[paste0("fold_", fold + 1)]] <- val_df
  train_df_l[[paste0("fold_", fold + 1)]] <- df

  # Fit all models
  for (model_name in names(models_list)) {
    print(paste("Fitting model:", model_name))

    # Determine if this is a Lancet model
    is_lancet <- grepl("lancet", model_name)

    # Z-score variables if using Lancet variables
    if (is_lancet) {
      lancet_vars <- c(
        "smoke", "alcohol", "aerobic", "walking",
        "gdtotal", "staital", "vsbsys", "vsdia"
      )
      means <- apply(df[, lancet_vars], 2, mean, na.rm = TRUE)
      sds <- apply(df[, lancet_vars], 2, sd, na.rm = TRUE)

      df[, lancet_vars] <- scale(df[, lancet_vars],
        center = means, scale = sds
      )
      val_df[, lancet_vars] <- scale(val_df[, lancet_vars],
        center = means, scale = sds
      )
    }

    # Get base model type
    base_type <- gsub("_lancet", "", model_name)

    # Get formula
    formula <- get_model_formula(base_type, is_lancet)

    # Fit model
    model <- coxph(formula, data = val_df, x = TRUE)
    models_list[[model_name]][[paste0("fold_", fold + 1)]] <- model
  }
}

# create long table out of models_list
# Simpler version with just a numeric index
extract_model_stats_indexed <- function(models_list) {
  results <- list()
  idx <- 1

  for (model_name in names(models_list)) {
    for (fold_name in names(models_list[[model_name]])) {
      model <- models_list[[model_name]][[fold_name]]
      coefs <- summary(model)$coefficients

      model_df <- data.frame(
        variable = rownames(coefs),
        beta = coefs[, "coef"],
        se = coefs[, "se(coef)"],
        model_index = idx, # Simple numeric index
        row.names = NULL
      )

      results[[idx]] <- model_df
      idx <- idx + 1
    }
  }

  do.call(rbind, results)
}

stats_df <- extract_model_stats_indexed(models_list)
stats_df <- stats_df[stats_df$model_index != 3, ]

# Run meta-analysis for EACH variable
# Loop-based approach (works reliably with rma)
unique_vars <- unique(stats_df$variable)
meta_results <- lapply(unique_vars, function(v) {
  subset_df <- stats_df[stats_df$variable == v, ]
  fit <- rma(yi = subset_df$beta, sei = subset_df$se, method = "REML")
  data.frame(
    variable    = v,
    pooled_beta = as.numeric(fit$beta),
    pooled_se   = fit$se,
    p_val       = fit$pval,
    tau2        = fit$tau2,
    I2          = fit$I2,
    H2          = fit$H2
  )
})
meta_summary <- do.call(rbind, meta_results)
clean_variable_names <- c(
  "PET", "Age", "Age2", "Education", "Sex",
  "APOE24", "APOE34", "APOE44", "APOE2_carrier", "Age:APOE24", "Age:APOE34",
  "Age:APOE44", "Age:APOE2_carrier", "Age2:APOE24", "Age2:APOE34",
  "Age2:APOE44", "Age2:APOE2_carrier"
)
meta_summary$variable <- clean_variable_names
print(meta_summary)

# save html table of results
library(kableExtra)
meta_summary %>%
  kbl(digits = 3, caption = "Meta-Analysis Results") %>%
  kable_styling(bootstrap_options = c("striped", "hover")) %>%
  save_kable(paste0(exp_path, "meta_analysis_table.html"))

# forest plot
# Build a dataframe with study-level and pooled estimates
plot_data <- do.call(rbind, lapply(unique_vars, function(v) {
  subset_df <- stats_df[stats_df$variable == v, ]
  fit <- rma(yi = subset_df$beta, sei = subset_df$se, method = "REML")

  # Per-study rows
  study_rows <- data.frame(
    variable = v,
    study = paste0("Fold ", subset_df$model_index),
    beta = subset_df$beta,
    ci_lo = subset_df$beta - 1.96 * subset_df$se,
    ci_hi = subset_df$beta + 1.96 * subset_df$se,
    type = "study"
  )

  # Pooled estimate row
  pooled_row <- data.frame(
    variable = v,
    study = "Pooled",
    beta = as.numeric(fit$beta),
    ci_lo = fit$ci.lb,
    ci_hi = fit$ci.ub,
    type = "pooled"
  )

  rbind(study_rows, pooled_row)
}))

# Map variable names to clean names
var_name_map <- setNames(clean_variable_names, unique_vars)
plot_data$variable <- var_name_map[plot_data$variable]

# Order study factor so Pooled appears at bottom
plot_data$study <- factor(plot_data$study,
  levels = c("Pooled", sort(unique(plot_data$study[plot_data$type == "study"])))
)

# remove interaction terms
plot_data <- plot_data[!grepl(":", plot_data$variable), ]


# Plot
ggplot(plot_data, aes(x = beta, y = study, xmin = ci_lo, xmax = ci_hi)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray50") +
  geom_errorbarh(data = subset(plot_data, type == "study"), height = 0.2) +
  geom_point(data = subset(plot_data, type == "study"), size = 3) +
  geom_errorbarh(data = subset(plot_data, type == "pooled"), height = 0.3, linewidth = 1) +
  geom_point(data = subset(plot_data, type == "pooled"), shape = 23, size = 4, fill = "black") +
  facet_wrap(~variable, scales = "free", ncol = 1) +
  labs(x = "Beta Coefficient", y = "") +
  theme_bw()

ggsave("../../pet_all_cohorts/figures/meta_analysis_forest_plot.png", height = 15, width = 5)
