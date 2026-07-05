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
library(mice)

setwd(dirname(this.path()))
options(pillar.width = Inf)

source("../time2event/plot_figures.R")
source("../time2event/metrics.R")

# cut_time_data <- function(td_data, interval_years = 1.7) {
#   # Create sequence of timepoints for each ID
#   td_data %>%
#     group_by(id) %>%
#     mutate(
#       # Round start and stop times to nearest interval
#       tstart = floor(tstart / interval_years) * interval_years,
#       tstop = ceiling(tstop / interval_years) * interval_years
#     ) %>%
#     # If this creates duplicate rows, keep last observation
#     group_by(id, tstart, tstop) %>%
#     slice_tail(n = 1) %>%
#     ungroup()
# }

for (ad_outcome in c(TRUE, 
                     FALSE)) {
  load_path <- "../../nacc/tidy_data/"
  if (ad_outcome) {
    load_path <- paste0(load_path, "ad_outcome/")
  }
  print(paste('AD outcome:', ad_outcome))
  # tmerge data for all models
  format_df <- function(df, #ptau = FALSE, lancet = FALSE, pet = FALSE,
                        clinical) {
    df$SEX <- factor(df$SEX)
    df$apoe <- factor(df$apoe)
    clinical$NACCTBI <- factor(clinical$NACCTBI)
    clinical$TBI <- factor(clinical$TBI)
    clinical$TBIBRIEF <- factor(clinical$TBIBRIEF)
    clinical$TBIWOLOS <- factor(clinical$TBIWOLOS)
    clinical$TOBAC30 <- factor(clinical$TOBAC30)
    clinical$TOBAC100 <- factor(clinical$TOBAC100)
    clinical$QUITSMOK <- factor(clinical$QUITSMOK)
    clinical$ALCFREQ <- factor(clinical$ALCFREQ)
    clinical$ALCOHOL <- factor(clinical$ALCOHOL)
    clinical$ALCABUSE <- factor(clinical$ALCABUSE)
    clinical$HYPERT <- factor(clinical$HYPERT)
    clinical$HYPERTEN <- factor(clinical$HYPERTEN)
    clinical$HXHYPER <- factor(clinical$HXHYPER)
    clinical$NACCAHTN <- factor(clinical$NACCAHTN)
    clinical$NACCHTNC <- factor(clinical$NACCHTNC)
    clinical$NACCDBMD <- factor(clinical$NACCDBMD)
    clinical$DIABET <- factor(clinical$DIABET)
    clinical$DIABETES <- factor(clinical$DIABETES)
    clinical$HEARING <- factor(clinical$HEARING)
    clinical$HEARAID <- factor(clinical$HEARAID)
    clinical$HEARWAID <- factor(clinical$HEARWAID)
    clinical$DEPD <- factor(clinical$DEPD)
    clinical$DEPDSEV <- factor(clinical$DEPDSEV)
    clinical$NACCADEP <- factor(clinical$NACCADEP)
    clinical$DEPTREAT <- factor(clinical$DEPTREAT)
    clinical$DEP2YRS <- factor(clinical$DEP2YRS)
    clinical$DEPOTHR <- factor(clinical$DEPOTHR)

    df <- within(df, apoe <- relevel(apoe, ref = "33"))

    base <- df[!duplicated(df$id), c(
      "id", "time_to_event", "event",
      "age", "age_squared",
      "SEX", "EDUC",
      "apoe"
    )]

    colnames(base) <- c(
      "id", "time", "event",
      "age", "age2",
      "sex", "educ",
      "apoe"
    )

    csf <- df[, c(
      "id", "time", "CSFABETA", "CSFPTAU", "CSFTTAU", "CSFABMD", "ratio_ptau_abeta"
    )]
    colnames(csf) <- c(
      "id", "time", "abeta", "ptau",
      "tau", "abeta_md", 
      "ratio_ptau_abeta"
    )

    # if (lancet) {
    clinical <- clinical[clinical$id %in% df$id, c(
      "id", "visit_to_days",
      "NACCTBI", "TBI", "TBIBRIEF", "TBIWOLOS",
      "TOBAC30", "TOBAC100", "SMOKYRS", "PACKSPER",
      "QUITSMOK", "ALCFREQ", "ALCOHOL", "ALCABUSE",
      "HYPERT", "HYPERTEN", "HXHYPER", "NACCAHTN",
      "NACCHTNC", "NACCBMI", "NACCDBMD", "DIABET",
      "DIABETES", "HEARING", "HEARAID", "HEARWAID",
      "DEPD", "DEPDSEV", "NACCGDS", "NACCADEP",
      "DEPTREAT", "DEP2YRS", "DEPOTHR"
    )]
    colnames(clinical) <- c(
      "id", "time", "nacctbi", "tbi", "tbi_brief", "tbi_wolos",
      "tobacco_30", "tobacco_100", "smoking_years", "pack_years",
      "quit_smoking_years", "alcohol_frequency", "alcohol", "alcohol_abuse",
      "current_hypertension", "hypertension", "hypertension_history",
      "hypertension_medication", "hypertension_combo_med", "bmi",
      "diabetes_medication", "diabetes_present", "diabetes", "hearing",
      "hearing_aid", "hearing_waid",
      "depression", "depression_severity", "gds_score", "current_depression_treatment",
      "depression_treated", "depression_lasttwoyears", "depression_overtwoyearsago"
    )

    clinical$time <- clinical$time / 365.25
    # }

    # base$time <- base$time / 365.25

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

    # if (ptau) {
      # Add the ptau column
    td_data <- tmerge(
      td_data,
      csf,
      id = id,
      ptau = tdc(time, ptau),
      abeta = tdc(time, abeta),
      tau = tdc(time, tau),
      abeta_md = tdc(time, abeta_md),
      ratio_ptau_abeta = tdc(time, ratio_ptau_abeta)
    )
    # }

    # if (pet) {
      # Add the centiloids column
    td_data <- tmerge(
      td_data,
      clinical,
      id = id,
      nacctbi = tdc(time, nacctbi),
      tbi = tdc(time, tbi),
      tbi_brief = tdc(time, tbi_brief),
      tbi_wolos = tdc(time, tbi_wolos),
      tobacco_30 = tdc(time, tobacco_30),
      tobacco_100 = tdc(time, tobacco_100),
      smoking_years = tdc(time, smoking_years),
      pack_years = tdc(time, pack_years),
      quit_smoking_years = tdc(time, quit_smoking_years),
      alcohol_frequency = tdc(time, alcohol_frequency),
      alcohol = tdc(time, alcohol),
      alcohol_abuse = tdc(time, alcohol_abuse),
      current_hypertension = tdc(time, current_hypertension),
      hypertension = tdc(time, hypertension),
      hypertension_history = tdc(time, hypertension_history),
      hypertension_medication = tdc(time, hypertension_medication),
      hypertension_combo_med = tdc(time, hypertension_combo_med),
      bmi = tdc(time, bmi),
      diabetes_medication = tdc(time, diabetes_medication),
      diabetes_present = tdc(time, diabetes_present),
      diabetes = tdc(time, diabetes),
      hearing = tdc(time, hearing),
      hearing_aid = tdc(time, hearing_aid),
      hearing_waid = tdc(time, hearing_waid),
      depression = tdc(time, depression),
      depression_severity = tdc(time, depression_severity),
      gds_score = tdc(time, gds_score),
      current_depression_treatment = tdc(time, current_depression_treatment),
      depression_treated = tdc(time, depression_treated),
      depression_lasttwoyears = tdc(time, depression_lasttwoyears),
      depression_overtwoyearsago = tdc(time, depression_overtwoyearsago)
    )
    # }

    # # if (lancet) {
    # td_data <- tmerge(
    #   td_data,
    #   habits,
    #   id = id,
    #   smoke = tdc(time, smoke),
    #   alcohol = tdc(time, alcohol),
    #   subuse = tdc(time, subuse),
    #   aerobic = tdc(time, aerobic),
    #   walking = tdc(time, walking)
    # )

    # td_data <- tmerge(
    #   td_data,
    #   psychwell,
    #   id = id,
    #   gdtotal = tdc(time, gdtotal),
    #   staital = tdc(time, staital)
    # )

    # td_data <- tmerge(
    #   td_data,
    #   vitals,
    #   id = id,
    #   vsbsys = tdc(time, vsbsys),
    #   vsdia = tdc(time, vsdia),
    #   bmi = tdc(time, bmi)
    # )
    # }

    # td_data <- td_data[complete.cases(td_data), ]

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


    td_data <- td_data[order(td_data$id, td_data$tstart), ]
    # Perform last observation carried forward (LOCF) within each subject
    td_data <- td_data %>%
      group_by(id) %>%
      fill(everything(), .direction = "down") %>%
      # Also carry first value backward for any remaining NAs
      fill(everything(), .direction = "up") %>%
      ungroup()

    # print number of NAs per column
    # print(sort(colSums(is.na(td_data))))
    drops <- c('quit_smoking_years', 'depression_severity',
               'hearing_waid', 'tbi', 'tbi_wolos', 'tbi_brief',
               'depression_treated', 'alcohol_frequency',
               'diabetes_present', 'current_hypertension',
               'hypertension_history', 'alcohol_abuse')

    td_data <- td_data %>% select(-all_of(drops))

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

  # read in Lancet data
  clinical <- read_parquet(paste0(load_path,
    "nacc_clinical_demographics.parquet"))

  # Define model formulas
  get_model_formula <- function(model_type, lancet = FALSE) {
    base_formulas <- list(
      "demographics_no_apoe" = Surv(tstart, tstop, event) ~ age + age2 +
        sex + educ,
      "demographics" = Surv(tstart, tstop, event) ~ age + age2 +
        sex + educ +
        apoe + age * apoe + age2 * apoe,
      "lancet" = Surv(tstart, tstop, event) ~ 1,
      "csf" = Surv(tstart, tstop, event) ~ ptau + abeta + tau + abeta_md +
        ratio_ptau_abeta,
      "csf_demographics_no_apoe" = Surv(tstart, tstop, event) ~ ptau +
        abeta + tau + abeta_md + ratio_ptau_abeta +
        age + age2 +
        sex + educ,
      "csf_demographics" = Surv(tstart, tstop, event) ~ ptau + abeta +
        tau + abeta_md + ratio_ptau_abeta +
        age + age2 + sex + educ + apoe + age * apoe + age2 * apoe
    )

    formula <- base_formulas[[model_type]]

    if (lancet) {
      formula <- update(formula, . ~ . + nacctbi + tobacco_30 + tobacco_100 +
      smoking_years + pack_years + alcohol + hypertension + hypertension_medication +
      hypertension_combo_med + bmi + diabetes_medication + diabetes + hearing +
      hearing_aid + depression + gds_score + current_depression_treatment +
      depression_lasttwoyears + depression_overtwoyearsago)
    }

    return(formula)
  }

  eval_times <- seq(2, 15)

  # lancet_vars <- c(
  #   "tbi", "tbi_brief", "tbi_wolos",
  #   "tobacco_30", "tobacco_100", "smoking_years", "pack_years",
  #   "quit_smoking_years", "alcohol_frequency", "alcohol_abuse",
  #   "hypertension", "hypertension_treatment", "hypertension_medication",
  #   "diabetes", "diabetes_medication", "hearing_aid", "hearing_waid",
  #   "depression", "depression_severity", "gad_score",
  #   "depression_treatment",
  #   "depression_duration", "depression_treatment_duration"
  # )

  # Initialize lists to store results for all models
  models_list <- list(
    "demographics_no_apoe" = list(),
    "demographics" = list(),
    "demographics_lancet_no_apoe" = list(),
    "demographics_lancet" = list(),
    "lancet" = list(),
    "csf" = list(),
    "csf_demographics_no_apoe" = list(),
    "csf_demographics" = list(),
    "csf_demographics_lancet_no_apoe" = list(),
    "csf_demographics_lancet" = list()
  )

  val_df_l <- list()
  train_df_l <- list()

  # Initialize lists to store results for all models
  metrics_list <- list()

  # iterate over folds and run experiments
  for (fold in seq(0, 4)) {
    print(paste0("Fold ", fold + 1))

    # Read and format data
    train_df_raw <- read_parquet(paste0(
      load_path, "train_", fold, ".parquet"
    ))
    sort(colSums(is.na(train_df_raw)))
    train_df_raw$CSFABMD <- as.factor(train_df_raw$CSFABMD)
    train_df_raw$CSFTTMD <- as.factor(train_df_raw$CSFTTMD)
    sort(colSums(is.na(train_df_raw)))

    train_imp <- mice(train_df_raw[,c("CSFABETA", "CSFPTAU",
                                      "CSFTTAU", "CSFABMD",
                                      "CSFTTMD", "age",
                                      "SEX", "EDUC", "apoe")])
    sort(colSums(is.na(complete(train_imp))))
    train_df_raw[, c("CSFABETA", "CSFPTAU", "CSFTTAU", "CSFABMD",
                     "CSFTTMD", "age", "SEX",
                     "EDUC", "apoe")] <- complete(train_imp)

    val_df_raw <- read_parquet(paste0(
      load_path, "val_", fold, ".parquet"
    ))

    val_df_raw$CSFABMD <- as.factor(val_df_raw$CSFABMD)
    val_df_raw$CSFTTMD <- as.factor(val_df_raw$CSFTTMD)
    val_imp <- mice.mids(train_imp, newdata = val_df_raw[,c("CSFABETA",
                          "CSFPTAU", "CSFTTAU", "CSFABMD", "CSFTTMD",
                          "age", "SEX", "EDUC", "apoe")])
    val_df_raw[, c("CSFABETA", "CSFPTAU", "CSFTTAU", "CSFABMD",
                   "CSFTTMD", "age", "SEX",
                   "EDUC", "apoe")] <- complete(val_imp)

    df <- format_df(train_df_raw, #ptau = is_ptau, lancet = is_lancet, pet = is_pet,
                      #habits, psychwell, vitals, centiloids
                      clinical
                      )
    val_df <- format_df(val_df_raw, #ptau = is_ptau, lancet = is_lancet,
                          #habits, psychwell, vitals, centiloids
                          clinical
                          )

    # scale lancet variables
    zscore_lancet_vars <- c("smoking_years", "pack_years", "bmi",
                            "gds_score"
                           )
    means <- apply(df[, zscore_lancet_vars], 2, mean, na.rm = TRUE)
    sds <- apply(df[, zscore_lancet_vars], 2, sd, na.rm = TRUE)

    df[, zscore_lancet_vars] <- scale(df[, zscore_lancet_vars],
      center = means, scale = sds
    )
    val_df[, zscore_lancet_vars] <- scale(val_df[, zscore_lancet_vars],
      center = means, scale = sds
    )
    
    train_df_l[[paste0("fold_", fold + 1)]] <- df
    val_df_l[[paste0("fold_", fold + 1)]] <- val_df
    
    # Fit all models
    for (model_name in names(models_list)) {
      print(paste("Fitting model:", model_name))

      # Determine if this is a Lancet model
      is_lancet <- grepl("lancet", model_name)
      is_csf <- grepl("csf", model_name)

      # print number of unique ids in df and val_df
      # print(paste0("Number of unique ids in df: ", length(unique(df$id))))
      # print(paste0("Number of unique ids in val_df: ", length(unique(val_df$id))))
      # print(fold)
      # print(model_name)
      # print(dim(df))
      # print(dim(val_df))
      # val_df_l[[paste0("fold_", fold + 1, "_", model_name)]] <- val_df
      # train_df_l[[paste0("fold_", fold + 1, "_", model_name)]] <- df
      # Z-score variables if using Lancet variables
      # if (is_lancet) {
      # }

      # Get base model type
      base_type <- gsub("_lancet", "", model_name)

      # Get formula
      formula <- get_model_formula(base_type, is_lancet)

      # Fit model
      model <- coxph(formula, data = df, x = TRUE)
      gc()
      models_list[[model_name]][[paste0("fold_", fold + 1)]] <- model

      # Calculate metrics
      metrics_results <- calculate_survival_metrics(
        model = model,
        model_name = model_name,
        data = val_df,
        times = eval_times
      )
      if (!model_name %in% names(metrics_list)) {
        metrics_list[[model_name]] <- list()
      }
      gc()
      metrics_list[[model_name]][[paste0("fold_", fold + 1)]] <- metrics_results
      gc()
    }
  }

  # Save results
  qs::qsave(models_list, paste0(load_path, "fitted_models.qs"))
  qs::qsave(val_df_l, paste0(load_path, "val_df_l.qs"))
  qs::qsave(train_df_l, paste0(load_path, "train_df_l.qs"))
  qs::qsave(metrics_list, paste0(load_path, "metrics.qs"))

  get_auc_ci_all_folds <- function(metrics_list, summarize = FALSE) {
    # Initialize empty dataframe for results
    all_results <- data.frame()

    # Loop through each model
    for (model_name in names(metrics_list)) {
      # Loop through each fold
      fold_results <- lapply(1:5, function(fold) {
        troc <- metrics_list[[model_name]][[paste0("fold_", fold)]]$troc
        ci <- timeROC:::confint.ipcwsurvivalROC(troc)

        data.frame(
          model = model_name,
          time = troc$times,
          auc = troc$AUC,
          ci_lower = ci$CI_AUC[, 1] / 100,
          ci_upper = ci$CI_AUC[, 2] / 100,
          fold = fold
        )
      })

      # Combine results from all folds
      model_results <- do.call(rbind, fold_results)

      if (summarize) {
        # Calculate mean values across folds for each time point
        summary_stats <- aggregate(
          cbind(auc, ci_lower, ci_upper) ~ model + time,
          data = model_results,
          FUN = mean
        )
      } else {
        summary_stats <- model_results
      }

      all_results <- rbind(all_results, summary_stats)
    }

    # Sort results by model and time
    all_results <- all_results[order(all_results$model, all_results$time), ]

    return(all_results)
  }

  auc_summary <- get_auc_ci_all_folds(metrics_list)
  write_parquet(auc_summary, paste0(load_path, "auc_summary.parquet"))
}
