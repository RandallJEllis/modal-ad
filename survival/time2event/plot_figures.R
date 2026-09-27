# Shared identical plotting helpers (see survival/time2event/plotting_common.R)
library(this.path)
source(file.path(this.dir(), "plotting_common.R"))

library(survival)
library(timeROC)
library(riskRegression)
library(ggplot2)
library(patchwork) # for combining plots
library(scales) # for nice formatting
library(viridis) # for colorblind-friendly colors
library(gridExtra)
library(cowplot)
library(ggscidca)
library(tidyverse)

# Theme for consistent styling
get_colors_labels <- function(modality = NULL) {
  # Define colors for all models using a colorblind-friendly palette
  lookup_model_colors <- c(
    # Core models
    "demographics_lancet" = "#E69F00", # orange-yellow
    "ptau" = "#882255", # wine
    "ptau_demographics_lancet" = "#CC79A7",
    "centiloids_demographics_lancet" = "#79A2CC", # lighter indigo blue
    "ptau_centiloids_demographics_lancet" = "#009E73", # green
    "centiloids" = "#225588", # indigo/deep blue
    "csf_demographics_lancet" = "#009E73", # green
    "csf_demographics" = "#CC79A7", # pink
    "csf" = "#882255", # wine

    "demographics" = "#D55E00", # orange-red for CSF; "#E69F00", # orange-yellow for PET
    "demographics_no_apoe" = "#D55E00", # orange-red
    # PET (centiloids) base models
    "centiloids_demographics" = "#44AA99", # teal
    "centiloids_demographics_no_apoe" = "#117733", # dark green

    # Lancet variations
    "lancet" = "#56B4E9", # sky blue
    "demographics_lancet_no_apoe" = "#F0E442", # yellow

    # pTau combinations
    "ptau_demographics" = "#44AA99", # teal
    "ptau_demographics_no_apoe" = "#882255", # wine red
    "ptau_demographics_lancet_no_apoe" = "#117733", # green

    # PET with Lancet
    "centiloids_demographics_lancet_no_apoe" = "#CC6677", # olive

    # PET with pTau combinations
    "ptau_centiloids" = "#999933", # rose
    "ptau_centiloids_demographics" = "#AA4400", # brown
    "ptau_centiloids_demographics_no_apoe" = "#888888", # grey

    "ptau_centiloids_demographics_lancet_no_apoe" = "#44AA99" # blue green
  )

  if (!is.null(modality)) {
    if (modality == "csf") {
      # change color for demographics to #D55E00
      lookup_model_colors["demographics"] <- "#D55E00"
    }
  }

  # map model names to labels
  lookup_model_labels <- c(
    "demographics_lancet" = "Demo+APOE+Lancet",
    "ptau_demographics_lancet" = "pTau217+Demo+APOE+Lancet",
    "demographics" = "Demo+APOE",
    "demographics_no_apoe" = "Demo",
    "lancet" = "Lancet",
    "ptau" = "pTau217",
    "ptau_demographics" = "pTau217+Demo+APOE",
    "ptau_demographics_no_apoe" = "pTau217+Demo",
    "ptau_demographics_lancet_no_apoe" = "pTau217+Demo+Lancet",
    "demographics_lancet_no_apoe" = "Demo+Lancet",
    "centiloids_demographics_lancet" = "PET+Demo+APOE+Lancet",
    "ptau_centiloids_demographics_lancet" = "pTau217+PET+Demo+APOE+Lancet",
    "centiloids" = "PET",
    "centiloids_demographics" = "PET+Demo+APOE",
    "centiloids_demographics_no_apoe" = "PET+Demo",
    "centiloids_demographics_lancet" = "PET+Demo+APOE+Lancet",
    "centiloids_demographics_lancet_no_apoe" = "PET+Demo+Lancet",
    "ptau_centiloids" = "pTau217+PET",
    "ptau_centiloids_demographics" = "pTau217+PET+Demo+APOE",
    "ptau_centiloids_demographics_no_apoe" = "pTau217+PET+Demo",
    "ptau_centiloids_demographics_lancet" = "pTau217+PET+Demo+APOE+Lancet",
    "ptau_centiloids_demographics_lancet_no_apoe" = "pTau217+PET+Demo+APOE+Lancet (-APOE)",
    "csf_demographics_lancet" = "CSF+Demo+APOE+Lancet",
    "csf_demographics_lancet_no_apoe" = "CSF+Demo+Lancet",
    "csf_demographics" = "CSF+Demo+APOE",
    "csf_demographics_no_apoe" = "CSF+Demo",
    "csf" = "CSF"
  )

  return(list(
    colors = lookup_model_colors,
    labels = lookup_model_labels
  ))
}

td_plot <- function(summary_df, all_results_df = NULL, model_names, metric = "auc", all_models = FALSE, modality = NULL) {
  # Filter models if all_models is FALSE
  if (!all_models) {
    summary_df <- summary_df %>%
      filter(model %in%
        model_names)
  }

  lookup_model_colors <- get_colors_labels(modality)$colors
  lookup_model_labels <- get_colors_labels(modality)$labels

  # Create model_labels by subsetting lookup_model_labels
  model_labels <- lookup_model_labels[model_names]

  # Create model_colors by subsetting lookup_model_colors
  model_colors <- lookup_model_colors[model_names]

  # Set y-axis label based on metric
  y_labels <- list(
    auc = "AUROC",
    brier = "Brier Score",
    concordance = "Concordance"
  )

  titles <- list(
    auc = "Time-varying AUROC",
    brier = "Time-varying Brier score",
    concordance = "Time-varying Concordance"
  )

  # Create base plot using appropriate y-value column
  if (metric == "auc") {
    y_col <- "auc"
    lower_col <- "ci_lower"
    upper_col <- "ci_upper"
  } else {
    y_col <- "mean_metric"
    lower_col <- "ymin"
    upper_col <- "ymax"
  }

  base_plot <- ggplot() +
    # Add confidence interval ribbons
    geom_ribbon(
      data = summary_df,
      aes(
        x = time,
        ymin = .data[[lower_col]],
        ymax = .data[[upper_col]],
        fill = model
      ),
      alpha = 0.2
    ) +
    # Plot mean lines
    geom_line(
      data = summary_df,
      aes(
        x = time,
        y = .data[[y_col]],
        color = model
      ),
      linewidth = 1
    ) +
    # Add white circles at each time point
    geom_point(
      data = summary_df,
      aes(
        x = time,
        y = .data[[y_col]],
        color = model
      ),
      size = 3, shape = 21, fill = "white"
    ) +
    labs(
      title = titles[[metric]],
      x = "Time (years)",
      y = y_labels[[metric]]
    ) +
    scale_color_manual(
      values = model_colors,
      labels = model_labels,
      name = "Model"
    ) +
    scale_fill_manual(
      values = model_colors,
      labels = model_labels,
      name = "Model"
    ) +
    {
      if (metric == "auc") {
        list(
          scale_y_continuous(breaks = seq(0.3, 1, by = 0.1)),
          scale_x_continuous(breaks = seq(1, 10, 1)),
          geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50")
        )
      }
    } +
    theme_bw(base_size = 14) +
    theme(
      plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
      plot.subtitle = element_text(size = 14, hjust = 0.5),
      axis.title = element_text(face = "bold", size = 14),
      axis.text = element_text(size = 12),
      legend.title = element_text(face = "bold", size = 14),
      legend.text = element_text(size = 12),
      panel.grid.minor = element_blank(),
      panel.grid = element_blank(),
      panel.background = element_rect(fill = "white", color = NA),
      legend.position = "right"
    )
  return(base_plot)
}

# Helper function to process predictions and calibration data for one model
create_publication_figures <- function(baseline_model, biomarker_model,
                                       data, auc_summary, brier_summary,
                                       cal_data, times) {
  publication_theme <- get_publication_theme()

  # Colors
  model_colors <- c("Baseline" = "#287271", "Biomarker" = "#B63679")

  # 1. Time-Dependent AUC Plot
  td_auc <- td_plot(auc_summary, metric = "auc")
  td_brier <- td_plot(brier_summary, metric = "brier")

  # 2. Calibration Plots
  calibration <- calibration_plots(cal_data, times, model_colors)

  # 3. Decision Curve Analysis
  dca_plots_l <- dca_plots(
    baseline_model, biomarker_model,
    data, times, model_colors
  )

  # Combine plots
  combined_plot <- (
    (td_auc | dca_plots_l$final_plot) /
      (calibration$final_plot | plot_spacer())
  ) +
    plot_layout(widths = c(1, 1)) +
    plot_annotation(
      title = "Model Performance Comparison",
      subtitle = "Baseline vs. Biomarker Model",
      theme = theme(
        plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
        plot.subtitle = element_text(size = 12, hjust = 0.5)
      )
    )

  return(list(
    time_dependent_auc = td_auc,
    time_dependent_brier = td_brier,
    calibration = calibration$all_plots,
    decision_curve = dca_plots_l$all_plots,
    combined_plot = combined_plot
  ))
}

library(survival)
library(timeROC)
library(riskRegression)
library(ggplot2)
library(patchwork)
library(survcomp)
library(pec)

create_additional_figures <- function(models, val_data_dict, times) {
  publication_theme <- get_publication_theme()

  # Initialize lists to store ROC curves for each model
  roc_curves <- list()
  roc_data_list <- list()

  # Calculate ROC curves for each model using its corresponding validation data
  for (model_name in names(models)) {
    # Use the appropriate validation data for this model
    val_data <- val_data_dict[[model_name]]

    roc_curves[[model_name]] <- timeROC(
      T = val_data$tstop,
      delta = val_data$event,
      marker = predict(models[[model_name]], newdata = val_data, type = "lp"),
      cause = 1,
      times = times,
      iid = TRUE,
      ROC = TRUE
    )

    # Create data frame for this model
    roc_data_list[[model_name]] <- data.frame(
      FPR = as.vector(roc_curves[[model_name]]$FP),
      TPR = as.vector(roc_curves[[model_name]]$TP),
      Model = model_name,
      Time = rep(times, each = length(roc_curves[[model_name]]$FP) / length(times))
    )
  }

  # Combine all ROC data
  roc_data <- do.call(rbind, roc_data_list)

  # Create ROC plot
  p5 <- ggplot(roc_data, aes(x = FPR, y = TPR, color = Model)) +
    geom_line(linewidth = 1) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
    scale_color_manual(
      values = c(
        "demographics" = "#287271",
        "demographics_no_apoe" = "#B63679",
        "demographics_lancet" = "#4B0082",
        "ptau" = "#FF4500",
        "ptau_demographics_lancet" = "#008000"
      ),
      labels = c(
        "Demographics",
        "Demographics (no APOE)",
        "Demographics + Lifestyle",
        "Plasma p-tau217",
        "Full Model"
      )
    ) +
    facet_wrap(~Time, labeller = label_both) +
    labs(
      x = "False Positive Rate",
      y = "True Positive Rate",
      title = "Dynamic ROC Curves",
      subtitle = "At Different Follow-up Times"
    ) +
    coord_equal() +
    publication_theme

  # Calculate prediction error curves using appropriate validation data for each model
  pe <- tryCatch(
    {
      # Create a list to store predictions for each model
      predictions <- list()
      for (model_name in names(models)) {
        val_data <- val_data_dict[[model_name]]
        predictions[[model_name]] <- pec::predictSurvProb(
          models[[model_name]],
          newdata = val_data,
          times = times
        )
      }

      # Use the first validation dataset's structure for the reference
      reference_data <- val_data_dict[[names(models)[1]]]

      pec(
        object = predictions,
        data = reference_data,
        times = times,
        exact = FALSE,
        reference = TRUE,
        splitMethod = "none",
        formula = Surv(tstop, event) ~ 1,
        start = 3,
        verbose = FALSE
      )
    },
    error = function(e) {
      warning("Prediction error calculation failed, returning NULL")
      NULL
    }
  )

  # Create prediction error plot if calculation succeeded
  if (!is.null(pe)) {
    # Create data frame for all models
    pe_data <- data.frame(
      time = rep(pe$time, length(models)),
      error = unlist(lapply(
        names(models),
        function(model_name) pe$AppErr[[model_name]]
      )),
      Model = factor(rep(names(models), each = length(pe$time)))
    )

    p6 <- ggplot(pe_data, aes(x = time, y = error, color = Model)) +
      geom_line(linewidth = 1) +
      scale_color_manual(
        values = c(
          "demographics" = "#287271",
          "demographics_no_apoe" = "#B63679",
          "demographics_lancet" = "#4B0082",
          "ptau" = "#FF4500",
          "ptau_demographics_lancet" = "#008000"
        ),
        labels = c(
          "Demographics",
          "Demographics (no APOE)",
          "Demographics + Lifestyle",
          "Plasma p-tau217",
          "Full Model"
        )
      ) +
      labs(
        x = "Time (Years)",
        y = "Prediction Error",
        title = "Integrated Prediction Error",
        subtitle = "Lower Values Indicate Better Performance"
      ) +
      publication_theme
  } else {
    p6 <- NULL
  }

  # Create combined plot
  combined_additional <- (p5 + p6) +
    plot_annotation(
      title = "Additional Model Performance Metrics",
      theme = theme(
        plot.title = element_text(size = 16, face = "bold", hjust = 0.5)
      )
    )

  return(list(
    dynamic_roc = list(
      plot = p5,
      data = roc_data
    ),
    prediction_error = list(
      plot = p6,
      data = if (!is.null(pe)) pe_data else NULL
    ),
    combined_additional = combined_additional,
    troc = roc_curves[[1]] # Return first ROC curve for backwards compatibility
  ))
}


plot_auc_over_time <- function(auc_summary, model_names, modality = NULL) {
  sub_auc_summary <- auc_summary %>%
    filter(model %in% model_names) %>%
    mutate(
      fold = as.factor(fold), # Make fold a factor for better grouping
      metric = auc
    )
  sub_auc_summary$model <- factor(sub_auc_summary$model, levels = model_names)

  agg_sub_auc_summary <- aggregate(
    cbind(auc, ci_lower, ci_upper) ~ model + time,
    data = sub_auc_summary,
    FUN = mean,
    na.rm = TRUE
  )
  agg_sub_auc_summary$model <- factor(agg_sub_auc_summary$model, levels = model_names)

  # plot auc over time
  auc_plot <- td_plot(agg_sub_auc_summary,
    sub_auc_summary,
    model_names,
    metric = "auc",
    modality = modality
  )

  return(auc_plot)
}

plot_all_roc_curves <- function(model_names, eval_times, modality = NULL) {
  width <- 8
  height <- 6
  roc_summary <- pull_roc_summary(model_names, eval_times)

  lookup_model_colors <- get_colors_labels(modality)$colors
  lookup_model_labels <- get_colors_labels(modality)$labels

  model_colors <- lookup_model_colors[model_names]
  model_labels <- lookup_model_labels[model_names]

  # Create faceted plot of ROC curves
  roc_plot <- ggplot(roc_summary, aes(x = FPR, y = mean_TPR, color = Model)) +
    geom_ribbon(aes(
      ymin = ci_lower,
      ymax = ci_upper,
      fill = Model
    ), alpha = 0.3, color = NA) +
    geom_line(linewidth = 0.5) +
    geom_abline(
      slope = 1, intercept = 0,
      linetype = "dashed", color = "gray50"
    ) +
    scale_color_manual(values = model_colors, labels = model_labels) +
    scale_fill_manual(values = model_colors, labels = model_labels, guide = "none") +
    facet_wrap(~Time,
      labeller = labeller(Time = function(x) sprintf("%s years", x))
    ) +
    labs(
      x = "False Positive Rate",
      y = "True Positive Rate",
      title = "ROC Curves at Different Time Points"
    ) +
    coord_fixed() +
    theme_minimal() +
    theme(
      plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
      plot.subtitle = element_text(size = 14, hjust = 0.5),
      axis.title = element_text(face = "bold", size = 14),
      # axis.text = element_text(size = 12),
      legend.title = element_text(face = "bold", size = 14),
      legend.text = element_text(size = 12),
      panel.grid.minor = element_blank(),
      # panel.grid = element_blank(),
      # panel.background = element_rect(fill = "white", color = NA),
      legend.position = "right",
      panel.spacing = unit(1, "cm"),
      axis.text = element_text(size = 8),
      plot.margin = margin(0.5, 0.5, 0.5, 0.5, "cm")
    )

  return(roc_plot)
}

plot_roc_biggest_year_difference <- function(auc_summary, agg_auc_summary, model_names, eval_times, modality = NULL) {
  lookup_model_colors <- get_colors_labels(modality)$colors
  lookup_model_labels <- get_colors_labels(modality)$labels

  model_colors <- lookup_model_colors[model_names]
  model_labels <- lookup_model_labels[model_names]

  mean_diffs <- agg_auc_summary %>%
    filter(model %in% model_names) %>%
    pivot_wider(
      id_cols = time,
      names_from = model,
      values_from = auc
    ) %>%
    mutate(auc_difference = .data[[model_names[2]]] - .data[[model_names[1]]]) %>%
    select(time, auc_difference)

  # Find the year with the largest difference in AUC between the two models
  year <- mean_diffs$time[which.max(abs(mean_diffs$auc_difference))]

  ##### Figure 1B - Create individual panel for time = 7
  roc_summary <- pull_roc_summary(model_names, eval_times)
  roc_year <- roc_summary %>%
    filter(Time == year)

  model_labels <- lookup_model_labels[model_names]
  model_colors <- lookup_model_colors[model_names]

  p_year <- ggplot(roc_year, aes(x = FPR, y = mean_TPR, color = Model)) +
    geom_ribbon(
      aes(
        ymin = ci_lower,
        ymax = ci_upper,
        fill = Model
      ),
      alpha = 0.3,
      color = NA
    ) +
    geom_line(linewidth = 1) +
    geom_abline(
      slope = 1, intercept = 0,
      linetype = "dashed", color = "gray50"
    ) +
    scale_color_manual(values = model_colors, labels = model_labels) +
    scale_fill_manual(values = model_colors, labels = model_labels) +
    labs(
      x = "False Positive Rate",
      y = "True Positive Rate",
      title = paste(
        year, "years\n",
        # Create the text output using map2 from purrr
        paste(
          map2(model_names, model_labels, function(model_name, model_label) {
            sprintf(
              "%s: %.3f (%.3f-%.3f)",
              model_label,
              mean(auc_summary$auc[auc_summary$time == year &
                auc_summary$model == model_name], na.rm = TRUE),
              mean(auc_summary$ci_lower[auc_summary$time == year &
                auc_summary$model == model_name], na.rm = TRUE),
              mean(auc_summary$ci_upper[auc_summary$time == year &
                auc_summary$model == model_name], na.rm = TRUE)
            )
          }) %>%
            paste(collapse = "\n")
        )
      )
    ) +
    coord_fixed() +
    get_publication_theme() +
    theme(
      legend.position = "bottom",
      plot.margin = margin(0.5, 0.5, 0.5, 0.5, "cm"),
      panel.border = element_rect(color = "black", fill = NA, linewidth = 0.5)
    )

  # return plot and year
  return(list(plot = p_year, year = year))
}

plot_brier_over_time <- function(metrics_list, model_names, modality = NULL) {
  brier_results <- collate_metric(metrics_list, metric = "brier")

  brier_summary <- brier_results %>%
    group_by(model, time) %>%
    summarise(
      mean_metric = mean(metric, na.rm = TRUE),
      sd_metric = sd(metric, na.rm = TRUE),
      ymin = pmax(mean_metric - sd_metric, 0),
      ymax = pmin(mean_metric + sd_metric, 1),
      .groups = "drop"
    )
  brier_summary$model <- factor(brier_summary$model, levels = model_names)

  # Figure 1D - plot brier score over time
  brier_plot <- td_plot(brier_summary,
    model_names = model_names,
    metric = "brier",
    all_models = F,
    modality = modality
  )

  return(brier_plot)
}

plot_concordance_over_time <- function(metrics_list, model_names, modality = NULL) {
  concordance_results <- collate_metric(metrics_list, metric = "concordance")
  cc_sub <- concordance_results %>%
    filter(model %in% model_names) %>%
    mutate(
      fold = as.factor(fold),
      metric = metric
    )
  cc_sub$model <- factor(cc_sub$model, levels = model_names)

  concordance_summary <- concordance_results %>%
    group_by(model, time) %>%
    summarise(
      mean_metric = mean(metric, na.rm = TRUE),
      sd_metric = sd(metric, na.rm = TRUE),
      ymin = pmax(mean_metric - sd_metric, 0),
      ymax = pmin(mean_metric + sd_metric, 1),
      .groups = "drop"
    )
  concordance_summary$model <- factor(concordance_summary$model, levels = model_names)


  # Figure 1C - plot concordance over time
  concordance_plot <- td_plot(concordance_summary,
    concordance_results,
    model_names = model_names,
    metric = "concordance",
    modality = modality
  )

  return(concordance_plot)
}


plot_SeSpPPVNPV <- function(data, metric, modality = NULL) {
  color_label_info <- get_colors_labels(modality)
  colors <- color_label_info$colors
  labels <- color_label_info$labels

  # set y-axis label to capitalize first letter unless it's PPV or NPV
  if (!metric %in% c("ppv", "npv")) {
    y_label <- tools::toTitleCase(metric)
  } else {
    y_label <- toupper(tools::toTitleCase(metric))
  }

  ggplot(data, aes(x = time, y = get(paste0("mean_", metric)), color = model)) +
    geom_ribbon(
      aes(
        ymin = get(paste0("mean_", metric)) - get(paste0("sd_", metric)),
        ymax = get(paste0("mean_", metric)) + get(paste0("sd_", metric)),
        fill = model
      ),
      alpha = 0.2,
      color = NA
    ) +
    geom_line(linewidth = 1) +
    # Add white circles at each time point
    geom_point(aes(color = model), fill = "white", size = 3, shape = 21) +
    scale_color_manual(values = colors, labels = labels) +
    scale_fill_manual(values = colors, labels = labels) +
    labs(
      x = "Time (years)",
      y = y_label,
      color = "Model",
      fill = "Model" # Add fill legend
    ) +
    theme_bw(base_size = 14) +
    theme(
      plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
      plot.subtitle = element_text(size = 14, hjust = 0.5),
      axis.title = element_text(face = "bold", size = 14),
      axis.text = element_text(size = 12),
      legend.title = element_text(face = "bold", size = 14),
      legend.text = element_text(size = 12),
      panel.grid.minor = element_blank(),
      panel.grid = element_blank(),
      panel.background = element_rect(fill = "white", color = NA),
      legend.position = "right"
    )
}

save_all_figures <- function(modality = NULL, model_names, models_list, metrics_list,
                             train_df_l, val_df_l, eval_times,
                             width, height, dpi, main_path) {
  ##### FIGURE 1A: AUC over time
  # AUROC and CIs
  print("Plotting AUC over time")
  auc_summary <- read_parquet(paste0(main_path, "auc_summary.parquet"))

  # filter to only include eval_times
  auc_summary <- auc_summary %>%
    filter(time >= eval_times[1] & time <= eval_times[length(eval_times)])

  # aggregate to get mean and CI
  agg_auc_summary <- aggregate(
    cbind(auc, ci_lower, ci_upper) ~ model + time,
    data = auc_summary,
    FUN = mean,
    na.rm = TRUE
  )
  auc_plot <- plot_auc_over_time(auc_summary, model_names, modality = modality)

  # Save plots
  ggsave(paste0(main_path, "final_auc_Over_Time.pdf"),
    plot = auc_plot,
    width = width,
    height = height,
    dpi = 300
  )

  print("Plotting individual year ROC curves")
  roc_plot <- plot_all_roc_curves(model_names, eval_times = eval_times, modality = modality)
  # Save the plot
  ggsave(paste0(main_path, "ROC_curves_by_timepoint.pdf"),
    plot = roc_plot,
    width = width * 1.5,
    height = height,
    dpi = 300
  )


  # Find the year with the largest difference in AUC between demographics_lancet and ptau_demographics_lancet
  print("Plotting ROC curve for the year with the largest difference in AUC")
  p_year <- plot_roc_biggest_year_difference(auc_summary,
    agg_auc_summary,
    model_names,
    eval_times = eval_times,
    modality = modality
  )
  # Save plots
  ggsave(paste0(main_path, "final_ROCcurve_", p_year$year, "years.pdf"),
    plot = p_year$plot,
    width = width,
    height = height,
    dpi = 300
  )


  ###### Figure 1D: BRIER SCORE - plot brier score over time
  print("Plotting Brier score over time")
  brier_plot <- plot_brier_over_time(metrics_list, model_names, modality = modality)

  # Save plots
  ggsave(paste0(main_path, "final_brier_Over_Time.pdf"),
    plot = brier_plot,
    width = width,
    height = height,
    dpi = 300
  )

  ##### Figure 1C: plot concordance over time
  print("Plotting concordance over time")
  concordance_plot <- plot_concordance_over_time(metrics_list, model_names, modality = modality)

  # Save plots
  ggsave(paste0(main_path, "final_concordance_Over_Time.pdf"),
    plot = concordance_plot,
    width = width,
    height = height,
    dpi = 300
  )

  ########################################################
  # Sensitivity, Specificity, PPV, NPV
  # Function to calculate SeSpPPVNPV for a model and fold
  # Initialize list to store results

  ##### Figure 1E: Sensitivity, Specificity, PPV, NPV
  print("Plotting sensitivity, specificity, PPV, NPV")
  # Set up parallel processing
  df_sespppvnpv <- SeSpPPVNPV_summary(models_list, model_names, eval_times, train_df_l, val_df_l)
  write_parquet(df_sespppvnpv, paste0(main_path, "sespppvnpv_summary.parquet"))

  # Create individual plots
  sensitivity_plot <- plot_SeSpPPVNPV(df_sespppvnpv, "sensitivity", modality = modality)
  specificity_plot <- plot_SeSpPPVNPV(df_sespppvnpv, "specificity", modality = modality)
  ppv_plot <- plot_SeSpPPVNPV(df_sespppvnpv, "ppv", modality = modality)
  npv_plot <- plot_SeSpPPVNPV(df_sespppvnpv, "npv", modality = modality)

  ggsave(
    paste0(main_path, "sensitivity_plot.pdf"),
    plot = sensitivity_plot,
    width = width * 1.2,
    height = 6,
    dpi = dpi
  )

  ggsave(
    paste0(main_path, "specificity_plot.pdf"),
    plot = specificity_plot,
    width = width * 1.2,
    height = 6,
    dpi = dpi
  )

  ggsave(
    paste0(main_path, "ppv_plot.pdf"),
    plot = ppv_plot,
    width = width * 1.2,
    height = 6,
    dpi = dpi
  )

  ggsave(
    paste0(main_path, "npv_plot.pdf"),
    plot = npv_plot,
    width = width * 1.2,
    height = 6,
    dpi = dpi
  )
}
