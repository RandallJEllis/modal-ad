# plotting_common.R
# Shared plotting/formatting helpers used by the survival publication-figure
# scripts. These functions are identical across the time-to-event analyses, so
# they live here and are sourced by the per-cohort plot_figures.R scripts.

get_publication_theme <- function() {
  publication_theme <- theme_minimal() +
    theme(
      # text = element_text(family = "Arial", size = 12),
      axis.title = element_text(size = 12, face = "bold"),
      axis.text = element_text(size = 10),
      legend.title = element_text(size = 12, face = "bold"),
      legend.text = element_text(size = 10),
      legend.position = "bottom",
      panel.grid.minor = element_blank(),
      panel.border = element_rect(fill = NA, color = "grey80"),
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 10, hjust = 0.5)
    )
  return(publication_theme)
}


process_calibration_data <- function(model_name, model, val_df, time,
                                     fixed_breaks, fold) {
  # Get predictions for current model
  pred_probs <- 1 - pec::predictSurvProb(
    model,
    newdata = val_df,
    times = time
  )

  # Create risk groups using fixed breaks
  risk_groups <- cut(pred_probs, breaks = fixed_breaks, include.lowest = TRUE)

  # Calculate calibration metrics for each risk group
  cal_data <- data.frame()
  for (group in levels(risk_groups)) {
    group_data <- val_df[risk_groups == group, ]
    if (nrow(group_data) > 0) {
      surv_fit <- survfit(Surv(tstop, event) ~ 1, data = group_data)
      surv_summary <- summary(surv_fit, times = time)

      if (length(surv_summary$surv) > 0) {
        cal_data <- rbind(cal_data, data.frame(
          fold = fold,
          time = time,
          model = model_name,
          risk_group = group,
          pred = mean(pred_probs[risk_groups == group]),
          actual = 1 - surv_summary$surv[1]
        ))
      }
    }
  }

  return(cal_data)
}

# Function to calculate calibration data across all models and folds


calculate_calibration_data <- function(models_list, val_df_l,
                                       times = seq(3, 8)) {
  selected_models <- c(
    # "demographics", "demographics_no_apoe",
    "demographics_lancet", "ptau",
    "ptau_demographics_lancet"
  )

  cal_data_all <- list()

  # First pass: collect all predictions to create fixed breaks
  for (t in times) {
    all_preds_by_model <- list()

    # Collect predictions across all folds and models
    for (fold in 0:4) {
      for (model_name in selected_models) {
        model <- overwrite_na_coef_to_zero(
          models_list[[model_name]][[paste0("fold_", fold + 1)]]
        )

        pred_probs <- 1 - pec::predictSurvProb(
          model,
          newdata = val_df_l[[paste0("fold_", fold + 1, "_", model_name)]],
          times = t
        )

        # Store predictions by model
        all_preds_by_model[[model_name]] <- c(
          all_preds_by_model[[model_name]],
          pred_probs
        )
      }
    }

    # Calculate fixed breaks using all predictions
    all_preds <- unlist(all_preds_by_model)
    raw_breaks <- quantile(all_preds, probs = seq(0, 1, length.out = 11))
    fixed_breaks <- numeric(length(raw_breaks))

    # Handle duplicate break points
    for (i in seq_along(raw_breaks)) {
      duplicates <- sum(raw_breaks[1:i] == raw_breaks[i])
      fixed_breaks[i] <- if (duplicates > 1) {
        raw_breaks[i] + (duplicates - 1) * .Machine$double.eps
      } else {
        raw_breaks[i]
      }
    }

    # Second pass: calculate calibration using fixed breaks
    cal_data_time <- list()

    for (fold in 0:4) {
      fold_data <- list()

      for (model_name in selected_models) {
        model <- overwrite_na_coef_to_zero(
          models_list[[model_name]][[paste0("fold_", fold + 1)]]
        )

        model_data <- process_calibration_data(
          model_name,
          model,
          val_df_l[[paste0("fold_", fold + 1, "_", model_name)]],
          t,
          fixed_breaks,
          fold
        )

        fold_data[[model_name]] <- model_data
      }

      cal_data_time[[paste0("fold_", fold + 1)]] <- do.call(rbind, fold_data)
    }

    cal_data_all[[as.character(t)]] <- cal_data_time
  }

  # Combine and summarize calibration data
  all_cal_data <- do.call(rbind, lapply(names(cal_data_all), function(t) {
    do.call(rbind, cal_data_all[[t]])
  }))

  # First, calculate the mean predictions and observed outcomes for each fold
  fold_level <- all_cal_data %>%
    group_by(time, model, risk_group, fold) %>%
    summarize(
      pred_mean = mean(pred),
      actual_prop = mean(actual),
      n = n(),
      .groups = "keep"
    )

  # Then aggregate across folds to get the final calibration points
  calibration_points <- fold_level %>%
    group_by(time, model, risk_group) %>%
    summarize(
      pred = mean(pred_mean), # Average prediction across folds
      actual = mean(actual_prop), # Average actual proportion across folds
      sd = sd(actual_prop), # SD
      lower = actual - sd,
      upper = actual + sd,
      n_folds = n(), # Number of folds contributing to this point
      .groups = "drop"
    )

  return(calibration_points)
}

# Modified plotting function to include confidence intervals


calibration_plots <- function(cal_data, times, model_colors) {
  publication_theme <- get_publication_theme()
  model_colors <- c(
    "demographics_lancet" = "#E69F00", # orange
    "ptau" = "#CC79A7", # pink
    "ptau_demographics_lancet" = "#009292" # turquoise
    # "demographics" = "#440154",           # deep purple
    # "demographics_no_apoe" = "#009E73"   # teal
  )

  model_labels <- c(
    # "Demo", "Demo (-APOE)",
    "Demo + Lancet",
    # "Demo+ Lancet\n(-APOE)",
    "pTau217",
    # "Demo + pTau217",
    # "Demo + pTau217\n(-APOE)",
    # "Demo + pTau217\n+ Lancet (-APOE)",
    "Demo + pTau217\n+ Lancet"
  )

  plots <- list()

  for (t in times) {
    t_data <- cal_data[cal_data$time == t, ]

    is_leftmost <- as.numeric(t) %in% c(3, 6)
    is_bottom <- as.numeric(t) >= 6
    is_middle_bottom <- t == 7

    max_limit <- max(max(t_data$pred), max(t_data$actual)) * 1.05

    current_plot <- ggplot(
      t_data,
      aes(
        x = pred, y = actual, color = model,
        fill = model
      )
    ) +
      geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.2) +
      geom_abline(
        slope = 1, intercept = 0, linetype = "dashed",
        color = "gray50"
      ) +
      geom_line(linewidth = 1) +
      geom_point(size = 2) +
      scale_color_manual(values = model_colors, name = "Model", labels = model_labels) +
      scale_fill_manual(values = model_colors, name = "Model", labels = model_labels) +
      labs(
        x = if (is_bottom) "Predicted Probability" else "",
        y = if (is_leftmost) "Observed Probability" else "",
        title = paste0(t, " years")
      ) +
      coord_equal(xlim = c(0, max_limit), ylim = c(0, max_limit)) +
      publication_theme +
      theme(legend.position = if (is_middle_bottom) "bottom" else "none")

    plots[[as.character(t)]] <- current_plot
  }

  return(wrap_plots(plots, ncol = 3))
}


dca_plots <- function(all_dca_data, times = seq(3, 8), model_colors = NULL) {
  if (is.null(model_colors)) {
    model_colors <- c(
      "demographics" = "#287271",
      "demographics_no_apoe" = "#B63679",
      "demographics_lancet" = "#E69F00", # orange
      "ptau" = "#CC79A7", # pink
      "ptau_demographics_lancet" = "#009292" # turquoise
    )
  }

  plots <- list()

  for (t in times) {
    # Pre-filter data for this timepoint
    t_data_all <- all_dca_data[all_dca_data$time == t, ]

    # Initialize matrices for all models
    models_to_analyze <- unique(t_data_all$model)
    thresholds <- NULL
    model_vals <- list()

    # Process each fold
    for (fold in unique(t_data_all$fold)) {
      fold_data <- t_data_all[t_data_all$fold == fold, ]

      # Calculate DCA once for reference strategies
      dca_ref <- stdca(
        data = fold_data[fold_data$model == models_to_analyze[1], ],
        outcome = "event",
        ttoutcome = "tstop",
        timepoint = t,
        predictors = "pred_prob",
        xstart = 0,
        xstop = 1,
        probability = FALSE,
        harm = NULL,
        graph = FALSE
      )

      # Store thresholds on first iteration
      if (is.null(thresholds)) {
        thresholds <- dca_ref$net.benefit$threshold
        n_thresholds <- length(thresholds)
        none_vals <- matrix(NA, nrow = length(unique(t_data_all$fold)), ncol = n_thresholds)
        all_vals <- matrix(NA, nrow = length(unique(t_data_all$fold)), ncol = n_thresholds)
        for (model_name in models_to_analyze) {
          model_vals[[model_name]] <- matrix(NA, nrow = length(unique(t_data_all$fold)), ncol = n_thresholds)
        }
      }

      # Store reference values
      none_vals[fold + 1, ] <- dca_ref$net.benefit$none
      all_vals[fold + 1, ] <- dca_ref$net.benefit$all

      # Calculate DCA for each model
      for (model_name in models_to_analyze) {
        model_data <- fold_data[fold_data$model == model_name, ]
        if (nrow(model_data) > 0) {
          dca_model <- stdca(
            data = model_data,
            outcome = "event",
            ttoutcome = "tstop",
            timepoint = t,
            predictors = "pred_prob",
            xstart = 0,
            xstop = 1,
            probability = FALSE,
            harm = NULL,
            graph = FALSE
          )
          model_vals[[model_name]][fold + 1, ] <- dca_model$net.benefit$pred_prob
        }
      }
    }

    # Calculate means and SDs
    none_mean <- colMeans(none_vals, na.rm = TRUE)
    all_mean <- colMeans(all_vals, na.rm = TRUE)
    model_means <- lapply(model_vals, colMeans, na.rm = TRUE)
    model_sds <- lapply(model_vals, function(x) apply(x, 2, sd, na.rm = TRUE))

    is_leftmost <- as.numeric(t) %in% c(3, 6)
    is_bottom <- as.numeric(t) >= 6
    is_middle_bottom <- t == 7

    # Create plot
    current_plot <- create_dca_plot(
      thresholds, none_mean, all_mean,
      model_means, model_sds, model_colors,
      models_to_analyze, t,
      is_leftmost, is_bottom, is_middle_bottom
    )

    plots[[as.character(t)]] <- current_plot
  }

  # Create final combined plot
  wrap_plots(plots, ncol = 3) +
    plot_layout(guides = "collect") &
    theme(legend.position = "bottom")
}

# Helper function to create individual DCA plot


create_dca_plot <- function(thresholds, none_mean, all_mean,
                            model_means, model_sds, model_colors,
                            models_to_analyze, t,
                            is_leftmost, is_bottom, is_middle_bottom) {
  current_plot <- ggplot() +
    # Add reference lines
    geom_line(
      data = data.frame(x = thresholds, y = none_mean),
      aes(x = x, y = y, linetype = "Treat None"),
      color = "gray50"
    ) +
    geom_line(
      data = data.frame(x = thresholds, y = all_mean),
      aes(x = x, y = y, linetype = "Treat All"),
      color = "gray50"
    )

  # Add model lines and ribbons
  for (model_name in models_to_analyze) {
    plot_data <- data.frame(
      x = thresholds,
      y = model_means[[model_name]],
      ymin = model_means[[model_name]] - model_sds[[model_name]],
      ymax = model_means[[model_name]] + model_sds[[model_name]],
      model = model_name
    )

    current_plot <- current_plot +
      geom_ribbon(
        data = plot_data,
        aes(x = x, y = y, ymin = ymin, ymax = ymax, fill = model),
        alpha = 0.2
      ) +
      geom_line(
        data = plot_data,
        aes(x = x, y = y, color = model),
        linewidth = 1
      )
  }

  current_plot +
    scale_color_manual(
      values = model_colors,
      name = "Model",
      labels = c(
        "Demographics",
        "Demographics (no APOE)",
        "Demographics + Lifestyle",
        "Plasma p-tau217",
        "Full Model"
      )
    ) +
    scale_fill_manual(
      values = model_colors,
      name = "Model",
      labels = c(
        "Demographics",
        "Demographics (no APOE)",
        "Demographics + Lifestyle",
        "Plasma p-tau217",
        "Full Model"
      )
    ) +
    scale_linetype_manual(
      values = c("Treat None" = "dashed", "Treat All" = "dotted"),
      name = "Strategy"
    ) +
    scale_y_continuous(limits = c(-0.05, NA)) +
    labs(
      x = if (is_bottom) "Threshold Probability" else "",
      y = if (is_leftmost) "Net Benefit" else "",
      title = paste(t, "years")
    ) +
    get_publication_theme() +
    theme(
      legend.position = if (is_middle_bottom) "bottom" else "none",
      legend.box = "vertical",
      aspect.ratio = 0.7
    )
}

# Function to create publication-quality figures


histogram_pvals <- function(results_table) {
  # Fig S1 - Histogram of p-values, bin size 0.05
  hist_pvalues <- ggplot(
    pvals_compare_trocs$all_results,
    aes(x = p_value)
  ) +
    geom_histogram(
      breaks = seq(0, 1, by = 0.05),
      fill = "#009292",
      alpha = 0.8,
      color = "white"
    ) + # Add white lines between bars
    geom_vline(xintercept = 0.05, linetype = "dashed", color = "red") +
    labs(
      x = "p-value",
      y = "Count"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 12),
      axis.text = element_text(size = 12), # Increased from 10
      axis.title = element_text(size = 14), # Increased from 11
      panel.grid.major = element_line(linewidth = 0.3), # Thicker grid lines
      panel.grid.minor = element_line(linewidth = 0.15) # Thicker minor grid lines
    )
  return(hist_pvalues)
}
