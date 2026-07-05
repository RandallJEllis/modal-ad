suppressPackageStartupMessages({
  library(car)
  library(dplyr)
  library(readr)
})

script_dir <- function() {
  file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(file_arg) == 0) {
    return(getwd())
  }
  dirname(normalizePath(sub("^--file=", "", file_arg[[1]]), mustWork = TRUE))
}

make_file_slug <- function(x) {
  x <- gsub("[^A-Za-z0-9_+.-]+", "_", x)
  gsub("_+", "_", x)
}

vif_to_data_frame <- function(vif_res, cohort, outcome, analysis_set, model_name, fold) {
  if (is.matrix(vif_res)) {
    vif_cols <- colnames(vif_res)
    if ("GVIF^(1/(2*Df))" %in% vif_cols) {
      comparable_vif <- vif_res[, "GVIF^(1/(2*Df))"]
    } else if (all(c("GVIF", "Df") %in% vif_cols)) {
      comparable_vif <- vif_res[, "GVIF"]^(1 / (2 * vif_res[, "Df"]))
    } else {
      stop("VIF matrix did not include GVIF/Df columns.")
    }

    raw_gvif <- if ("GVIF" %in% vif_cols) vif_res[, "GVIF"] else NA_real_
    degrees_freedom <- if ("Df" %in% vif_cols) vif_res[, "Df"] else NA_real_

    return(data.frame(
      cohort = cohort,
      outcome = outcome,
      analysis_set = analysis_set,
      model = model_name,
      fold = fold,
      variable = rownames(vif_res),
      vif = as.numeric(comparable_vif),
      raw_gvif = as.numeric(raw_gvif),
      df = as.numeric(degrees_freedom),
      vif_type = "GVIF_adjusted",
      stringsAsFactors = FALSE
    ))
  }

  data.frame(
    cohort = cohort,
    outcome = outcome,
    analysis_set = analysis_set,
    model = model_name,
    fold = fold,
    variable = names(vif_res),
    vif = as.numeric(vif_res),
    raw_gvif = NA_real_,
    df = 1,
    vif_type = "VIF",
    stringsAsFactors = FALSE
  )
}

summarise_vif <- function(vif_df) {
  vif_df %>%
    group_by(cohort, outcome, analysis_set, model, variable, vif_type) %>%
    summarise(
      mean_vif = mean(vif, na.rm = TRUE),
      sd_vif = sd(vif, na.rm = TRUE),
      min_vif = min(vif, na.rm = TRUE),
      max_vif = max(vif, na.rm = TRUE),
      n_folds = dplyr::n(),
      n_folds_vif_gt_5 = sum(vif > 5, na.rm = TRUE),
      n_folds_vif_gt_10 = sum(vif > 10, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(desc(mean_vif))
}

select_model_name <- function(models_list, model_name = NULL, model_regex = NULL) {
  available_models <- names(models_list)

  if (!is.null(model_name) && !is.na(model_name) && model_name %in% available_models) {
    return(model_name)
  }

  if (!is.null(model_regex) && !is.na(model_regex)) {
    candidates <- grep(model_regex, available_models, ignore.case = TRUE, value = TRUE)
    if (length(candidates) == 1) {
      return(candidates[[1]])
    }
    if (length(candidates) > 1) {
      stop(paste0(
        "Model regex matched multiple models. Pass --model_name explicitly. Matches: ",
        paste(candidates, collapse = ", ")
      ))
    }
  }

  if (!is.null(model_name) && !is.na(model_name)) {
    stop(paste0(
      "Model '", model_name, "' was not found. Available models: ",
      paste(available_models, collapse = ", ")
    ))
  }

  stop(paste0(
    "No model name or unique regex match was supplied. Available models: ",
    paste(available_models, collapse = ", ")
  ))
}

run_vif_diagnostics <- function(fitted_models_path,
                                output_dir,
                                cohort,
                                outcome,
                                analysis_set,
                                model_name = NULL,
                                model_regex = NULL) {
  if (!file.exists(fitted_models_path)) {
    stop(paste0("Could not find fitted model object: ", fitted_models_path))
  }

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  models_list <- qs::qread(fitted_models_path)
  selected_model <- select_model_name(models_list, model_name, model_regex)
  model_by_fold <- models_list[[selected_model]]

  if (inherits(model_by_fold, "coxph")) {
    fold_names <- "fold_1"
    fold_models <- list(model_by_fold)
  } else {
    fold_models <- model_by_fold
    fold_names <- names(model_by_fold)
    if (is.null(fold_names) || any(fold_names == "")) {
      fold_names <- paste0("fold_", seq_along(model_by_fold))
    }
  }

  vif_rows <- list()
  status_rows <- list()

  for (i in seq_along(fold_models)) {
    fold_label <- fold_names[[i]]
    model <- fold_models[[i]]
    coef_values <- tryCatch(stats::coef(model), error = function(e) numeric())
    n_obs <- tryCatch(stats::nobs(model), error = function(e) NA_integer_)

    vif_res <- tryCatch(
      car::vif(model),
      error = function(e) e
    )

    if (inherits(vif_res, "error")) {
      status_rows[[length(status_rows) + 1]] <- data.frame(
        cohort = cohort,
        outcome = outcome,
        analysis_set = analysis_set,
        model = selected_model,
        fold = fold_label,
        status = "error",
        message = conditionMessage(vif_res),
        n_obs = n_obs,
        n_coefficients = length(coef_values),
        n_na_coefficients = sum(is.na(coef_values)),
        stringsAsFactors = FALSE
      )
      next
    }

    vif_rows[[length(vif_rows) + 1]] <- vif_to_data_frame(
      vif_res,
      cohort = cohort,
      outcome = outcome,
      analysis_set = analysis_set,
      model_name = selected_model,
      fold = fold_label
    )
    status_rows[[length(status_rows) + 1]] <- data.frame(
      cohort = cohort,
      outcome = outcome,
      analysis_set = analysis_set,
      model = selected_model,
      fold = fold_label,
      status = "ok",
      message = "",
      n_obs = n_obs,
      n_coefficients = length(coef_values),
      n_na_coefficients = sum(is.na(coef_values)),
      stringsAsFactors = FALSE
    )
  }

  status_df <- bind_rows(status_rows)
  vif_df <- bind_rows(vif_rows)

  slug <- make_file_slug(paste(cohort, outcome, analysis_set, selected_model, sep = "_"))
  status_path <- file.path(output_dir, paste0(slug, "_vif_status.csv"))
  readr::write_csv(status_df, status_path)

  if (nrow(vif_df) > 0) {
    by_fold_path <- file.path(output_dir, paste0(slug, "_vif_by_fold.csv"))
    summary_path <- file.path(output_dir, paste0(slug, "_vif_summary.csv"))
    readr::write_csv(vif_df, by_fold_path)
    readr::write_csv(summarise_vif(vif_df), summary_path)
  }

  invisible(list(
    model = selected_model,
    status = status_df,
    vif = vif_df,
    output_dir = output_dir
  ))
}
