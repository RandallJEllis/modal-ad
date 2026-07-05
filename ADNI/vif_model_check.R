suppressPackageStartupMessages({
  library(qs)
})

setwd({
  file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(file_arg) == 0) getwd() else dirname(normalizePath(sub("^--file=", "", file_arg[[1]]), mustWork = TRUE))
})

source("../vif_utils.R")

model_name <- "ptau_demographics_lancet"
outcomes <- c("allcausedementia_outcome", "alzheimers_outcome")
analysis_sets <- c(
  primary_all_ages = "",
  age_65 = "age_65",
  agecutoff_65 = "agecutoff_65"
)

results <- list()

for (outcome in outcomes) {
  for (analysis_set in names(analysis_sets)) {
    subdir <- analysis_sets[[analysis_set]]
    load_dir <- if (subdir == "") {
      file.path("../../tidy_data/ADNI", outcome)
    } else {
      file.path("../../tidy_data/ADNI", outcome, subdir)
    }
    fitted_models_path <- file.path(load_dir, "fitted_models.qs")

    if (!file.exists(fitted_models_path)) {
      message("Skipping missing ADNI fitted models: ", fitted_models_path)
      next
    }

    output_dir <- file.path("../../results/ADNI/vif_diagnostics", outcome, analysis_set)
    message("Running ADNI VIF diagnostics for ", outcome, " / ", analysis_set)
    results[[paste(outcome, analysis_set, sep = "_")]] <- run_vif_diagnostics(
      fitted_models_path = fitted_models_path,
      output_dir = output_dir,
      cohort = "ADNI",
      outcome = outcome,
      analysis_set = analysis_set,
      model_name = model_name
    )
  }
}

message("Finished ADNI VIF diagnostics.")
