suppressPackageStartupMessages({
  library(qs)
})

setwd({
  file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(file_arg) == 0) getwd() else dirname(normalizePath(sub("^--file=", "", file_arg[[1]]), mustWork = TRUE))
})

source("../vif_utils.R")

get_arg <- function(flag, default = NA_character_) {
  args <- commandArgs(trailingOnly = TRUE)
  idx <- match(flag, args)
  if (is.na(idx) || idx == length(args)) {
    return(default)
  }
  args[[idx + 1]]
}

load_dir <- get_arg("--load_dir", "../../tidy_data/pet_all_cohorts/age_65_cutoff")
model_name <- get_arg("--model_name", "centiloids_demographics")
outcome <- get_arg("--outcome", "allcausedementia_outcome")
analysis_set <- get_arg("--analysis_set", "age_65_cutoff")
output_dir <- get_arg(
  "--output_dir",
  file.path("../../results/pet_all_cohorts/vif_diagnostics", outcome, analysis_set)
)

fitted_models_path <- file.path(load_dir, "fitted_models_all.qs")
if (!file.exists(fitted_models_path)) {
  stop(paste0(
    "Could not find pooled PET fitted Cox models at ", fitted_models_path,
    ". Pass --load_dir pointing to a directory containing fitted_models_all.qs."
  ))
}

run_vif_diagnostics(
  fitted_models_path = fitted_models_path,
  output_dir = output_dir,
  cohort = "pooled_PET",
  outcome = outcome,
  analysis_set = analysis_set,
  model_name = model_name
)

message("Finished pooled PET VIF diagnostics.")
