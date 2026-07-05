suppressPackageStartupMessages({
  library(qs)
})

setwd({
  file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(file_arg) == 0) getwd() else dirname(normalizePath(sub("^--file=", "", file_arg[[1]]), mustWork = TRUE))
})

source("../../analysis/vif_utils.R")

get_arg <- function(flag, default = NA_character_) {
  args <- commandArgs(trailingOnly = TRUE)
  idx <- match(flag, args)
  if (is.na(idx) || idx == length(args)) {
    return(default)
  }
  args[[idx + 1]]
}

load_dir <- get_arg("--load_dir", "../../tidy_data/nacc_csf/ad_outcome")
model_name <- get_arg("--model_name", "csf_demographics_lancet")
model_regex <- get_arg(
  "--model_regex",
  "CSF\\+Demo\\+APOE\\+Lancet|csf.*(demo|demographics).*apoe.*lancet|csf.*(demo|demographics).*lancet"
)
outcome <- get_arg("--outcome", "ad_outcome")
analysis_set <- get_arg("--analysis_set", "primary")
output_dir <- get_arg(
  "--output_dir",
  file.path("../../results/nacc_csf/vif_diagnostics", outcome, analysis_set)
)

fitted_models_path <- file.path(load_dir, "fitted_models.qs")
if (!file.exists(fitted_models_path)) {
  stop(paste0(
    "Could not find NACC CSF fitted Cox models at ", fitted_models_path,
    ". This checkout only appears to contain derived NACC CSF summaries; rerun this script ",
    "on the machine/path with fitted_models.qs for the CSF+Demo+APOE+Lancet model."
  ))
}

run_vif_diagnostics(
  fitted_models_path = fitted_models_path,
  output_dir = output_dir,
  cohort = "NACC_CSF",
  outcome = outcome,
  analysis_set = analysis_set,
  model_name = model_name,
  model_regex = model_regex
)

message("Finished NACC CSF VIF diagnostics.")
