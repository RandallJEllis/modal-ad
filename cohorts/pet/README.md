# `pet/` — pooled amyloid-PET analysis across five cohorts

Time-to-event analysis of amyloid **PET** (centiloids) pooled across five cohorts:
**OASIS**, **NACC**, **HABS**, **ADNI**, and **AIBL**.

| File | Purpose |
| --- | --- |
| `build_datasets.py` | Harmonize and pool PET + demographic data across the five cohorts; standardize columns; build stratified folds and time-to-event labels. |
| `tvPET.R` | Time-varying-covariate Cox model on PET amyloid (with education; last-value carry-forward). |
| `meta_regression.R` | Cross-cohort meta-regression / meta-analysis of cohort-level estimates. |
| `generate_plots.R` | Survival / performance figures. |
| `vif_model_check.R` | Variance-inflation-factor multicollinearity diagnostics. |

See the root [`README.md`](../../README.md) and [`../../survival/time2event/`](../../survival/time2event/).
