# `ADNI/` — Alzheimer's Disease Neuroimaging Initiative

External validation in ADNI, focused on plasma pTau and demographic + Lancet risk
factors for time-to-dementia.

| File | Purpose |
| --- | --- |
| `build_datasets.py` | Assemble the ADNI analysis dataset (biomarkers, demographics, outcome/time-to-event). |
| `tvcox.R` | Time-varying-covariate Cox proportional-hazards model. |
| `generate_plots.R` | Survival / performance figures. |
| `vif_model_check.R` | Variance-inflation-factor multicollinearity diagnostics for the Cox design matrix. |

See the root [`README.md`](../../README.md) and [`../../survival/time2event/`](../../survival/time2event/).
