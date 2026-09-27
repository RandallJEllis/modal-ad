# `nacc_csf/` — NACC cerebrospinal-fluid (CSF) biomarkers

Time-to-event analysis using **NACC** cerebrospinal-fluid biomarkers (e.g. amyloid-β,
tau) with demographics and Lancet risk factors, plus multicollinearity diagnostics.

| File | Purpose |
| --- | --- |
| `build_csf_datasets.py` | Assemble the CSF analysis dataset (biomarkers, demographics, outcome/time-to-event). |
| `tvCSF.R` | Time-varying-covariate Cox proportional-hazards model on CSF biomarkers. |
| `generate_plots.R` | Survival / performance figures (sources the shared plotting/metrics utilities in [`../../survival/time2event/`](../../survival/time2event/)). |
| `vif_model_check.R` | Variance-inflation-factor multicollinearity diagnostics for the CSF Cox models (uses [`../../analysis/vif_utils.R`](../../analysis/vif_utils.R)). |

See the root [`README.md`](../../README.md) and [`../../survival/time2event/`](../../survival/time2event/).
