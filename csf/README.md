# `csf/` — NACC cerebrospinal-fluid (CSF) biomarkers

Time-to-event analysis using **NACC** cerebrospinal-fluid biomarkers (e.g. amyloid-β,
tau) with demographics and Lancet risk factors.

| File | Purpose |
| --- | --- |
| `build_csf_datasets.py` | Assemble the CSF analysis dataset (biomarkers, demographics, outcome/time-to-event). |
| `tvCSF.R` | Time-varying-covariate Cox proportional-hazards model on CSF biomarkers. |
| `generate_plots.R` | Survival / performance figures. |

Multicollinearity diagnostics for this cohort live in
[`../nacc_csf/`](../nacc_csf/). See the root [`README.md`](../README.md) and
[`../time2event/`](../time2event/).
