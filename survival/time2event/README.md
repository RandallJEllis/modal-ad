# `time2event/` — shared survival-analysis utilities and figures

Common R utilities for the time-to-event analyses across cohorts (UK Biobank, A4,
ADNI, PET, CSF), plus publication figure generation.

| File | Purpose |
| --- | --- |
| `metrics.R` | Time-dependent survival metrics: **Brier score with decomposition** (reliability / resolution / uncertainty), time-dependent AUC, **net reclassification improvement (NRI)** (`nricens`), and **decision-curve analysis** (`rmda`/`dcurves`). Includes model-label helpers. |
| `time2event.R` | Core survival-modeling routines shared across cohorts. |
| `plot_figures.R` | Working / diagnostic survival figures. |
| `pub_figures.R` | Publication-ready survival figures. |

These helpers are sourced by the cohort-specific survival scripts. See the root
[`README.md`](../../README.md).
