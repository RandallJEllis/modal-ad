# `time2event/` — shared survival-analysis utilities and figures

Common R utilities for the time-to-event analyses across cohorts (UK Biobank, A4,
ADNI, PET, CSF), plus publication figure generation.

| File | Purpose |
| --- | --- |
| `metrics.R` | Time-dependent survival metrics: **Brier score with decomposition** (reliability / resolution / uncertainty), time-dependent AUC, **net reclassification improvement (NRI)** (`nricens`), and **decision-curve analysis** (`rmda`/`dcurves`). Includes model-label helpers. |
| `time2event.R` | Core survival-modeling routines shared across cohorts. |
| `plot_figures.R` | Full survival figure library (18 functions). Sources `plotting_common.R` for the shared helpers, then defines the cohort-specific plots. |
| `plotting_common.R` | Plotting/formatting helpers that are **identical** across the survival analyses (theme, calibration, decision-curve, p-value histogram). Sourced by every `plot_figures.R` so the code lives in one place. |
| `pub_figures.R` | Reduced publication-figure set (a subset of `plot_figures.R`). |

These helpers are sourced by the cohort-specific survival scripts (each locates this
folder via `this.path`, so sourcing works regardless of the caller's working directory).
See the root [`README.md`](../../README.md).

> Note: `plot_figures.R` still has cohort-specific variants under
> [`../../cohorts/A4/cdr/`](../../cohorts/A4/cdr/) (11 functions genuinely diverged and
> were left per-cohort). Only the byte-identical helpers were consolidated here.
