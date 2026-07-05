# `A4/pacc/` — PACC outcome sub-analysis

Models the **Preclinical Alzheimer's Cognitive Composite (PACC)** as a continuous
outcome in the A4 cohort, as an alternative to the CDR-progression outcome used
elsewhere in [`../`](../). Three complementary approaches are included.

### Machine-learning regression
| File | Purpose |
| --- | --- |
| `pacc_regression.py` | FLAML/LightGBM **regression** predicting the continuous PACC score, per visit × fold × feature set (demographics, Lancet variables, pTau, centiloids, and combinations). Imports `encode_categorical_vars` / `pull_columns_by_prefix` from [`../../../utils/`](../../../utils/). |
| `pacc.sh` | SLURM array-task worker; runs `pacc_regression.py` for one visit. |
| `call_pacc.sh` | Submits the SLURM array over the 22 visit codes. |

> Note: `pacc_regression.py` was renamed from the original `pacc_binary_classification.py`
> — the task is regression (`task="regression"`, MSE), not classification.

### Longitudinal / trajectory models (R)
| File | Purpose |
| --- | --- |
| `pacc_tmerge.R` | Time-varying-covariate (`tmerge`) modeling of PACC over follow-up. |
| `spline_pacc.R` | Cubic-spline modeling of PACC trajectories (`mgcv`/`splines`). |
| `compare_pacc_rsquared.R` | Mixed-effects comparison (`lme4`/`emmeans`) of cross-validated R² across the spline models. |

The R scripts source the shared plotting/metric helpers from
[`../../../survival/time2event/`](../../../survival/time2event/) (located via `this.path`,
so they work from any working directory).

> Data-directory paths (`../../../tidy_data/A4/`, `../../../raw_data/`,
> `../../../results/A4/PACC/`) are environment-specific — adjust to your local layout, as
> noted in the root [`README.md`](../../../README.md).
