# `A4/pacc/` — PACC outcome sub-analysis

Models the **Preclinical Alzheimer's Cognitive Composite (PACC)** in the A4 cohort, as
an alternative outcome to the CDR-progression analysis in [`../`](../).

| File | Purpose |
| --- | --- |
| `pacc_tmerge.R` | Time-varying-covariate (`tmerge`) survival-style modeling of PACC over follow-up, across the demographics / Lancet / pTau / centiloid feature sets and CV folds. Writes per-model results and metrics under `results/A4/PACC/tmerge_model/`. |

`pacc_tmerge.R` sources the shared plotting/metric helpers from
[`../../../survival/time2event/`](../../../survival/time2event/) (located via `this.path`,
so it works from any working directory).

> Data-directory paths (`../../../tidy_data/A4/`, `../../../results/A4/PACC/`) are
> environment-specific — adjust to your local layout, as noted in the root
> [`README.md`](../../../README.md).
