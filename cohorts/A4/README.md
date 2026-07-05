# `A4/` — Anti-Amyloid Treatment in Asymptomatic Alzheimer's (A4) study

Time-to-event analysis in the A4 secondary-prevention cohort, using plasma **pTau217**
and **Clinical Dementia Rating (CDR)** progression. Cases are defined by CDR ≥ 0.5 on
two consecutive visits.

### Dataset build
| File | Purpose |
| --- | --- |
| `build_datasets.py` | Process pTau217 and CDR measurements; define cases from CDR progression; compute time-to-event. |

### Survival model
| File | Purpose |
| --- | --- |
| `tvcox.R` | Time-varying-covariate Cox proportional-hazards model — the A4 time-to-event model (run interactively in R). |

### `cdr/` sub-analysis
CDR-outcome time-to-event analysis: `tvcox_CDR.R` (model), `metrics.R`,
`plot_figures.R`, `generate_plots_tables.R` (evaluation/figures), and
`vif_model_check.R` (multicollinearity diagnostics).

See the root [`README.md`](../../README.md) and [`../../survival/time2event/`](../../survival/time2event/) for the
shared survival-metric utilities.
