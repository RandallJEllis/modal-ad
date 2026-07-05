# `A4/` — Anti-Amyloid Treatment in Asymptomatic Alzheimer's (A4) study

Time-to-event analysis in the A4 secondary-prevention cohort, using plasma **pTau217**
and **Clinical Dementia Rating (CDR)** progression. Cases are defined by CDR ≥ 0.5 on
two consecutive visits.

### Dataset build
| File | Purpose |
| --- | --- |
| `build_datasets.py` | Process pTau217 and CDR measurements; define cases from CDR progression; compute time-to-event. |

### Survival models
| File | Purpose |
| --- | --- |
| `tvcox.R` | Time-varying-covariate Cox proportional-hazards model. |
| `two_cox.R` | Two-model / comparative Cox specification. |
| `t2e.py`, `t2e_final.R` | Time-to-event modeling and final analysis. |
| `timevarycovars_jointmodel.R`, `chatgpt_timevarycovars_jointmodel.R` | Joint longitudinal–survival models (biomarker trajectory + event), via `JMbayes2`. |
| `locf_vs_jointmodeling.R` | Compare last-observation-carried-forward against joint modeling of the longitudinal biomarker. |
| `mediation.R` | Mediation analysis. |
| `loop_t2e.sh`, `sh_t2e.sh` | SLURM submission scripts. |
| `*_summary.txt` | Saved fitted-model summaries (baseline Cox, joint model). |

### `cdr/` sub-analysis
CDR-outcome time-to-event analysis: `tvcox_CDR.R` (model), `metrics.R`,
`plot_figures.R`, `generate_plots_tables.R` (evaluation/figures), and
`vif_model_check.R` (multicollinearity diagnostics).

See the root [`README.md`](../../README.md) and [`../../survival/time2event/`](../../survival/time2event/) for the
shared survival-metric utilities.
