# `analysis/` — feature-importance reporting & robustness checks

Post-hoc analyses that interpret and stress-test the primary models (several were added
in response to peer review).

### Feature importance
| Path | Purpose |
| --- | --- |
| `feature_importance_tables.py` | Assemble feature importances saved during the UK Biobank AutoML runs into publication tables/figures. |

Feature importances are extracted by `ukbiobank/ml_experiments.py` and its runner,
`ukbiobank/loop_ml.sh`; this directory contains the reporting script, not a separate
retraining workflow.

### Subgroup heterogeneity
| Script | Purpose |
| --- | --- |
| `heterogeneity_analysis.py` | Subgroup performance + DerSimonian–Laird heterogeneity (see [`../docs/heterogeneity_analysis.md`](../docs/heterogeneity_analysis.md)). |
| `plot_heterogeneity_comparison.py` | Compare heterogeneity across two experiments. |

### Multicollinearity (VIF)
| Script | Purpose |
| --- | --- |
| `vif_maximal_models.py` | VIF / conditioning diagnostics for the ML feature matrix. |
| `vif_utils.R` | Shared VIF helpers for the Cox models (used by the per-cohort `vif_model_check.R`). |
| `make_vif_reviewer_tables.py` | Format VIF results into Word tables for the reviewer response. |

See the root [`README.md`](../README.md).
