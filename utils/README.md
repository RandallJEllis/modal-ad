# `utils/` — shared project library

Utility modules imported across the whole codebase (UK Biobank pipeline, external
cohorts, and robustness analyses). Consuming scripts locate this folder automatically —
each walks up the directory tree to find `utils/` and adds it to the import path — then
`import <module>` directly. There is no `__init__.py`; modules are imported by name,
not as a package. It lives at the repository root because it is shared, not
UKB-specific.

| Module | Purpose |
| --- | --- |
| `ml_utils.py` | Core ML metrics and result I/O: `calc_results` (threshold selection, sensitivity/specificity/PPV/NPV/MCC), `save_labels_probas`, plus `brier_decomp`, NRI (`calculate_nri_from_paths`), decision-curve analysis (`decision_curve_analysis*`), and regression metrics. |
| `dementia_utils.py` | Construction of dementia / Alzheimer's case labels and follow-up windows. |
| `ukb_utils.py` | UK Biobank field helpers, including assessment-centre region grouping (`group_assessment_center`). |
| `df_utils.py` | DataFrame helpers (e.g. `pull_columns_by_prefix`). |
| `icd.py` | ICD diagnosis-code handling. |
| `f3.py` | F3-score metric (`f3_metric`) used as a FLAML optimization objective. |
| `plot_results.py` | Figure generation: ROC/PR/calibration curves, MCC raincloud plots, Brier-decomposition strip plots. |
| `bootstrap.py` | Bootstrap confidence intervals. |
| `utils.py` | Miscellaneous helpers (`save_pickle`, folder checks). |
| `requirements.txt` | Full frozen conda specification (linux-64) of the original HPC environment. |

These modules are dependencies of nearly every analysis script; see the root
[`README.md`](../README.md) for the overall pipeline.
