# `utils/` — shared project library

Python utility modules imported across the codebase. Consuming scripts locate this
folder automatically — each walks up the directory tree to find `utils/` and adds it to
the import path — then `import <module>` directly. There is no `__init__.py`; modules
are imported by name, not as a package. It lives at the repository root because it is
shared, though (as noted below) most of its modules were written for the UK Biobank
pipeline.

## Where the code is used

The **UK Biobank pipeline** (`ukbiobank/`) uses essentially all of these modules. A
**subset** is also imported by the external-cohort and robustness scripts. If you are
working only on the non-UK-Biobank cohorts, these are the only modules you need:

| Module | Imported by (outside `ukbiobank/`) |
| --- | --- |
| `ml_utils.py` | `cohorts/nacc_csf/build_csf_datasets.py`, `cohorts/A4/pacc/pacc_regression.py`, `analysis/feature_importance/retrain_extract_fi.py` |
| `plot_results.py` | `analysis/feature_importance_tables.py`, `analysis/feature_importance/retrain_extract_fi.py` |
| `df_utils.py` | `cohorts/A4/pacc/pacc_regression.py`, `analysis/feature_importance/retrain_extract_fi.py` |
| `utils.py` | `analysis/feature_importance/retrain_extract_fi.py` |
| `ukb_utils.py` | `analysis/heterogeneity_analysis.py`, `analysis/feature_importance_tables.py`, `analysis/feature_importance/retrain_extract_fi.py` |
| `dementia_utils.py` | `analysis/heterogeneity_analysis.py` |
| `icd.py` | `analysis/feature_importance/retrain_extract_fi.py` |
| `f3.py` | `analysis/feature_importance/retrain_extract_fi.py` |

> Note: among the external cohorts, only `cohorts/nacc_csf/build_csf_datasets.py`
> (`ml_utils`) and `cohorts/A4/pacc/pacc_regression.py` (`ml_utils`, `df_utils`) import
> this Python library. The rest of the cohort analysis is R-based and relies on the
> shared R helpers in [`../survival/time2event/`](../survival/time2event/) instead.

## Modules

### General-purpose (reusable across cohorts)
These are not tied to UK Biobank data structures and are the ones most often reused by
the external cohorts and robustness analyses.

| Module | Purpose |
| --- | --- |
| `ml_utils.py` | Core ML metrics and result I/O: `calc_results` (threshold selection, sensitivity/specificity/PPV/NPV/MCC), `save_labels_probas`, plus `brier_decomp`, NRI (`calculate_nri_from_paths`), decision-curve analysis (`decision_curve_analysis*`), and regression metrics. |
| `plot_results.py` | Figure generation: ROC/PR/calibration curves, MCC raincloud plots, Brier-decomposition strip plots. |
| `df_utils.py` | Generic DataFrame helpers (e.g. `pull_columns_by_prefix`). |
| `f3.py` | F3-score metric (`f3_metric`) used as a FLAML optimization objective. |
| `utils.py` | Miscellaneous I/O helpers (`save_pickle`, folder-existence checks). |

### UK Biobank–specific
These encode UK Biobank data structures, field codings, and cohort-definition logic.
They are written for the UK Biobank pipeline; a few of their helpers are reused by the
external-cohort/robustness scripts above (e.g. `ukb_utils` for shared formatting), but
their core content is UK-Biobank–specific.

| Module | Purpose |
| --- | --- |
| `ukb_utils.py` | UK Biobank field helpers, including assessment-centre region grouping (`group_assessment_center`) used to build the region hold-out cross-validation. |
| `dementia_utils.py` | Construction of UK Biobank dementia / Alzheimer's case labels and follow-up windows from the diagnosis fields. |
| `icd.py` | ICD-9/10 diagnosis-code handling for UK Biobank first-occurrence/HES fields. |

### Other
| File | Purpose |
| --- | --- |
| `requirements.txt` | Full frozen conda specification (linux-64) of the original HPC environment. |

See the root [`README.md`](../README.md) for the overall pipeline.
