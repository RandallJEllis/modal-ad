# `proteomics/` — UK Biobank blood proteomics modality

Prediction of incident dementia from UK Biobank Olink blood-plasma proteomics,
combined with demographics and 2024 Lancet Commission risk factors.

| File | Purpose |
| --- | --- |
| `build_ml_datasets.py` | Merge proteomics with demographics + dementia labels, drop prevalent cases at/before the assay visit, encode categoricals, and write `X.parquet` / `y.npy` + region indices. Args: `--data_path`, `--output_path`. |
| `ml_experiments.py` | Modality-scoped copy of the AutoML classifier (see the canonical [`../ml_experiments.py`](../ml_experiments.py)). |
| `feature_selection_experiments.py` | Feature-selection (`fs_*`) experiment variant. |
| `flaml_test.py` | Minimal FLAML sanity check. |
| `loop_ml.sh`, `sh_ml_experiments.sh`, `fs_loop_ml.sh`, `fs_sh_ml_experiments.sh` | SLURM submission scripts for the modality (and its feature-selection runs). |
| `3d_sens_spec_prev.png`, `conf_mtx_workflow_figure_panel.pdf` | Example workflow / results figures. |

Run `build_ml_datasets.py` first (Stage 1), then submit experiments via the root
pipeline (Stage 2). See the root [`README.md`](../../README.md).
