# `proteomics/` — UK Biobank blood proteomics modality

Prediction of incident dementia from UK Biobank Olink blood-plasma proteomics,
combined with demographics and 2024 Lancet Commission risk factors.

| File | Purpose |
| --- | --- |
| `build_ml_datasets.py` | Merge proteomics with demographics + dementia labels, drop prevalent cases at/before the assay visit, encode categoricals, and write `X.parquet` / `y.npy` + region indices. Args: `--data_path`, `--output_path`. |

Run `build_ml_datasets.py` first (Stage 1), then run experiments with the canonical
[`ukbiobank/ml_experiments.py`](../ml_experiments.py) entry point (Stage 2). See the
root [`README.md`](../../README.md).
