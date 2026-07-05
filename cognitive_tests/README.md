# `cognitive_tests/` — UK Biobank cognitive tests modality

Prediction of incident dementia from UK Biobank cognitive-test measures, combined with
demographics and 2024 Lancet Commission risk factors.

| File | Purpose |
| --- | --- |
| `build_ml_datasets.py` | Merge cognitive-test features with demographics + dementia labels, drop prevalent cases, encode categoricals, and write `X.parquet` / `y.npy` + region indices. |
| `old/` | Legacy feature-selection and experiment scripts, retained for provenance. |

Evaluated with assessment-centre region hold-outs (`region_index`), like proteomics.
Run `build_ml_datasets.py` first, then the root pipeline (Stage 2). See the root
[`README.md`](../README.md).
