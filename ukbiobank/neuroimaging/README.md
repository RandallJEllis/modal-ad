# `neuroimaging/` — UK Biobank neuroimaging modality

Prediction of incident dementia from UK Biobank brain imaging-derived phenotypes
(IDPs), combined with demographics and 2024 Lancet Commission risk factors.

| File | Purpose |
| --- | --- |
| `build_ml_datasets.py` | Merge brain IDPs (UKB **data instance 2**) with demographics + dementia labels, drop prevalent cases, encode categoricals, and write `X.parquet` / `y.npy`. |
| `old/` | Legacy feature-selection and experiment scripts, retained for provenance. |

Unlike proteomics/cognitive tests (which use assessment-centre region hold-outs),
neuroimaging models are evaluated with **10-fold stratified cross-validation**
(`region_index` selects the fold). Run `build_ml_datasets.py` first, then the root
pipeline (Stage 2). See the root [`README.md`](../../README.md).
