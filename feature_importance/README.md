# `feature_importance/` — model feature-importance extraction

Retrains the best models selected during AutoML and extracts feature importances for
interpretation and reporting.

| File | Purpose |
| --- | --- |
| `retrain_extract_fi.py` | Retrain the best model per experiment (from the saved AutoML log) and extract/serialize feature importances. |
| `loop_fi.sh` | Submit the retrain-and-extract grid via SLURM. |
| `sh_fi.sh` | SLURM sbatch wrapper for a single `retrain_extract_fi.py` run. |

Importances are aggregated into tables/figures by the root
[`../feature_importance_tables.py`](../feature_importance_tables.py). See the root
[`README.md`](../README.md).
