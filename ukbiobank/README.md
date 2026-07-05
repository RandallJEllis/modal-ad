# `ukbiobank/` — UK Biobank primary pipeline

The main modeling cohort. Contains the core experiment driver and the
modality-specific dataset builders. UK Biobank prediction is **cross-sectional
classification only** — time-to-event modeling is done for the external cohorts (see
[`../cohorts/`](../cohorts/)), not here.

### Core scripts
| Script | Purpose |
| --- | --- |
| `ml_experiments.py` | Cross-sectional AutoML classifier (one model per modality × experiment × metric × model × age-cutoff × region). |
| `sh_ml_experiments.sh`, `loop_ml.sh` | SLURM wrapper and full-grid submitter for `ml_experiments.py`. |

### Data-modality subfolders
| Folder | Modality |
| --- | --- |
| [`proteomics/`](proteomics/) | Olink blood-plasma proteomics |
| [`neuroimaging/`](neuroimaging/) | Brain imaging-derived phenotypes (IDPs) |
| [`cognitive_tests/`](cognitive_tests/) | Cognitive-test measures |
| [`lancet_2024_variables/`](lancet_2024_variables/) | 2024 Lancet Commission modifiable risk factors |

Run each modality's `build_ml_datasets.py` first, then the core scripts. See the root
[`README.md`](../README.md) for the full pipeline and CLI options.
