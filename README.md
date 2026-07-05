# Multimodal Prediction of Dementia and Alzheimer's Disease

Code accompanying the study of machine-learning prediction of incident dementia and
Alzheimer's disease (AD) from multiple data modalities in the **A4**, **ADNI**, 
multi-cohort **PET**, **NACC CSF**, and **UK Biobank** cohorts.

> **Associated publication.** _Add full citation, journal, year, and DOI here once
> the paper is published_ (see [`CITATION.cff`](CITATION.cff)). Please cite the paper
> if you use this code.

---

## 1. Overview

The project asks how well incident Alzheimer's (and, as a secondary outcome, 
all-cause dementia) can be predicted from different classes of measurement, and how
those predictions behave across clinically relevant subgroups.

- **Data modalities:** blood biomarkers, amyloid PET, cerebrospinal fluid markers,
  proteomics, neuroimaging (brain IDPs), and
  cognitive tests, each combined with demographics, APOE genotype, and the modifiable risk
  factors from the 2024 Lancet Commission on dementia.
- **Models:** Cox survival models, gradient-boosted trees (**LightGBM**) and
  **L1-regularized logistic regression**, tuned with the
  [FLAML](https://microsoft.github.io/FLAML/) AutoML framework. Discrimination is optimized
  on log loss.
- **Two prediction framings:**
  1. **Cross-sectional classification** — predict whether a participant will be
     diagnosed within the follow-up window (`ml_experiments.py`).
  2. **Time-to-event / survival analysis** — model time to diagnosis with Cox
     proportional-hazards, time-varying-covariate, and joint longitudinal-survival
     models (`timetoevent_experiments.py`, the `time2event/` R suite, and the
     per-cohort `tv*.R` scripts).
- **Cohorts:** ADNI, the A4 secondary-prevention trial (pTau217 + Clinical Dementia Rating
  progression), a pooled five-cohort PET analysis (OASIS, NACC, HABS, ADNI, AIBL),
  NACC cerebrospinal-fluid (CSF) biomarkers, and UK Biobank for proteomics, multi-modal
  brain imaging, and cognitive tests.
- **Post-hoc / robustness analyses:** subgroup **heterogeneity** (DerSimonian–Laird),
  **variance-inflation-factor (VIF)** multicollinearity diagnostics, **Brier-score
  decomposition** (reliability / resolution / uncertainty), **net reclassification
  improvement (NRI)**, and **decision-curve analysis**.

---

## 2. Repository structure

```text
.
├── README.md                     ← this file
├── CITATION.cff                  ← how to cite (fill in once published)
├── requirements.txt              ← core Python dependencies (pip)
├── environment.yml               ← conda environment (Python)
├── ukb_func/                     ← shared Python library (imported everywhere)
│   ├── README.md
│   ├── requirements.txt          ← full frozen conda spec (linux-64)
│   └── *.py                       (ml_utils, dementia_utils, ukb_utils, df_utils, …)
│
│   ── UK Biobank main pipeline (repo root) ──
├── ml_experiments.py             ← cross-sectional AutoML classifier (core script)
├── sh_ml_experiments.sh          ← SLURM sbatch wrapper for one ml_experiments run
├── loop_ml.sh                    ← submits the full grid of ml_experiments jobs
├── timetoevent_experiments.py    ← survival/time-to-event AutoML variant
├── sh_timetoevent_experiments.sh
├── loop_timetoevent.sh
│
│   ── UK Biobank data modalities ──
├── proteomics/                   ← build_ml_datasets.py + experiment scripts + figures
├── neuroimaging/                 ← build_ml_datasets.py (brain IDPs)
├── cognitive_tests/              ← build_ml_datasets.py
├── lancet_2024_variables/        ← extract 2024 Lancet Commission risk factors
│
│   ── External validation cohorts ──
├── ADNI/                         ← ADNI build + survival + plots
├── A4/                           ← A4 trial: pTau217, CDR progression, joint models
│   └── cdr/                       (CDR-based time-to-event sub-analysis)
├── pet/                          ← pooled PET amyloid across 5 cohorts
├── csf/                          ← NACC CSF biomarker datasets + survival
├── nacc_csf/                     ← NACC CSF VIF diagnostics
│
│   ── Survival analysis (shared R) ──
├── time2event/                   ← metrics.R, plotting, publication figures
│
│   ── Feature importance & robustness ──
├── feature_importance/           ← retrain best models and extract importances
├── feature_importance_tables.py  ← assemble importance tables/figures
├── heterogeneity_analysis.py     ← subgroup heterogeneity (see its README)
├── heterogeneity_analysis_README.md
├── plot_heterogeneity_comparison.py
├── vif_maximal_models.py         ← VIF for the ML feature matrix
├── vif_utils.R                   ← shared VIF helpers (Cox models)
└── make_vif_reviewer_tables.py   ← formats VIF results into Word tables
```

Most directories contain their own `README.md` with details; start there for any
specific analysis.

---

## 3. Data access

The raw and derived data are **not** included in this repository and must be obtained
from the respective data providers under their access agreements:

| Source | Access |
| --- | --- |
| UK Biobank | Approved application via <https://www.ukbiobank.ac.uk/> |
| ADNI | <https://adni.loni.usc.edu/> |
| A4 | <https://www.a4studydata.org/> |
| OASIS / NACC / HABS / AIBL (PET) | Respective data-use agreements |

### Expected data layout

Scripts reference data through **relative paths anchored near the repository**, using
directories such as `tidy_data/`, `results/`, `metadata/`, and an adjacent
`proj_idp/`. The conventional layout places these as siblings of the code checkout:

```text
<workspace>/
├── modal-ad/            ← this repository
├── tidy_data/           ← model-ready datasets (X.parquet, y.npy, …)
├── results/             ← model outputs (probabilities, metrics, figures)
├── metadata/            ← UKB field lookups (e.g. coding10.tsv)
└── proj_idp/            ← upstream tidy_data source (e.g. allcausedementia.parquet)
```

> ⚠️ **Path caveat.** These scripts were developed across several HPC environments,
> and some relative paths / data-subfolder names (e.g. `tidy_data/dementia` vs
> `tidy_data/UKBiobank/dementia`) are inconsistent between scripts. Before running,
> verify the `--data_path` / `--output_path` / root arguments against your local
> layout. Where possible, paths are exposed as command-line arguments.

---

## 4. Environment setup

### Python

```bash
# Option A — conda (recommended)
conda env create -f environment.yml
conda activate modal-ad

# Option B — pip
pip install -r requirements.txt
```

Core Python stack: `scikit-learn`, `pandas`, `numpy`, `flaml`, `lightgbm`,
`scikit-survival`, `lifelines`, `pyarrow`, `matplotlib`, `seaborn`, `scipy`,
`python-docx`. A complete frozen conda specification (linux-64) used on the original
HPC systems is preserved at [`ukb_func/requirements.txt`](ukb_func/requirements.txt).

### R (survival analysis and figures)

The `*.R` scripts require R (≥ 4.2). Key packages:

```r
install.packages(c(
  "tidyverse", "survival", "survminer", "riskRegression", "pec", "timeROC",
  "pROC", "yardstick", "nricens", "rmda", "JMbayes2", "mice", "nlme",
  "arrow", "ggplot2", "patchwork", "cowplot", "xtable", "this.path"
))
```

---

## 5. How to reproduce the analyses

The pipeline runs in four stages. Steps 2–4 are independent given the built datasets.

### Stage 1 — Build model-ready datasets

Each modality/cohort has a build script that merges the modality measurements with
demographics and dementia labels, removes prevalent cases, encodes categoricals, and
writes `X.parquet` / `y.npy` plus cross-validation indices.

```bash
python proteomics/build_ml_datasets.py       --data_path <...> --output_path <...>
python neuroimaging/build_ml_datasets.py      --data_path <...> --output_path <...>
python cognitive_tests/build_ml_datasets.py   --data_path <...> --output_path <...>
python A4/build_datasets.py                    # + ADNI/, pet/, csf/ build scripts
```

### Stage 2 — Cross-sectional prediction (UK Biobank)

`ml_experiments.py` trains one model for one (modality, experiment, metric, model,
age cutoff, region) combination. `loop_ml.sh` submits the full grid via SLURM.

```bash
python ml_experiments.py \
  --modality proteomics \
  --experiment demographics_modality_lancet2024 \
  --model lgbm --metric log_loss --age_cutoff 65 --region_index 0
```

- **modality:** `proteomics`, `neuroimaging`, `cognitive_tests`
- **experiment:** `age_only`, `all_demographics`, `age_sex_lancet2024`,
  `demographics_and_lancet2024`, `modality_only`, `demographics_and_modality`,
  `demographics_modality_lancet2024`, and their `fs_` feature-selection variants
- **model:** `lgbm`, `lrl1` · **metric:** `log_loss`, `roc_auc`, `f3`, `ap`
- **age_cutoff:** `0` (all ages) or `65`
- **region_index:** assessment-centre region hold-out (proteomics/cognitive) or CV
  fold (neuroimaging)

Outputs (probabilities, labels, per-region metrics) are written under
`results/UKBiobank/{outcome}/{modality}/{experiment}/{metric}/{model}/{age_cutoff}/`.

### Stage 3 — Time-to-event / survival

`timetoevent_experiments.py` mirrors Stage 2 for survival outcomes (random survival
forests + AutoML). The cohort-specific R scripts (`A4/`, `ADNI/`, `pet/`, `csf/`,
`time2event/`) fit Cox, time-varying-covariate, and joint models and produce the
publication survival figures and metrics (including Brier decomposition, NRI, and
decision curves via `time2event/metrics.R`).

### Stage 4 — Feature importance and robustness

```bash
bash feature_importance/loop_fi.sh          # retrain best models, extract importances
python feature_importance_tables.py         # assemble importance tables/figures
python heterogeneity_analysis.py            # subgroup heterogeneity (see its README)
python vif_maximal_models.py                # VIF diagnostics for the ML feature matrix
```

See [`heterogeneity_analysis_README.md`](heterogeneity_analysis_README.md) for the
full heterogeneity workflow and output schema.

---

## 6. Compute environment

The experiment grids were run on SLURM HPC clusters. The `loop_*.sh` scripts build
the parameter grid and submit one `sbatch sh_*.sh` job per combination. To run a
single job locally, call the underlying Python script directly (as in Stage 2) and
ignore the `sbatch`/SLURM directives.

---

## 7. Citation

If you use this code, please cite the associated paper (see [`CITATION.cff`](CITATION.cff)).

## 8. License

Released under the **MIT License** — see [`LICENSE`](LICENSE). You are free to use,
modify, and redistribute the code with attribution. Note that this license covers the
**code only**; the underlying study data remain governed by each provider's data-use
agreement (see §3).

## 9. Contact

Randall J. Ellis — questions and issues via the repository's GitHub Issues page.
