# UK Biobank Heterogeneity Analysis

This document explains `heterogeneity_analysis.py`, the post hoc analysis script used
to assess reviewer-requested subgroup heterogeneity in UK Biobank prediction results.

## Purpose

The script uses saved UK Biobank test-set predictions and labels to evaluate whether
model performance varies across clinically relevant patient strata:

- age bands
- sex
- APOE polymorphism encoding

For each outcome/modality/model result, it reconstructs the original held-out test
patients, attaches subgroup labels, calculates subgroup performance, and reports
formal DerSimonian-Laird heterogeneity statistics.

## Inputs

The default analysis reads:

- saved probabilities:
  `results/UKBiobank/{outcome}/{modality}/demographics_modality_lancet2024/log_loss/lgbm/agecutoff_65/test_probas_region_*.pkl`
- saved labels:
  `results/UKBiobank/{outcome}/{modality}/demographics_modality_lancet2024/log_loss/lgbm/agecutoff_65/test_true_labels_region_*.pkl`
- patient metadata:
  `tidy_data/UKBiobank/dementia/{modality}/X.parquet`
- outcome labels:
  `tidy_data/UKBiobank/dementia/{modality}/y.npy`
- Alzheimer-only case filtering:
  `../proj_idp/tidy_data/acd/allcausedementia.parquet`

The script reconstructs test splits using the same logic as `ml_experiments.py`:

- neuroimaging: 10-fold stratified CV, using UKB data instance 2
- proteomics and cognitive tests: assessment-center region holdouts, using UKB data instance 0

Strict label checking is on by default. If reconstructed labels do not exactly match
the saved `test_true_labels_region_*.pkl`, the script raises an error instead of
writing potentially misaligned subgroup results.

## Running

Default run:

```bash
python3 heterogeneity_analysis.py
```

Run with five-year age bands:

```bash
python3 heterogeneity_analysis.py --age-bins 65 70 75 80 85
```

The age-bin cutpoints are lower bounds. For example, `--age-bins 65 70 75 80 85`
creates left-closed, right-open bands:

- `65-70`: age >= 65 and < 70
- `70-75`: age >= 70 and < 75
- `75-80`: age >= 75 and < 80
- `80-85`: age >= 80 and < 85
- `85+`: age >= 85

Useful options:

- `--outcomes dementia alzheimers`
- `--modalities proteomics neuroimaging cognitive_tests`
- `--experiment demographics_modality_lancet2024`
- `--metric log_loss`
- `--model lgbm`
- `--age-cutoff 65`
- `--age-cutoff none` to use all-age result folders
- `--dca-threshold-min 0.01`
- `--dca-threshold-max 0.50`
- `--dca-threshold-step 0.01`
- `--no-plots`
- `--no-strict-label-check`

## Outputs

Outputs are written under:

```text
results/UKBiobank/formal_heterogeneity/{experiment}/{metric}/{model}/{age_cutoff_label}/
```

If non-default age bins are used, the script appends an extra folder such as:

```text
agebins_65_70_75_80_85_plus/
```

Each run writes:

- `manifest.json`: run configuration and paths
- `oof_predictions_with_subgroups.csv`: one row per held-out prediction with subgroup labels
- `subgroup_metrics.csv`: subgroup AUC and Brier score with standard errors and 95% CIs
- `decision_curves.csv`: subgroup decision-curve net benefit by threshold
- `heterogeneity_summary.csv`: DerSimonian-Laird heterogeneity summaries
- `decision_curve_plots/*.png`: decision-curve plots by outcome, modality, and subgroup type

## Subgroup Encodings

Age uses the UKB age-at-assessment field from the same instance as the training run:

- instance 0 for proteomics and cognitive tests
- instance 2 for neuroimaging

Sex is decoded from `31-0.0_1.0` as:

- `1`: Male
- `0`: Female

APOE is reported as the raw one-hot encoded `apoe_polymorphism` category available
in the saved ML feature tables:

- `apoe_polymorphism=0.0`
- `apoe_polymorphism=1.0`
- `apoe_polymorphism=2.0`
- `apoe_polymorphism=missing`

Do not interpret these labels as full APOE genotype labels unless they are mapped
back to the underlying SNP fields in a separate genotype-specific analysis.

## Statistical Notes

`subgroup_metrics.csv`:

- AUC is computed within each subgroup. It is undefined when the subgroup has only
  cases or only controls.
- AUC standard errors use DeLong variance when possible, with a Hanley-McNeil
  fallback for very small case/control counts.
- Brier score is the mean squared error of predicted probabilities against binary
  outcomes. Lower is better.

`decision_curves.csv`:

- Net benefit is computed as `TP/n - FP/n * threshold/(1 - threshold)`.
- Compare `net_benefit` with `net_benefit_treat_all` and `net_benefit_treat_none`.
- A positive model net benefit above both reference strategies supports clinical
  utility at that threshold.

`heterogeneity_summary.csv`:

- Heterogeneity is calculated across subgroup estimates within each
  outcome/modality/subgroup_type/metric combination.
- `k` is the number of valid subgroup estimates used.
- `i2` is the percent of total variability attributable to between-subgroup
  heterogeneity rather than sampling error.
- `tau2` is the between-subgroup variance.
- `tau` is the square root of `tau2`, on the metric scale. For AUC, `tau` is in
  AUC units; for Brier, in Brier units; for net benefit, in net-benefit units.
- `q` and `q_pvalue` are Cochran Q-test statistics.
- `fixed_effect` and `random_effect` are pooled estimates.
- If `k < 2`, `i2`, `tau`, and Q-test values are not estimable and are reported
  as missing.

