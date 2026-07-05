# `lancet_2024_variables/` — 2024 Lancet Commission risk factors

Extracts the modifiable dementia risk factors described by the **2024 Lancet
Commission on dementia prevention, intervention, and care** from UK Biobank fields,
producing the covariate set used in the `*_lancet2024` experiment arms throughout the
codebase.

| File | Purpose |
| --- | --- |
| `get_variables.py` | Pull and derive the Lancet 2024 risk-factor variables (e.g. education, hypertension, smoking, alcohol, obesity, hearing, depression, diabetes, physical activity, social isolation, air pollution) from UK Biobank. |

The resulting variables feed the `age_sex_lancet2024`, `demographics_and_lancet2024`,
and `demographics_modality_lancet2024` experiments. See the root
[`README.md`](../../README.md).
