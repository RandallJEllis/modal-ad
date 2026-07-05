# `cohorts/` — external validation cohorts

Independent cohorts used to evaluate and extend the UK Biobank findings, primarily via
time-to-event (survival) analysis.

| Folder | Cohort / data |
| --- | --- |
| [`A4/`](A4/) | Anti-Amyloid Treatment in Asymptomatic AD (A4) trial — plasma pTau217 and CDR progression; Cox, time-varying, and joint models. |
| [`ADNI/`](ADNI/) | Alzheimer's Disease Neuroimaging Initiative — plasma pTau. |
| [`pet/`](pet/) | Pooled amyloid-PET across OASIS, NACC, HABS, ADNI, AIBL. |
| [`csf/`](csf/) | NACC cerebrospinal-fluid biomarkers. |
| [`nacc_csf/`](nacc_csf/) | VIF multicollinearity diagnostics for the NACC CSF models. |

Shared survival-metric utilities live in [`../survival/time2event/`](../survival/time2event/).
See the root [`README.md`](../README.md).
