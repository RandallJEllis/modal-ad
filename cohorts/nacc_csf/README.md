# `nacc_csf/` — VIF diagnostics for the NACC CSF models

| File | Purpose |
| --- | --- |
| `vif_model_check.R` | Variance-inflation-factor multicollinearity diagnostics for the NACC CSF Cox models. |

The CSF dataset build and survival models are in [`../csf/`](../csf/). This folder
holds only the reviewer-requested collinearity check; see
[`../../analysis/vif_utils.R`](../../analysis/vif_utils.R) for the shared VIF helpers.
