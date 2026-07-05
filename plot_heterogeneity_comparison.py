"""
Compare UK Biobank subgroup heterogeneity results across two experiments.

The default comparison is:

- demographics_and_lancet2024
- demographics_modality_lancet2024

using the agecutoff_65, five-year age-band heterogeneity outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_EXPERIMENTS = (
    "demographics_and_lancet2024",
    "demographics_modality_lancet2024",
)
EXPERIMENT_LABELS = {
    "demographics_and_lancet2024": "Demog + Lancet",
    "demographics_modality_lancet2024": "Demog + modality + Lancet",
}
EXPERIMENT_COLORS = {
    "Demog + Lancet": "#4C78A8",
    "Demog + modality + Lancet": "#F58518",
}
EXPERIMENT_SHORT_LABELS = {
    "Demog + Lancet": "Demog",
    "Demog + modality + Lancet": "Modality+",
}
MODALITY_LABELS = {
    "proteomics": "Proteomics",
    "neuroimaging": "IDPs",
    "cognitive_tests": "Cognitive tests",
}
MODALITY_ORDER = ("Proteomics", "IDPs", "Cognitive tests")
OUTCOME_LABELS = {
    "dementia": "All-cause dementia",
    "alzheimers": "Alzheimer's disease",
}
SHORT_OUTCOME_LABELS = {
    "dementia": "Dementia",
    "alzheimers": "AD",
}
OUTCOME_ORDER = ("dementia", "alzheimers")
SUBGROUP_TYPE_LABELS = {
    "age_band": "Age band",
    "sex": "Sex",
    "apoe_polymorphism": "APOE genotype",
}
SHORT_SUBGROUP_TYPE_LABELS = {
    "age_band": "Age",
    "sex": "Sex",
    "apoe_polymorphism": "APOE",
}
SUBGROUP_TYPE_ORDER = ("age_band", "sex", "apoe_polymorphism")
AGE_BAND_ORDER = ("65-70", "70-75", "75-80", "80-85", "85+")
SEX_ORDER = ("Female", "Male", "Sex missing")
APOE_DISPLAY_ORDER = ("0", "1", "2", "missing")
APOE_REVERSED_MODALITIES = {"Proteomics"}
OOF_REQUIRED_COLUMNS = (
    "outcome",
    "modality",
    "region_index",
    "region",
    "age_band",
    "sex",
    "apoe_polymorphism",
    "y_true",
    "y_prob",
)


def age_bins_label(age_bins: tuple[int, ...]) -> str:
    return f"agebins_{'_'.join(str(value) for value in age_bins)}_plus"


def experiment_dir(
    results_root: Path,
    experiment: str,
    metric: str,
    model: str,
    age_cutoff: int,
    age_bins: tuple[int, ...],
) -> Path:
    return (
        results_root
        / experiment
        / metric
        / model
        / f"agecutoff_{age_cutoff}"
        / age_bins_label(age_bins)
    )


def apoe_display_value(modality_label: str | None, subgroup: str) -> str:
    value = subgroup.replace("apoe_polymorphism=", "")
    if value == "missing":
        return value
    try:
        allele_count = int(float(value))
    except ValueError:
        return value

    # The saved proteomics matrix encodes the rs429358 reference allele count,
    # so 2/1/0 corresponds to 0/1/2 e4 alleles for display.
    if modality_label in APOE_REVERSED_MODALITIES:
        allele_count = 2 - allele_count
    return str(allele_count)


def subgroup_sort_value(
    subgroup_type: str,
    subgroup: str,
    modality_label: str | None = None,
) -> int:
    if subgroup_type == "age_band":
        return AGE_BAND_ORDER.index(subgroup) if subgroup in AGE_BAND_ORDER else 999
    if subgroup_type == "sex":
        return SEX_ORDER.index(subgroup) if subgroup in SEX_ORDER else 999
    if subgroup_type == "apoe_polymorphism":
        display_value = apoe_display_value(modality_label, subgroup)
        return (
            APOE_DISPLAY_ORDER.index(display_value)
            if display_value in APOE_DISPLAY_ORDER
            else 999
        )
    return 999


def clean_subgroup_label(
    subgroup_type: str,
    subgroup: str,
    modality_label: str | None = None,
) -> str:
    if subgroup_type == "apoe_polymorphism":
        display_value = apoe_display_value(modality_label, subgroup)
        if display_value == "missing":
            return "missing e4 alleles"
        suffix = "e4 allele" if display_value == "1" else "e4 alleles"
        return f"{display_value} {suffix}"
    return subgroup


def stratum_label(
    subgroup_type: str,
    subgroup: str,
    modality_label: str | None = None,
) -> str:
    if subgroup_type == "apoe_polymorphism":
        return f"APOE genotype, {clean_subgroup_label(subgroup_type, subgroup, modality_label)}"
    return (
        f"{SUBGROUP_TYPE_LABELS.get(subgroup_type, subgroup_type)}: "
        f"{clean_subgroup_label(subgroup_type, subgroup, modality_label)}"
    )


def load_comparison_tables(
    results_root: Path,
    experiments: tuple[str, str],
    metric: str,
    model: str,
    age_cutoff: int,
    age_bins: tuple[int, ...],
    include_decision_curves: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_frames = []
    heterogeneity_frames = []
    decision_curve_frames = []
    oof_frames = []
    for experiment in experiments:
        input_dir = experiment_dir(
            results_root=results_root,
            experiment=experiment,
            metric=metric,
            model=model,
            age_cutoff=age_cutoff,
            age_bins=age_bins,
        )
        metrics_path = input_dir / "subgroup_metrics.csv"
        heterogeneity_path = input_dir / "heterogeneity_summary.csv"
        oof_path = input_dir / "oof_predictions_with_subgroups.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing subgroup metrics: {metrics_path}")
        if not heterogeneity_path.exists():
            raise FileNotFoundError(f"Missing heterogeneity summary: {heterogeneity_path}")
        if not oof_path.exists():
            raise FileNotFoundError(f"Missing OOF predictions: {oof_path}")
        if include_decision_curves:
            decision_curves_path = input_dir / "decision_curves.csv"
            if not decision_curves_path.exists():
                raise FileNotFoundError(f"Missing decision curves: {decision_curves_path}")

        metrics = pd.read_csv(metrics_path)
        heterogeneity = pd.read_csv(heterogeneity_path)
        oof = pd.read_csv(oof_path, usecols=list(OOF_REQUIRED_COLUMNS))
        frames = [metrics, heterogeneity, oof]
        if include_decision_curves:
            decision_curves = pd.read_csv(decision_curves_path)
            frames.append(decision_curves)
        for frame in frames:
            frame["experiment"] = experiment
            frame["experiment_label"] = EXPERIMENT_LABELS.get(experiment, experiment)
            frame["modality_label"] = frame["modality"].map(MODALITY_LABELS).fillna(
                frame["modality"]
            )
        metric_frames.append(metrics)
        heterogeneity_frames.append(heterogeneity)
        if include_decision_curves:
            decision_curve_frames.append(decision_curves)
        oof_frames.append(oof)

    decision_curves = (
        pd.concat(decision_curve_frames, ignore_index=True)
        if include_decision_curves
        else pd.DataFrame()
    )
    return (
        pd.concat(metric_frames, ignore_index=True),
        pd.concat(heterogeneity_frames, ignore_index=True),
        decision_curves,
        pd.concat(oof_frames, ignore_index=True),
    )


def mean_subgroup_n(
    metrics: pd.DataFrame,
    modality_label: str,
    outcome: str,
    subgroup_type: str,
) -> float:
    data = metrics[
        (metrics["modality_label"] == modality_label)
        & (metrics["outcome"] == outcome)
        & (metrics["subgroup_type"] == subgroup_type)
    ]
    if data.empty:
        return np.nan
    unique_sizes = data[
        ["experiment", "outcome", "modality", "subgroup_type", "subgroup", "n"]
    ].drop_duplicates()
    return float(unique_sizes["n"].mean())


def stratum_sample_n(
    metrics: pd.DataFrame,
    modality_label: str,
    outcome: str,
    subgroup_type: str,
    subgroup: str,
) -> float:
    data = metrics[
        (metrics["modality_label"] == modality_label)
        & (metrics["outcome"] == outcome)
        & (metrics["subgroup_type"] == subgroup_type)
        & (metrics["subgroup"] == subgroup)
    ]
    if data.empty:
        return np.nan
    unique_sizes = data[
        ["experiment", "outcome", "modality", "subgroup_type", "subgroup", "n"]
    ].drop_duplicates()
    return float(unique_sizes["n"].mean())


def format_n(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{value:,.0f}"


def sample_size_note(
    metrics: pd.DataFrame,
    modality_label: str,
    outcome: str,
    subgroup_types: tuple[str, ...] = SUBGROUP_TYPE_ORDER,
) -> str:
    lines = ["Mean subgroup n"]
    for subgroup_type in subgroup_types:
        label = SUBGROUP_TYPE_LABELS.get(subgroup_type, subgroup_type)
        mean_n = mean_subgroup_n(metrics, modality_label, outcome, subgroup_type)
        lines.append(f"{label}: {format_n(mean_n)}")
    return "\n".join(lines)


def total_cases_controls(
    metrics: pd.DataFrame,
    modality_label: str,
    outcome: str,
) -> tuple[float, float]:
    data = metrics[
        (metrics["modality_label"] == modality_label)
        & (metrics["outcome"] == outcome)
    ]
    if data.empty:
        return np.nan, np.nan

    for subgroup_type in ("sex", "age_band", "apoe_polymorphism"):
        subgroup_data = data[data["subgroup_type"] == subgroup_type]
        if subgroup_data.empty:
            continue
        unique_counts = subgroup_data[
            ["experiment", "outcome", "modality", "subgroup_type", "subgroup", "cases", "controls"]
        ].drop_duplicates()
        experiment_totals = unique_counts.groupby("experiment")[["cases", "controls"]].sum()
        if not experiment_totals.empty:
            return (
                float(experiment_totals["cases"].mean()),
                float(experiment_totals["controls"].mean()),
            )
    return np.nan, np.nan


def cases_controls_note(
    metrics: pd.DataFrame,
    modality_label: str,
    outcome: str,
) -> str:
    cases, controls = total_cases_controls(metrics, modality_label, outcome)
    if not np.isfinite(cases) or not np.isfinite(controls):
        return ""
    return f"{OUTCOME_LABELS[outcome]}: {format_n(cases)} cases / {format_n(controls)} controls"


def format_i2_tau(i2_value: float, tau_value: float) -> str:
    i2_text = "NA"
    tau_text = "NA"
    if np.isfinite(i2_value):
        i2_text = f"{i2_value:.0f}%"
    if np.isfinite(tau_value):
        if tau_value == 0:
            tau_text = "0"
        elif abs(tau_value) < 0.001:
            tau_text = f"{tau_value:.1e}"
        else:
            tau_text = f"{tau_value:.3f}"
    return f"I2={i2_text}, tau={tau_text}"


def heterogeneity_cell(
    heterogeneity: pd.DataFrame,
    modality_label: str,
    outcome: str,
    metric_name: str,
    subgroup_type: str,
    experiment_label: str,
) -> str:
    data = heterogeneity[
        (heterogeneity["modality_label"] == modality_label)
        & (heterogeneity["outcome"] == outcome)
        & (heterogeneity["metric"] == metric_name)
        & (heterogeneity["subgroup_type"] == subgroup_type)
        & (heterogeneity["experiment_label"] == experiment_label)
    ]
    if data.empty:
        return "I2=NA, tau=NA"
    row = data.iloc[0]
    return format_i2_tau(float(row["i2"]), float(row["tau"]))


def subgroup_block_centers(strata: list[tuple[str, str]]) -> dict[str, float]:
    centers = {}
    for subgroup_type in SUBGROUP_TYPE_ORDER:
        indices = [
            index
            for index, (stratum_type, _) in enumerate(strata)
            if stratum_type == subgroup_type
        ]
        if indices:
            centers[subgroup_type] = float(np.mean(indices))
    return centers


def add_heterogeneity_annotations(
    ax: plt.Axes,
    heterogeneity: pd.DataFrame,
    modality_label: str,
    outcome: str,
    metric_name: str,
    strata: list[tuple[str, str]],
    experiment_labels: tuple[str, str],
) -> None:
    centers = subgroup_block_centers(strata)
    transform = ax.get_yaxis_transform()
    line_gap = 0.52
    line_center = (len(experiment_labels) - 1) / 2
    for subgroup_type, center in centers.items():
        subgroup_label = SHORT_SUBGROUP_TYPE_LABELS.get(subgroup_type, subgroup_type)
        for line_index, experiment_label in enumerate(experiment_labels):
            y_position = center + (line_index - line_center) * line_gap
            text = (
                f"{subgroup_label} "
                f"{EXPERIMENT_SHORT_LABELS.get(experiment_label, experiment_label)}: "
                f"{heterogeneity_cell(heterogeneity, modality_label, outcome, metric_name, subgroup_type, experiment_label)}"
            )
            ax.text(
                0.985,
                y_position,
                text,
                transform=transform,
                ha="right",
                va="center",
                fontsize=6.4,
                color=EXPERIMENT_COLORS.get(experiment_label, "0.25"),
                bbox={
                    "boxstyle": "round,pad=0.13",
                    "facecolor": "white",
                    "alpha": 0.82,
                    "edgecolor": "none",
                },
                clip_on=False,
                zorder=4,
            )


def add_sample_size_note(
    ax: plt.Axes,
    metrics: pd.DataFrame,
    modality_label: str,
    outcome: str,
    subgroup_types: tuple[str, ...] = SUBGROUP_TYPE_ORDER,
) -> None:
    ax.text(
        0.99,
        0.03,
        sample_size_note(metrics, modality_label, outcome, subgroup_types),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.2,
        color="0.25",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.82, "edgecolor": "0.8"},
        zorder=5,
    )


def subgroup_label_with_mean_n(
    metrics: pd.DataFrame,
    modality_label: str,
    outcome: str,
    subgroup_type: str,
) -> str:
    label = SUBGROUP_TYPE_LABELS.get(subgroup_type, subgroup_type)
    mean_n = mean_subgroup_n(metrics, modality_label, outcome, subgroup_type)
    return f"{label} (mean n={format_n(mean_n)})"


def short_subgroup_label_with_mean_n(
    metrics: pd.DataFrame,
    modality_label: str,
    outcome: str,
    subgroup_type: str,
) -> str:
    label = SHORT_SUBGROUP_TYPE_LABELS.get(subgroup_type, subgroup_type)
    mean_n = mean_subgroup_n(metrics, modality_label, outcome, subgroup_type)
    return f"{label} (n={format_n(mean_n)})"


def stratum_label_with_n(
    metrics: pd.DataFrame,
    modality_label: str,
    outcome: str,
    subgroup_type: str,
    subgroup: str,
) -> str:
    label = stratum_label(subgroup_type, subgroup, modality_label)
    n_value = stratum_sample_n(metrics, modality_label, outcome, subgroup_type, subgroup)
    return f"{label} (n={format_n(n_value)})"


def region_level_performance(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    subgroup_columns = {
        "age_band": "age_band",
        "sex": "sex",
        "apoe_polymorphism": "apoe_polymorphism",
    }
    base_cols = [
        "experiment",
        "experiment_label",
        "outcome",
        "modality",
        "modality_label",
        "region_index",
        "region",
    ]

    for subgroup_type, subgroup_col in subgroup_columns.items():
        group_cols = base_cols + [subgroup_col]
        for keys, data in oof[oof[subgroup_col].notna()].groupby(group_cols, sort=False):
            row = dict(zip(group_cols, keys))
            y_true = data["y_true"].to_numpy()
            y_prob = data["y_prob"].to_numpy()
            cases = int(np.sum(y_true))
            controls = int(len(y_true) - cases)
            auc = np.nan
            if cases > 0 and controls > 0:
                auc = float(roc_auc_score(y_true, y_prob))
            row.update(
                {
                    "subgroup_type": subgroup_type,
                    "subgroup": row[subgroup_col],
                    "n": len(data),
                    "cases": cases,
                    "controls": controls,
                    "auc": auc,
                    "brier": float(np.mean((y_prob - y_true) ** 2)),
                }
            )
            rows.append(row)

    return pd.DataFrame(rows)


def dersimonian_laird_i2(
    estimates: np.ndarray,
    standard_errors: np.ndarray,
) -> float:
    estimates = np.asarray(estimates, dtype=float)
    standard_errors = np.asarray(standard_errors, dtype=float)
    variances = standard_errors**2
    keep = np.isfinite(estimates) & np.isfinite(variances) & (variances > 0)
    estimates = estimates[keep]
    variances = variances[keep]
    if len(estimates) < 2:
        return np.nan

    weights = 1 / variances
    fixed_effect = np.sum(weights * estimates) / np.sum(weights)
    q = np.sum(weights * (estimates - fixed_effect) ** 2)
    q_df = len(estimates) - 1
    if q <= 0:
        return 0.0
    return float(max(0.0, (q - q_df) / q) * 100)


def simulated_i2_interval(
    estimates: np.ndarray,
    standard_errors: np.ndarray,
    rng: np.random.Generator,
    simulations: int = 1000,
) -> tuple[float, float]:
    estimates = np.asarray(estimates, dtype=float)
    standard_errors = np.asarray(standard_errors, dtype=float)
    keep = np.isfinite(estimates) & np.isfinite(standard_errors) & (standard_errors > 0)
    estimates = estimates[keep]
    standard_errors = standard_errors[keep]
    if len(estimates) < 2:
        return np.nan, np.nan

    draws = rng.normal(loc=estimates, scale=standard_errors, size=(simulations, len(estimates)))
    i2_values = np.array(
        [dersimonian_laird_i2(draw, standard_errors) for draw in draws],
        dtype=float,
    )
    i2_values = i2_values[np.isfinite(i2_values)]
    if len(i2_values) == 0:
        return np.nan, np.nan
    lower, upper = np.nanpercentile(i2_values, [2.5, 97.5])
    return float(lower), float(upper)


def add_i2_uncertainty(
    heterogeneity: pd.DataFrame,
    metrics: pd.DataFrame,
    decision_curves: pd.DataFrame,
) -> pd.DataFrame:
    heterogeneity = heterogeneity.copy()
    heterogeneity["i2_ci_lower"] = np.nan
    heterogeneity["i2_ci_upper"] = np.nan
    rng = np.random.default_rng(20260630)

    for metric_name in ("auc", "brier"):
        source = metrics[metrics[metric_name].notna()].copy()
        group_cols = ["experiment", "outcome", "modality", "subgroup_type"]
        for keys, data in source.groupby(group_cols, sort=False):
            lower, upper = simulated_i2_interval(
                data[metric_name].to_numpy(),
                data[f"{metric_name}_se"].to_numpy(),
                rng,
            )
            mask = (
                (heterogeneity["experiment"] == keys[0])
                & (heterogeneity["outcome"] == keys[1])
                & (heterogeneity["modality"] == keys[2])
                & (heterogeneity["subgroup_type"] == keys[3])
                & (heterogeneity["metric"] == metric_name)
            )
            heterogeneity.loc[mask, ["i2_ci_lower", "i2_ci_upper"]] = lower, upper

    source = decision_curves[decision_curves["net_benefit"].notna()].copy()
    group_cols = ["experiment", "outcome", "modality", "subgroup_type", "threshold"]
    for keys, data in source.groupby(group_cols, sort=False):
        lower, upper = simulated_i2_interval(
            data["net_benefit"].to_numpy(),
            data["net_benefit_se"].to_numpy(),
            rng,
        )
        mask = (
            (heterogeneity["experiment"] == keys[0])
            & (heterogeneity["outcome"] == keys[1])
            & (heterogeneity["modality"] == keys[2])
            & (heterogeneity["subgroup_type"] == keys[3])
            & (heterogeneity["threshold"] == keys[4])
            & (heterogeneity["metric"] == "net_benefit")
        )
        heterogeneity.loc[mask, ["i2_ci_lower", "i2_ci_upper"]] = lower, upper

    return heterogeneity


def ordered_strata(modality_metrics: pd.DataFrame) -> list[tuple[str, str]]:
    modality_label = (
        str(modality_metrics["modality_label"].iloc[0])
        if "modality_label" in modality_metrics and not modality_metrics.empty
        else None
    )
    strata = (
        modality_metrics[["subgroup_type", "subgroup"]]
        .drop_duplicates()
        .assign(
            subgroup_type_order=lambda df: df["subgroup_type"].map(
                {value: index for index, value in enumerate(SUBGROUP_TYPE_ORDER)}
            ),
            subgroup_order=lambda df: [
                subgroup_sort_value(subgroup_type, subgroup, modality_label)
                for subgroup_type, subgroup in zip(df["subgroup_type"], df["subgroup"])
            ],
        )
        .sort_values(["subgroup_type_order", "subgroup_order", "subgroup"])
    )
    return list(strata[["subgroup_type", "subgroup"]].itertuples(index=False, name=None))


def add_group_separators(
    ax: plt.Axes,
    strata: list[tuple[str, str]],
    *,
    color: str = "0.85",
) -> None:
    for index in range(len(strata) - 1):
        if strata[index][0] != strata[index + 1][0]:
            ax.axhline(index + 0.5, color=color, linewidth=1.0, zorder=0)


def metric_limits(data: pd.DataFrame, metric_name: str) -> tuple[float, float]:
    lower_col = f"{metric_name}_ci_lower"
    upper_col = f"{metric_name}_ci_upper"
    values = data[[metric_name, lower_col, upper_col]].to_numpy(dtype=float).ravel()
    values = values[np.isfinite(values)]
    if metric_name == "auc":
        return 0.3, 1.0
    if len(values) == 0:
        return 0.0, 0.1
    return 0.0, max(0.02, float(np.nanmax(values)) * 1.12)


def plot_subgroup_performance(
    metrics: pd.DataFrame,
    individual_metrics: pd.DataFrame,
    heterogeneity: pd.DataFrame,
    output_dir: Path,
    experiment_labels: tuple[str, str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for modality in MODALITY_ORDER:
        modality_metrics = metrics[metrics["modality_label"] == modality].copy()
        if modality_metrics.empty:
            continue

        strata = ordered_strata(modality_metrics)
        y_positions = {stratum: index for index, stratum in enumerate(strata)}
        height = max(7.5, 0.46 * len(strata) + 3.0)
        fig, axes = plt.subplots(
            nrows=2,
            ncols=2,
            figsize=(15.8, height),
            sharey=False,
            constrained_layout=False,
        )

        for row_index, outcome in enumerate(OUTCOME_ORDER):
            y_labels = [
                stratum_label_with_n(
                    metrics,
                    modality,
                    outcome,
                    subgroup_type,
                    subgroup,
                )
                for subgroup_type, subgroup in strata
            ]
            for col_index, metric_name in enumerate(("auc", "brier")):
                ax = axes[row_index, col_index]
                panel = modality_metrics[
                    (modality_metrics["outcome"] == outcome)
                    & modality_metrics[metric_name].notna()
                ].copy()
                offset = 0.16
                offsets = {
                    experiment_labels[0]: -offset,
                    experiment_labels[1]: offset,
                }
                for experiment_label in experiment_labels:
                    individual_panel = individual_metrics[
                        (individual_metrics["modality_label"] == modality)
                        & (individual_metrics["outcome"] == outcome)
                        & (individual_metrics["experiment_label"] == experiment_label)
                        & individual_metrics[metric_name].notna()
                    ].copy()
                    color = EXPERIMENT_COLORS.get(experiment_label, "0.35")
                    for stratum, stratum_points in individual_panel.groupby(
                        ["subgroup_type", "subgroup"], sort=False
                    ):
                        if stratum not in y_positions:
                            continue
                        jitter = np.linspace(-0.055, 0.055, len(stratum_points))
                        ax.scatter(
                            stratum_points[metric_name],
                            y_positions[stratum] + offsets[experiment_label] + jitter,
                            s=13,
                            color=color,
                            alpha=0.28,
                            edgecolors="none",
                            zorder=2,
                        )

                    exp_panel = panel[panel["experiment_label"] == experiment_label]
                    for _, row in exp_panel.iterrows():
                        stratum = (row["subgroup_type"], row["subgroup"])
                        if stratum not in y_positions:
                            continue
                        estimate = row[metric_name]
                        lower = row[f"{metric_name}_ci_lower"]
                        upper = row[f"{metric_name}_ci_upper"]
                        if not np.isfinite(estimate):
                            continue
                        xerr = None
                        if np.isfinite(lower) and np.isfinite(upper):
                            xerr = [[max(0.0, estimate - lower)], [max(0.0, upper - estimate)]]
                        ax.errorbar(
                            estimate,
                            y_positions[stratum] + offsets[experiment_label],
                            xerr=xerr,
                            fmt="o",
                            markersize=5.5,
                            color=color,
                            ecolor=color,
                            elinewidth=1.5,
                            capsize=2.5,
                            label=experiment_label,
                            zorder=3,
                        )

                ax.set_yticks(range(len(strata)))
                if col_index == 0:
                    ax.set_yticklabels(y_labels)
                else:
                    ax.tick_params(axis="y", labelleft=False)
                ax.tick_params(axis="y", labelsize=8.8)
                ax.set_ylim(-0.5, len(strata) - 0.5)
                add_group_separators(ax, strata)
                ax.grid(axis="x", color="0.9", linewidth=0.8)
                ax.set_axisbelow(True)
                ax.set_xlim(*metric_limits(panel, metric_name))
                if metric_name == "auc":
                    ax.axvline(0.5, color="0.65", linestyle="--", linewidth=1.0)
                    ax.set_xlabel("AUC (95% CI)")
                else:
                    ax.set_xlabel("Brier score (95% CI; lower is better)")
                ax.set_title(metric_name.upper(), pad=14)
                add_heterogeneity_annotations(
                    ax,
                    heterogeneity,
                    modality,
                    outcome,
                    metric_name,
                    strata,
                    experiment_labels,
                )

        for row_index in range(len(OUTCOME_ORDER)):
            axes[row_index, 0].invert_yaxis()
            axes[row_index, 1].invert_yaxis()

        handles, labels = axes[0, 0].get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        fig.legend(
            unique.values(),
            unique.keys(),
            loc="upper center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 0.965),
        )
        fig.suptitle(f"{modality}: stratum-level performance comparison", y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.9], h_pad=3.2)
        for row_index, outcome in enumerate(OUTCOME_ORDER):
            note = cases_controls_note(metrics, modality, outcome)
            if not note:
                continue
            left_box = axes[row_index, 0].get_position()
            right_box = axes[row_index, 1].get_position()
            fig.text(
                (left_box.x0 + right_box.x1) / 2,
                max(left_box.y1, right_box.y1) + 0.035,
                note,
                ha="center",
                va="bottom",
                fontsize=9.4,
                color="0.25",
                fontweight="semibold",
            )
        slug = modality.lower().replace(" ", "_")
        fig.savefig(
            output_dir / f"subgroup_performance_comparison_{slug}.png",
            dpi=300,
            bbox_inches="tight",
        )
        fig.savefig(
            output_dir / f"subgroup_performance_comparison_{slug}.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_performance_i2_heatmap(
    heterogeneity: pd.DataFrame,
    metrics: pd.DataFrame,
    output_dir: Path,
) -> None:
    perf = heterogeneity[heterogeneity["metric"].isin(["auc", "brier"])].copy()
    perf["metric_label"] = perf["metric"].str.upper()
    perf["row_label"] = [
        f"{modality} | {OUTCOME_LABELS[outcome]} | "
        f"{subgroup_label_with_mean_n(metrics, modality, outcome, subgroup_type)} | "
        f"{metric_label}"
        for modality, outcome, subgroup_type, metric_label in zip(
            perf["modality_label"],
            perf["outcome"],
            perf["subgroup_type"],
            perf["metric_label"],
        )
    ]
    perf["sort_modality"] = perf["modality_label"].map(
        {value: index for index, value in enumerate(MODALITY_ORDER)}
    )
    perf["sort_outcome"] = perf["outcome"].map(
        {value: index for index, value in enumerate(OUTCOME_ORDER)}
    )
    perf["sort_subgroup"] = perf["subgroup_type"].map(
        {value: index for index, value in enumerate(SUBGROUP_TYPE_ORDER)}
    )
    perf["sort_metric"] = perf["metric"].map({"auc": 0, "brier": 1})
    perf = perf.sort_values(
        ["sort_modality", "sort_outcome", "sort_subgroup", "sort_metric"]
    )
    row_order = perf["row_label"].drop_duplicates().tolist()
    heatmap_data = (
        perf.pivot_table(
            index="row_label",
            columns="experiment_label",
            values="i2",
            aggfunc="first",
        )
        .reindex(row_order)
        .reindex(columns=[EXPERIMENT_LABELS[e] for e in DEFAULT_EXPERIMENTS])
    )
    lower_data = (
        perf.pivot_table(
            index="row_label",
            columns="experiment_label",
            values="i2_ci_lower",
            aggfunc="first",
        )
        .reindex(row_order)
        .reindex(columns=[EXPERIMENT_LABELS[e] for e in DEFAULT_EXPERIMENTS])
    )
    upper_data = (
        perf.pivot_table(
            index="row_label",
            columns="experiment_label",
            values="i2_ci_upper",
            aggfunc="first",
        )
        .reindex(row_order)
        .reindex(columns=[EXPERIMENT_LABELS[e] for e in DEFAULT_EXPERIMENTS])
    )
    annotations = heatmap_data.copy().astype(object)
    for row in annotations.index:
        for col in annotations.columns:
            value = heatmap_data.loc[row, col]
            lower = lower_data.loc[row, col]
            upper = upper_data.loc[row, col]
            if not np.isfinite(value):
                annotations.loc[row, col] = ""
            elif np.isfinite(lower) and np.isfinite(upper):
                annotations.loc[row, col] = f"{value:.0f}\n[{lower:.0f}, {upper:.0f}]"
            else:
                annotations.loc[row, col] = f"{value:.0f}"

    fig_height = max(10.0, 0.32 * len(heatmap_data) + 2.0)
    fig, ax = plt.subplots(figsize=(12.5, fig_height), constrained_layout=True)
    sns.heatmap(
        heatmap_data,
        ax=ax,
        cmap="YlOrRd",
        vmin=0,
        vmax=100,
        annot=annotations,
        fmt="",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "I2 (%)"},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
    ax.set_title("Formal heterogeneity by modality, stratum type, and metric")
    fig.savefig(
        output_dir / "performance_heterogeneity_i2_heatmap.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(output_dir / "performance_heterogeneity_i2_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_performance_i2_delta(
    heterogeneity: pd.DataFrame,
    metrics: pd.DataFrame,
    output_dir: Path,
    experiments: tuple[str, str],
) -> None:
    perf = heterogeneity[heterogeneity["metric"].isin(["auc", "brier"])].copy()
    index_cols = ["modality_label", "outcome", "subgroup_type", "metric"]
    wide = perf.pivot_table(
        index=index_cols,
        columns="experiment",
        values=["i2", "i2_ci_lower", "i2_ci_upper"],
        aggfunc="first",
    )
    wide.columns = [f"{value}_{experiment}" for value, experiment in wide.columns]
    wide = wide.reset_index()
    wide["delta_i2"] = wide[f"i2_{experiments[1]}"] - wide[f"i2_{experiments[0]}"]
    wide["delta_i2_ci_lower"] = (
        wide[f"i2_ci_lower_{experiments[1]}"] - wide[f"i2_ci_upper_{experiments[0]}"]
    )
    wide["delta_i2_ci_upper"] = (
        wide[f"i2_ci_upper_{experiments[1]}"] - wide[f"i2_ci_lower_{experiments[0]}"]
    )
    wide["row_label"] = [
        f"{SHORT_OUTCOME_LABELS.get(outcome, outcome)} | "
        f"{short_subgroup_label_with_mean_n(metrics, modality_label, outcome, subgroup_type)} | "
        f"{metric.upper()}"
        for modality_label, outcome, subgroup_type, metric in zip(
            wide["modality_label"],
            wide["outcome"],
            wide["subgroup_type"],
            wide["metric"],
        )
    ]

    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(24, 9.5),
        sharex=True,
        constrained_layout=True,
    )
    for ax, modality in zip(axes, MODALITY_ORDER):
        panel = wide[wide["modality_label"] == modality].copy()
        panel["sort_outcome"] = panel["outcome"].map(
            {value: index for index, value in enumerate(OUTCOME_ORDER)}
        )
        panel["sort_subgroup"] = panel["subgroup_type"].map(
            {value: index for index, value in enumerate(SUBGROUP_TYPE_ORDER)}
        )
        panel["sort_metric"] = panel["metric"].map({"auc": 0, "brier": 1})
        panel = panel.sort_values(["sort_outcome", "sort_subgroup", "sort_metric"])
        panel = panel[panel["delta_i2"].notna()]
        colors = np.where(panel["delta_i2"] >= 0, "#E45756", "#54A24B")
        ax.barh(panel["row_label"], panel["delta_i2"], color=colors, alpha=0.86)
        error_mask = (
            panel["delta_i2_ci_lower"].notna()
            & panel["delta_i2_ci_upper"].notna()
        )
        if error_mask.any():
            error_panel = panel[error_mask]
            xerr = np.vstack(
                [
                    np.maximum(
                        0,
                        error_panel["delta_i2"] - error_panel["delta_i2_ci_lower"],
                    ),
                    np.maximum(
                        0,
                        error_panel["delta_i2_ci_upper"] - error_panel["delta_i2"],
                    ),
                ]
            )
            ax.errorbar(
                error_panel["delta_i2"],
                error_panel["row_label"],
                xerr=xerr,
                fmt="none",
                ecolor="0.25",
                elinewidth=1.2,
                capsize=3,
                zorder=3,
            )
        ax.axvline(0, color="0.25", linewidth=1.0)
        ax.grid(axis="x", color="0.9", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_title(modality)
        ax.set_xlabel("Delta I2: modality-augmented minus demographics + Lancet")
        ax.invert_yaxis()
    fig.suptitle("Change in formal performance heterogeneity after adding modality features")
    fig.savefig(
        output_dir / "performance_heterogeneity_i2_delta.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(output_dir / "performance_heterogeneity_i2_delta.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_net_benefit_i2_by_threshold(
    heterogeneity: pd.DataFrame,
    metrics: pd.DataFrame,
    output_dir: Path,
    experiment_labels: tuple[str, str],
) -> None:
    dca = heterogeneity[heterogeneity["metric"] == "net_benefit"].copy()
    for modality in MODALITY_ORDER:
        modality_dca = dca[dca["modality_label"] == modality].copy()
        if modality_dca.empty:
            continue
        fig, axes = plt.subplots(
            nrows=2,
            ncols=3,
            figsize=(15.5, 7.5),
            sharex=True,
            sharey=True,
            constrained_layout=False,
        )
        for row_index, outcome in enumerate(OUTCOME_ORDER):
            for col_index, subgroup_type in enumerate(SUBGROUP_TYPE_ORDER):
                ax = axes[row_index, col_index]
                panel = modality_dca[
                    (modality_dca["outcome"] == outcome)
                    & (modality_dca["subgroup_type"] == subgroup_type)
                ].copy()
                for experiment_label in experiment_labels:
                    exp_panel = panel[
                        (panel["experiment_label"] == experiment_label)
                        & panel["i2"].notna()
                    ].sort_values("threshold")
                    if exp_panel.empty:
                        continue
                    color = EXPERIMENT_COLORS.get(experiment_label, "0.35")
                    band_panel = exp_panel[
                        exp_panel["i2_ci_lower"].notna()
                        & exp_panel["i2_ci_upper"].notna()
                    ]
                    if not band_panel.empty:
                        ax.fill_between(
                            band_panel["threshold"].to_numpy(dtype=float),
                            band_panel["i2_ci_lower"].to_numpy(dtype=float),
                            band_panel["i2_ci_upper"].to_numpy(dtype=float),
                            color=color,
                            alpha=0.16,
                            linewidth=0,
                            zorder=1,
                        )
                    ax.plot(
                        exp_panel["threshold"],
                        exp_panel["i2"],
                        color=color,
                        linewidth=2.0,
                        label=experiment_label,
                        zorder=2,
                    )
                ax.set_ylim(0, 100)
                ax.grid(color="0.9", linewidth=0.8)
                ax.set_axisbelow(True)
                if row_index == 1:
                    ax.set_xlabel("Decision threshold")
                if col_index == 0:
                    ax.set_ylabel(f"{OUTCOME_LABELS[outcome]}\nI2 (%)")
                ax.set_title(SUBGROUP_TYPE_LABELS[subgroup_type])

        handles, labels = axes[0, 0].get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        fig.legend(
            unique.values(),
            unique.keys(),
            loc="upper center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 0.965),
        )
        fig.suptitle(f"{modality}: decision-curve net-benefit heterogeneity", y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.9])
        slug = modality.lower().replace(" ", "_")
        fig.savefig(
            output_dir / f"net_benefit_i2_by_threshold_{slug}.png",
            dpi=300,
            bbox_inches="tight",
        )
        fig.savefig(
            output_dir / f"net_benefit_i2_by_threshold_{slug}.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)


def write_summary_tables(
    heterogeneity: pd.DataFrame,
    output_dir: Path,
    experiments: tuple[str, str],
) -> None:
    perf = heterogeneity[heterogeneity["metric"].isin(["auc", "brier"])].copy()
    summary = (
        perf.groupby(["modality_label", "experiment", "experiment_label", "metric"])
        .agg(
            n=("i2", "size"),
            median_i2=("i2", "median"),
            max_i2=("i2", "max"),
            significant_q=("q_pvalue", lambda values: int((values < 0.05).sum())),
            median_tau=("tau", "median"),
            max_tau=("tau", "max"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "performance_heterogeneity_summary.csv", index=False)

    index_cols = ["outcome", "modality_label", "subgroup_type", "metric"]
    wide = perf.pivot_table(
        index=index_cols,
        columns="experiment",
        values=["i2", "i2_ci_lower", "i2_ci_upper", "tau", "q_pvalue"],
        aggfunc="first",
    )
    wide.columns = [f"{value}_{experiment}" for value, experiment in wide.columns]
    wide = wide.reset_index()
    wide["delta_i2_modality_minus_demographics"] = (
        wide[f"i2_{experiments[1]}"] - wide[f"i2_{experiments[0]}"]
    )
    wide["delta_i2_ci_lower_modality_minus_demographics"] = (
        wide[f"i2_ci_lower_{experiments[1]}"]
        - wide[f"i2_ci_upper_{experiments[0]}"]
    )
    wide["delta_i2_ci_upper_modality_minus_demographics"] = (
        wide[f"i2_ci_upper_{experiments[1]}"]
        - wide[f"i2_ci_lower_{experiments[0]}"]
    )
    wide["delta_tau_modality_minus_demographics"] = (
        wide[f"tau_{experiments[1]}"] - wide[f"tau_{experiments[0]}"]
    )
    wide.to_csv(output_dir / "performance_heterogeneity_pairwise_delta.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot heterogeneity comparisons between two UK Biobank experiments."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "results" / "UKBiobank" / "formal_heterogeneity",
    )
    parser.add_argument("--experiments", nargs=2, default=list(DEFAULT_EXPERIMENTS))
    parser.add_argument("--metric", default="log_loss")
    parser.add_argument("--model", default="lgbm")
    parser.add_argument("--age-cutoff", type=int, default=65)
    parser.add_argument(
        "--age-bins",
        nargs="+",
        type=int,
        default=[65, 70, 75, 80, 85],
        help="Age-band lower bounds used by heterogeneity_analysis.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to a comparisons folder under the formal heterogeneity root.",
    )
    parser.add_argument(
        "--only-subgroup-performance",
        action="store_true",
        help="Only redraw subgroup_performance figures; skips decision-curve inputs and other plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiments = tuple(args.experiments)
    age_bins = tuple(args.age_bins)
    if len(experiments) != 2:
        raise ValueError("Exactly two experiments are required")
    if len(age_bins) < 2:
        raise ValueError("--age-bins requires at least two cutpoints")

    comparison_slug = f"{experiments[0]}_vs_{experiments[1]}"
    output_dir = args.output_dir or (
        args.results_root
        / "comparisons"
        / comparison_slug
        / args.metric
        / args.model
        / f"agecutoff_{args.age_cutoff}"
        / age_bins_label(age_bins)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics, heterogeneity, decision_curves, oof = load_comparison_tables(
        results_root=args.results_root,
        experiments=experiments,
        metric=args.metric,
        model=args.model,
        age_cutoff=args.age_cutoff,
        age_bins=age_bins,
        include_decision_curves=not args.only_subgroup_performance,
    )
    individual_metrics = region_level_performance(oof)
    experiment_labels = tuple(EXPERIMENT_LABELS.get(exp, exp) for exp in experiments)

    if args.only_subgroup_performance:
        plot_subgroup_performance(
            metrics,
            individual_metrics,
            heterogeneity,
            output_dir,
            experiment_labels,
        )
        print(f"Wrote subgroup performance comparison plots to {output_dir}")
        return

    heterogeneity = add_i2_uncertainty(heterogeneity, metrics, decision_curves)
    write_summary_tables(heterogeneity, output_dir, experiments)
    plot_subgroup_performance(
        metrics,
        individual_metrics,
        heterogeneity,
        output_dir,
        experiment_labels,
    )
    plot_performance_i2_heatmap(heterogeneity, metrics, output_dir)
    plot_performance_i2_delta(heterogeneity, metrics, output_dir, experiments)
    plot_net_benefit_i2_by_threshold(heterogeneity, metrics, output_dir, experiment_labels)

    print(f"Wrote heterogeneity comparison plots to {output_dir}")


if __name__ == "__main__":
    main()
