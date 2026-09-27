"""
Post hoc subgroup heterogeneity analysis for UK Biobank model predictions.

This script reconstructs the held-out patients used by ml_experiments.py, joins
saved test-set probabilities back to age/sex/APOE fields, and reports subgroup
performance plus formal DerSimonian-Laird heterogeneity statistics.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import chi2
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


REPO_ROOT = Path(__file__).resolve().parents[2]
import os, sys
_LIBDIR = os.path.dirname(os.path.abspath(__file__))
while _LIBDIR != os.path.dirname(_LIBDIR) and not os.path.isdir(os.path.join(_LIBDIR, "utils")):
    _LIBDIR = os.path.dirname(_LIBDIR)
sys.path.insert(0, os.path.join(_LIBDIR, "utils"))

import dementia_utils
import ukb_utils


DEFAULT_MODALITIES = ("proteomics", "neuroimaging", "cognitive_tests")
DEFAULT_OUTCOMES = ("dementia", "alzheimers")
DEFAULT_AGE_BINS = (65, 75, 85)
APOE_COLUMNS = (
    "apoe_polymorphism_0.0",
    "apoe_polymorphism_1.0",
    "apoe_polymorphism_2.0",
    "apoe_polymorphism_nan",
)
APOE_LABELS = {
    "apoe_polymorphism_0.0": "apoe_polymorphism=0.0",
    "apoe_polymorphism_1.0": "apoe_polymorphism=1.0",
    "apoe_polymorphism_2.0": "apoe_polymorphism=2.0",
    "apoe_polymorphism_nan": "apoe_polymorphism=missing",
}


@dataclass(frozen=True)
class AnalysisConfig:
    results_root: Path
    tidy_root: Path
    metadata_root: Path
    acd_path: Path
    output_root: Path
    outcomes: tuple[str, ...]
    modalities: tuple[str, ...]
    experiment: str
    metric: str
    model: str
    age_cutoff: int | None
    age_bins: tuple[int, ...]
    dca_threshold_min: float
    dca_threshold_max: float
    dca_threshold_step: float
    strict_label_check: bool
    make_plots: bool


def parse_none_int(value: str | None) -> int | None:
    if value is None:
        return None
    if str(value).lower() in {"none", "null", "na", "0"}:
        return None
    return int(value)


def age_bin_labels(age_bins: tuple[int, ...]) -> list[str]:
    labels = [f"{start}-{stop}" for start, stop in zip(age_bins[:-1], age_bins[1:])]
    labels.append(f"{age_bins[-1]}+")
    return labels


def age_bins_label(age_bins: tuple[int, ...]) -> str:
    return f"agebins_{'_'.join(str(value) for value in age_bins)}_plus"


def modality_instance(modality: str) -> int:
    return 2 if modality == "neuroimaging" else 0


def model_dir(config: AnalysisConfig, outcome: str, modality: str) -> Path:
    path = (
        config.results_root
        / outcome
        / modality
        / config.experiment
        / config.metric
        / config.model
    )
    if config.age_cutoff is not None:
        path = path / f"agecutoff_{config.age_cutoff}"
    return path


def read_pickle_array(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        obj = pickle.load(handle)
    while isinstance(obj, list) and len(obj) == 1:
        obj = obj[0]
    return np.asarray(obj)


def region_index_from_path(path: Path) -> int:
    match = re.search(r"region_(\d+)\.pkl$", path.name)
    if match is None:
        raise ValueError(f"Could not parse region index from {path}")
    return int(match.group(1))


def parquet_columns(path: Path) -> list[str]:
    return pq.read_schema(path).names


def columns_with_prefix(columns: Iterable[str], prefixes: Iterable[str]) -> list[str]:
    return [col for col in columns if any(col.startswith(prefix) for prefix in prefixes)]


def load_alzheimer_eids(acd_path: Path, eligible_eids: pd.Series) -> set[int]:
    prefixes = [
        "eid",
        "42019",
        "42021",
        "42023",
        "42025",
        "131037",
        "130837",
        "130839",
        "130841",
        "130843",
        "42018",
        "42020",
        "42022",
        "42024",
        "131036",
        "130836",
        "130838",
        "130840",
        "130842",
    ]
    cols = columns_with_prefix(parquet_columns(acd_path), prefixes)
    acd = pd.read_parquet(acd_path, columns=cols)
    acd = acd[acd["eid"].isin(eligible_eids)]
    alzheimer_eids, _, _ = dementia_utils.pull_dementia_cases(
        acd, alzheimers_only=True
    )
    return set(alzheimer_eids)


def load_base_metadata(
    config: AnalysisConfig, outcome: str, modality: str
) -> tuple[pd.DataFrame, np.ndarray, int]:
    instance = modality_instance(modality)
    age_col = f"21003-{instance}.0"
    center_col = f"54-{instance}.0"
    sex_col = "31-0.0_1.0"

    x_path = config.tidy_root / modality / "X.parquet"
    y_path = config.tidy_root / modality / "y.npy"
    available_cols = parquet_columns(x_path)
    requested_cols = ["eid", age_col, sex_col] + list(APOE_COLUMNS)
    if modality != "neuroimaging":
        requested_cols.append(center_col)
    use_cols = [col for col in requested_cols if col in available_cols]

    missing = sorted(set(requested_cols).difference(use_cols).difference({center_col}))
    if missing:
        raise ValueError(f"{modality} metadata is missing expected columns: {missing}")

    metadata = pd.read_parquet(x_path, columns=use_cols)
    y = np.load(y_path)

    if outcome == "alzheimers":
        alzheimer_eids = load_alzheimer_eids(config.acd_path, metadata["eid"])
        keep = (y == 0) | metadata["eid"].isin(alzheimer_eids).to_numpy()
        metadata = metadata.loc[keep].reset_index(drop=True)
        y = y[keep]

    if config.age_cutoff is not None:
        keep = metadata[age_col].to_numpy() >= config.age_cutoff
        metadata = metadata.loc[keep].reset_index(drop=True)
        y = y[keep]

    return metadata, y, instance


def region_indices_for_metadata(
    metadata: pd.DataFrame, instance: int, metadata_root: Path
) -> dict[str, pd.Index]:
    lookup = pd.read_csv(metadata_root / "coding10.tsv", sep="\t")
    return ukb_utils.group_assessment_center(metadata, instance, lookup)


def test_split_metadata(
    metadata: pd.DataFrame,
    y: np.ndarray,
    modality: str,
    instance: int,
    metadata_root: Path,
    region_index: int,
) -> tuple[pd.DataFrame, np.ndarray, str]:
    if modality == "neuroimaging":
        skf = StratifiedKFold(n_splits=10)
        for fold_index, (_, test_index) in enumerate(skf.split(metadata, y)):
            if fold_index == region_index:
                return (
                    metadata.iloc[test_index].reset_index(drop=True),
                    y[test_index],
                    str(fold_index),
                )
        raise ValueError(f"Fold {region_index} was not found")

    region_indices = region_indices_for_metadata(metadata, instance, metadata_root)
    region_names = list(region_indices.keys())
    region = region_names[region_index]
    test_index = np.asarray(region_indices[region])
    return (
        metadata.iloc[test_index].reset_index(drop=True),
        y[test_index],
        str(region),
    )


def assemble_oof_predictions(
    config: AnalysisConfig, outcome: str, modality: str
) -> pd.DataFrame:
    input_dir = model_dir(config, outcome, modality)
    probas_files = sorted(input_dir.glob("test_probas_region_*.pkl"))
    if not probas_files:
        raise FileNotFoundError(f"No test_probas_region_*.pkl files found in {input_dir}")

    metadata, y, instance = load_base_metadata(config, outcome, modality)
    age_col = f"21003-{instance}.0"
    rows = []

    for probas_path in probas_files:
        region_index = region_index_from_path(probas_path)
        labels_path = input_dir / f"test_true_labels_region_{region_index}.pkl"
        if not labels_path.exists():
            raise FileNotFoundError(f"Missing labels file: {labels_path}")

        probas = read_pickle_array(probas_path).astype(float)
        labels = read_pickle_array(labels_path).astype(int)
        test_meta, reconstructed_labels, region = test_split_metadata(
            metadata, y, modality, instance, config.metadata_root, region_index
        )
        reconstructed_labels = reconstructed_labels.astype(int)

        if len(probas) != len(test_meta):
            raise ValueError(
                f"{outcome}/{modality}/region_{region_index}: "
                f"{len(probas)} probabilities but {len(test_meta)} reconstructed patients"
            )
        if len(labels) != len(test_meta):
            raise ValueError(
                f"{outcome}/{modality}/region_{region_index}: "
                f"{len(labels)} labels but {len(test_meta)} reconstructed patients"
            )
        if config.strict_label_check and not np.array_equal(labels, reconstructed_labels):
            raise ValueError(
                f"{outcome}/{modality}/region_{region_index}: saved labels do not "
                "match reconstructed split labels"
            )

        fold_df = test_meta.copy()
        fold_df["outcome"] = outcome
        fold_df["modality"] = modality
        fold_df["region_index"] = region_index
        fold_df["region"] = region
        fold_df["age"] = fold_df[age_col]
        fold_df["y_true"] = labels
        fold_df["y_prob"] = probas
        rows.append(fold_df)

    oof = pd.concat(rows, ignore_index=True)
    return add_subgroup_labels(oof, config.age_bins)


def add_subgroup_labels(df: pd.DataFrame, age_bins: tuple[int, ...]) -> pd.DataFrame:
    df = df.copy()
    df["age_band"] = pd.cut(
        df["age"],
        bins=list(age_bins) + [np.inf],
        labels=age_bin_labels(age_bins),
        right=False,
    ).astype(object)

    sex_value = df["31-0.0_1.0"]
    df["sex"] = np.select(
        [sex_value == 1, sex_value == 0],
        ["Male", "Female"],
        default="Sex missing",
    )

    apoe_present = [col for col in APOE_COLUMNS if col in df.columns]
    apoe_values = df[apoe_present].fillna(0)
    apoe_col = apoe_values.idxmax(axis=1)
    no_apoe_flag = apoe_values.sum(axis=1) == 0
    df["apoe_polymorphism"] = apoe_col.map(APOE_LABELS)
    df.loc[no_apoe_flag, "apoe_polymorphism"] = "apoe_polymorphism=missing"
    return df


def compute_midrank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    order = np.argsort(values)
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1
        i = j
    return ranks


def hanley_mcneil_auc_variance(auc_value: float, n_cases: int, n_controls: int) -> float:
    if n_cases == 0 or n_controls == 0:
        return np.nan
    q1 = auc_value / (2 - auc_value)
    q2 = 2 * auc_value**2 / (1 + auc_value)
    variance = (
        auc_value * (1 - auc_value)
        + (n_cases - 1) * (q1 - auc_value**2)
        + (n_controls - 1) * (q2 - auc_value**2)
    ) / (n_cases * n_controls)
    return max(float(variance), 0.0)


def auc_with_se(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    n_cases = int(y_true.sum())
    n_controls = int(len(y_true) - n_cases)
    if n_cases == 0 or n_controls == 0:
        return np.nan, np.nan

    auc_value = float(roc_auc_score(y_true, y_prob))
    if n_cases < 2 or n_controls < 2:
        variance = hanley_mcneil_auc_variance(auc_value, n_cases, n_controls)
        return auc_value, float(np.sqrt(variance))

    pos = y_prob[y_true == 1]
    neg = y_prob[y_true == 0]
    all_scores = np.concatenate([pos, neg])
    tx = compute_midrank(pos)
    ty = compute_midrank(neg)
    tz = compute_midrank(all_scores)

    delong_auc = tz[:n_cases].sum() / (n_cases * n_controls)
    delong_auc -= (n_cases + 1) / (2 * n_controls)
    v01 = (tz[:n_cases] - tx) / n_controls
    v10 = 1 - (tz[n_cases:] - ty) / n_cases
    variance = np.var(v01, ddof=1) / n_cases + np.var(v10, ddof=1) / n_controls
    if not np.isfinite(variance) or variance <= 0:
        variance = hanley_mcneil_auc_variance(auc_value, n_cases, n_controls)

    return float(delong_auc), float(np.sqrt(max(variance, 0.0)))


def mean_metric_with_se(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan
    estimate = float(np.mean(values))
    se = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else np.nan
    return estimate, se


def subgroup_metric_rows(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    subgroup_cols = {
        "age_band": "age_band",
        "sex": "sex",
        "apoe_polymorphism": "apoe_polymorphism",
    }

    for subgroup_type, column in subgroup_cols.items():
        grouped = oof[oof[column].notna()].groupby(column, observed=True, sort=False)
        for subgroup, data in grouped:
            y_true = data["y_true"].to_numpy()
            y_prob = data["y_prob"].to_numpy()
            cases = int(y_true.sum())
            controls = int(len(y_true) - cases)
            auc_value, auc_se = auc_with_se(y_true, y_prob)
            brier_losses = (y_prob - y_true) ** 2
            brier, brier_se = mean_metric_with_se(brier_losses)
            rows.append(
                {
                    "outcome": data["outcome"].iloc[0],
                    "modality": data["modality"].iloc[0],
                    "subgroup_type": subgroup_type,
                    "subgroup": subgroup,
                    "n": len(data),
                    "cases": cases,
                    "controls": controls,
                    "auc": auc_value,
                    "auc_se": auc_se,
                    "auc_ci_lower": max(0.0, auc_value - 1.96 * auc_se)
                    if np.isfinite(auc_se)
                    else np.nan,
                    "auc_ci_upper": min(1.0, auc_value + 1.96 * auc_se)
                    if np.isfinite(auc_se)
                    else np.nan,
                    "brier": brier,
                    "brier_se": brier_se,
                    "brier_ci_lower": max(0.0, brier - 1.96 * brier_se)
                    if np.isfinite(brier_se)
                    else np.nan,
                    "brier_ci_upper": min(1.0, brier + 1.96 * brier_se)
                    if np.isfinite(brier_se)
                    else np.nan,
                }
            )

    return pd.DataFrame(rows)


def decision_curve_rows(oof: pd.DataFrame, thresholds: np.ndarray) -> pd.DataFrame:
    rows = []
    subgroup_cols = {
        "age_band": "age_band",
        "sex": "sex",
        "apoe_polymorphism": "apoe_polymorphism",
    }

    for subgroup_type, column in subgroup_cols.items():
        grouped = oof[oof[column].notna()].groupby(column, observed=True, sort=False)
        for subgroup, data in grouped:
            y_true = data["y_true"].to_numpy().astype(int)
            y_prob = data["y_prob"].to_numpy().astype(float)
            n = len(data)
            prevalence = float(y_true.mean()) if n else np.nan
            for threshold in thresholds:
                odds = threshold / (1 - threshold)
                y_pred = (y_prob >= threshold).astype(int)
                model_contrib = ((y_pred == 1) & (y_true == 1)).astype(float)
                model_contrib -= odds * ((y_pred == 1) & (y_true == 0)).astype(float)
                all_contrib = (y_true == 1).astype(float)
                all_contrib -= odds * (y_true == 0).astype(float)
                net_benefit, net_benefit_se = mean_metric_with_se(model_contrib)
                nb_all, nb_all_se = mean_metric_with_se(all_contrib)
                rows.append(
                    {
                        "outcome": data["outcome"].iloc[0],
                        "modality": data["modality"].iloc[0],
                        "subgroup_type": subgroup_type,
                        "subgroup": subgroup,
                        "threshold": threshold,
                        "n": n,
                        "cases": int(y_true.sum()),
                        "controls": int(n - y_true.sum()),
                        "prevalence": prevalence,
                        "net_benefit": net_benefit,
                        "net_benefit_se": net_benefit_se,
                        "net_benefit_ci_lower": net_benefit - 1.96 * net_benefit_se
                        if np.isfinite(net_benefit_se)
                        else np.nan,
                        "net_benefit_ci_upper": net_benefit + 1.96 * net_benefit_se
                        if np.isfinite(net_benefit_se)
                        else np.nan,
                        "net_benefit_treat_all": nb_all,
                        "net_benefit_treat_all_se": nb_all_se,
                        "net_benefit_treat_none": 0.0,
                    }
                )

    return pd.DataFrame(rows)


def dersimonian_laird(
    estimates: np.ndarray, standard_errors: np.ndarray
) -> dict[str, float]:
    estimates = np.asarray(estimates, dtype=float)
    standard_errors = np.asarray(standard_errors, dtype=float)
    variances = standard_errors**2
    keep = np.isfinite(estimates) & np.isfinite(variances) & (variances > 0)
    estimates = estimates[keep]
    variances = variances[keep]
    k = len(estimates)
    if k < 2:
        return {
            "k": k,
            "q": np.nan,
            "q_df": np.nan,
            "q_pvalue": np.nan,
            "i2": np.nan,
            "tau2": np.nan,
            "tau": np.nan,
            "fixed_effect": np.nan,
            "random_effect": np.nan,
        }

    weights = 1 / variances
    fixed_effect = float(np.sum(weights * estimates) / np.sum(weights))
    q = float(np.sum(weights * (estimates - fixed_effect) ** 2))
    q_df = k - 1
    c_value = float(np.sum(weights) - np.sum(weights**2) / np.sum(weights))
    tau2 = max(0.0, (q - q_df) / c_value) if c_value > 0 else 0.0
    random_weights = 1 / (variances + tau2)
    random_effect = float(np.sum(random_weights * estimates) / np.sum(random_weights))
    i2 = max(0.0, (q - q_df) / q) * 100 if q > 0 else 0.0
    return {
        "k": k,
        "q": q,
        "q_df": q_df,
        "q_pvalue": float(chi2.sf(q, q_df)),
        "i2": i2,
        "tau2": tau2,
        "tau": float(np.sqrt(tau2)),
        "fixed_effect": fixed_effect,
        "random_effect": random_effect,
    }


def heterogeneity_from_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, data in metrics.groupby(["outcome", "modality", "subgroup_type"], sort=False):
        outcome, modality, subgroup_type = keys
        for metric_name in ("auc", "brier"):
            stats = dersimonian_laird(
                data[metric_name].to_numpy(), data[f"{metric_name}_se"].to_numpy()
            )
            rows.append(
                {
                    "outcome": outcome,
                    "modality": modality,
                    "subgroup_type": subgroup_type,
                    "metric": metric_name,
                    "threshold": np.nan,
                    **stats,
                }
            )
    return pd.DataFrame(rows)


def heterogeneity_from_decision_curves(decision_curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["outcome", "modality", "subgroup_type", "threshold"]
    for keys, data in decision_curves.groupby(group_cols, sort=False):
        outcome, modality, subgroup_type, threshold = keys
        stats = dersimonian_laird(
            data["net_benefit"].to_numpy(), data["net_benefit_se"].to_numpy()
        )
        rows.append(
            {
                "outcome": outcome,
                "modality": modality,
                "subgroup_type": subgroup_type,
                "metric": "net_benefit",
                "threshold": threshold,
                **stats,
            }
        )
    return pd.DataFrame(rows)


def make_decision_curve_plots(decision_curves: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "decision_curve_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    group_cols = ["outcome", "modality", "subgroup_type"]

    for keys, data in decision_curves.groupby(group_cols, sort=False):
        outcome, modality, subgroup_type = keys
        fig, ax = plt.subplots(figsize=(7, 5))
        for subgroup, subgroup_data in data.groupby("subgroup", sort=False):
            subgroup_data = subgroup_data.sort_values("threshold")
            ax.plot(
                subgroup_data["threshold"],
                subgroup_data["net_benefit"],
                linewidth=2,
                label=str(subgroup),
            )
        first_subgroup = data.sort_values("threshold").groupby("threshold").first()
        ax.plot(
            first_subgroup.index,
            first_subgroup["net_benefit_treat_all"],
            color="0.45",
            linestyle="--",
            linewidth=1.5,
            label="Treat all",
        )
        ax.axhline(0, color="black", linestyle=":", linewidth=1.2, label="Treat none")
        ax.set_xlabel("Threshold probability")
        ax.set_ylabel("Net benefit")
        ax.set_title(f"{outcome}: {modality}, {subgroup_type}")
        ax.legend(frameon=False, fontsize=9)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{outcome}_{modality}_{subgroup_type}.png", dpi=300)
        plt.close(fig)


def dca_thresholds(config: AnalysisConfig) -> np.ndarray:
    threshold_count = int(
        np.floor(
            (config.dca_threshold_max - config.dca_threshold_min)
            / config.dca_threshold_step
        )
        + 1
    )
    thresholds = config.dca_threshold_min + np.arange(threshold_count) * config.dca_threshold_step
    return np.round(thresholds, 10)


def output_dir_for_config(config: AnalysisConfig) -> Path:
    age_label = f"agecutoff_{config.age_cutoff}" if config.age_cutoff else "all_ages"
    output_dir = (
        config.output_root
        / config.experiment
        / config.metric
        / config.model
        / age_label
    )
    if config.age_bins != DEFAULT_AGE_BINS:
        output_dir = output_dir / age_bins_label(config.age_bins)
    return output_dir


def write_manifest(config: AnalysisConfig, output_dir: Path) -> None:
    manifest = {
        "results_root": str(config.results_root),
        "tidy_root": str(config.tidy_root),
        "metadata_root": str(config.metadata_root),
        "acd_path": str(config.acd_path),
        "outcomes": list(config.outcomes),
        "modalities": list(config.modalities),
        "experiment": config.experiment,
        "metric": config.metric,
        "model": config.model,
        "age_cutoff": config.age_cutoff,
        "age_bins": list(config.age_bins),
        "dca_threshold_min": config.dca_threshold_min,
        "dca_threshold_max": config.dca_threshold_max,
        "dca_threshold_step": config.dca_threshold_step,
        "strict_label_check": config.strict_label_check,
    }
    with (output_dir / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)


def parse_args() -> AnalysisConfig:
    parser = argparse.ArgumentParser(
        description="Calculate UK Biobank subgroup metrics and heterogeneity statistics."
    )
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "results" / "UKBiobank")
    parser.add_argument(
        "--tidy-root",
        type=Path,
        default=REPO_ROOT / "tidy_data" / "UKBiobank" / "dementia",
    )
    parser.add_argument("--metadata-root", type=Path, default=REPO_ROOT / "metadata")
    parser.add_argument(
        "--acd-path",
        type=Path,
        default=REPO_ROOT.parent / "proj_idp" / "tidy_data" / "acd" / "allcausedementia.parquet",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "results" / "UKBiobank" / "formal_heterogeneity",
    )
    parser.add_argument("--outcomes", nargs="+", default=list(DEFAULT_OUTCOMES))
    parser.add_argument("--modalities", nargs="+", default=list(DEFAULT_MODALITIES))
    parser.add_argument("--experiment", default="demographics_modality_lancet2024")
    parser.add_argument("--metric", default="log_loss")
    parser.add_argument("--model", default="lgbm")
    parser.add_argument(
        "--age-cutoff",
        default="65",
        help="Use matching agecutoff_* result folders. Use 'none' for all-age folders.",
    )
    parser.add_argument(
        "--age-bins",
        nargs="+",
        type=int,
        default=list(DEFAULT_AGE_BINS),
        help=(
            "Lower bounds for left-closed age bands. Example: "
            "--age-bins 65 70 75 80 85 gives 65-70, 70-75, 75-80, 80-85, 85+."
        ),
    )
    parser.add_argument("--dca-threshold-min", type=float, default=0.01)
    parser.add_argument("--dca-threshold-max", type=float, default=0.50)
    parser.add_argument("--dca-threshold-step", type=float, default=0.01)
    parser.add_argument(
        "--no-strict-label-check",
        action="store_true",
        help="Do not require saved labels to exactly match reconstructed split labels.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip writing decision-curve PNGs.",
    )
    args = parser.parse_args()
    age_bins = tuple(args.age_bins)
    if len(age_bins) < 2:
        parser.error("--age-bins requires at least two cutpoints")
    if any(stop <= start for start, stop in zip(age_bins[:-1], age_bins[1:])):
        parser.error("--age-bins values must be strictly increasing")

    return AnalysisConfig(
        results_root=args.results_root,
        tidy_root=args.tidy_root,
        metadata_root=args.metadata_root,
        acd_path=args.acd_path,
        output_root=args.output_root,
        outcomes=tuple(args.outcomes),
        modalities=tuple(args.modalities),
        experiment=args.experiment,
        metric=args.metric,
        model=args.model,
        age_cutoff=parse_none_int(args.age_cutoff),
        age_bins=age_bins,
        dca_threshold_min=args.dca_threshold_min,
        dca_threshold_max=args.dca_threshold_max,
        dca_threshold_step=args.dca_threshold_step,
        strict_label_check=not args.no_strict_label_check,
        make_plots=not args.no_plots,
    )


def main() -> None:
    config = parse_args()
    output_dir = output_dir_for_config(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_oof = []
    all_metrics = []
    all_decision_curves = []
    thresholds = dca_thresholds(config)

    for outcome in config.outcomes:
        for modality in config.modalities:
            print(f"Processing {outcome}/{modality}")
            oof = assemble_oof_predictions(config, outcome, modality)
            all_oof.append(oof)
            all_metrics.append(subgroup_metric_rows(oof))
            all_decision_curves.append(decision_curve_rows(oof, thresholds))

    oof_df = pd.concat(all_oof, ignore_index=True)
    metrics_df = pd.concat(all_metrics, ignore_index=True)
    decision_curves_df = pd.concat(all_decision_curves, ignore_index=True)
    heterogeneity_df = pd.concat(
        [
            heterogeneity_from_metrics(metrics_df),
            heterogeneity_from_decision_curves(decision_curves_df),
        ],
        ignore_index=True,
    )

    oof_columns = [
        "outcome",
        "modality",
        "region_index",
        "region",
        "eid",
        "age",
        "sex",
        "apoe_polymorphism",
        "age_band",
        "y_true",
        "y_prob",
    ]
    oof_df[oof_columns].to_csv(output_dir / "oof_predictions_with_subgroups.csv", index=False)
    metrics_df.to_csv(output_dir / "subgroup_metrics.csv", index=False)
    decision_curves_df.to_csv(output_dir / "decision_curves.csv", index=False)
    heterogeneity_df.to_csv(output_dir / "heterogeneity_summary.csv", index=False)
    write_manifest(config, output_dir)

    if config.make_plots:
        make_decision_curve_plots(decision_curves_df, output_dir)

    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
