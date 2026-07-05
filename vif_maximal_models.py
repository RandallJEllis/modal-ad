#!/usr/bin/env python3
"""VIF diagnostics for UK Biobank maximal ML feature sets.

LightGBM does not have a coefficient design matrix in the same sense as the Cox
models, so this script diagnoses the maximal feature matrix used for training.
It reports rank/conditioning for the as-trained matrix and VIF after dropping one
reference level from detected one-hot groups, which is the closest analogue to
factor-aware VIF/GVIF in the R Cox models.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


LANCET_VARS = [
    "4700-0.0",
    "5901-0.0",
    "30780-0.0",
    "head_injury",
    "22038-0.0",
    "20161-0.0",
    "alcohol_consumption",
    "hypertension",
    "obesity",
    "diabetes",
    "hearing_loss",
    "depression",
    "freq_friends_family_visit",
    "24012-0.0",
    "24018-0.0",
    "24019-0.0",
    "24006-0.0",
    "24015-0.0",
    "24011-0.0",
    "2020-0.0_-3.0",
    "2020-0.0_-1.0",
    "2020-0.0_0.0",
    "2020-0.0_1.0",
    "2020-0.0_nan",
]

DEMOGRAPHIC_PREFIXES = [
    "31-0.0",
    "apoe",
    "max_educ_complete",
    "845-0.0",
    "21000-0.0",
]

CATEGORY_SUFFIX_RE = re.compile(r"_(?:-?\d+(?:\.\d+)?|nan)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate VIF diagnostics for UK Biobank maximal feature matrices."
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=["cognitive_tests", "proteomics", "neuroimaging"],
        choices=["cognitive_tests", "proteomics", "neuroimaging"],
    )
    parser.add_argument(
        "--data-root",
        default="../../tidy_data/UKBiobank/dementia",
        help="Path to tidy_data/UKBiobank/dementia, relative to this script.",
    )
    parser.add_argument(
        "--output-dir",
        default="../../results/UKBiobank/vif_diagnostics/demographics_modality_lancet2024",
        help="Output directory, relative to this script.",
    )
    parser.add_argument(
        "--outcome",
        default="dementia",
        choices=["dementia", "alzheimers"],
        help="Outcome row set to diagnose. Alzheimer's keeps controls plus Alzheimer disease cases.",
    )
    parser.add_argument(
        "--acd-path",
        default="../../../proj_idp/tidy_data/acd/allcausedementia.parquet",
        help="All-cause dementia source file used to identify Alzheimer disease cases.",
    )
    parser.add_argument(
        "--age-cutoff",
        type=float,
        default=65,
        help="Subset to participants at or above this age. Use 0 to disable.",
    )
    parser.add_argument(
        "--rcond",
        type=float,
        default=1e-10,
        help="Reciprocal condition threshold used for rank and pseudo-inverse fallback.",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Rebuild the combined summary from existing per-modality outputs.",
    )
    return parser.parse_args()


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def resolve_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (script_dir() / p).resolve()


def parquet_columns(path: Path) -> list[str]:
    return pq.ParquetFile(path).schema_arrow.names


def pull_columns_by_prefix(columns: Iterable[str], prefixes: Iterable[str]) -> list[str]:
    return [col for col in columns if any(col.startswith(prefix) for prefix in prefixes)]


def pull_columns_by_suffix(columns: Iterable[str], suffixes: Iterable[str]) -> list[str]:
    return [col for col in columns if any(col.endswith(suffix) for suffix in suffixes)]


def columns_with_prefix(columns: Iterable[str], prefixes: Iterable[str]) -> list[str]:
    return [col for col in columns if any(col.startswith(prefix) for prefix in prefixes)]


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def dedupe_keep_order(columns: Iterable[str]) -> list[str]:
    seen = set()
    kept = []
    for col in columns:
        if col not in seen:
            seen.add(col)
            kept.append(col)
    return kept


def maximal_columns(data_root: Path, modality: str, all_columns: list[str]) -> list[str]:
    data_instance = 2 if modality == "neuroimaging" else 0
    demographic_cols = pull_columns_by_prefix(
        all_columns,
        [f"21003-{data_instance}.0", *DEMOGRAPHIC_PREFIXES],
    )

    if modality == "proteomics":
        modality_cols = pull_columns_by_suffix(all_columns, ["-0"])
    elif modality == "neuroimaging":
        modality_cols = load_pickle(data_root / "neuroimaging" / "idp_variables.pkl")
    elif modality == "cognitive_tests":
        modality_cols = load_pickle(
            data_root / "cognitive_tests" / "cognitive_columns.pkl"
        )
    else:
        raise ValueError(f"Unsupported modality: {modality}")

    selected = dedupe_keep_order([*demographic_cols, *modality_cols, *LANCET_VARS])
    missing = [col for col in selected if col not in all_columns]
    if missing:
        print(
            f"[{modality}] Dropping {len(missing)} requested columns absent from X.parquet: "
            f"{missing[:10]}"
        )
    return [col for col in selected if col in all_columns]


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
    acd_cols = columns_with_prefix(parquet_columns(acd_path), prefixes)
    acd = pd.read_parquet(acd_path, columns=acd_cols)
    acd = acd[acd["eid"].isin(eligible_eids)]

    disease_source = acd[columns_with_prefix(acd.columns, ["eid", "42021"])]
    keep_disease_source = disease_source.drop(columns=["eid"]).isin([1, 2, 11, 12, 21, 22]).any(axis=1)

    icd_source = acd[columns_with_prefix(acd.columns, ["eid", "131037", "130837"])]
    keep_icd_source = icd_source.drop(columns=["eid"]).isin([20, 21, 30, 31, 40, 41, 51]).any(axis=1)

    return set(disease_source.loc[keep_disease_source, "eid"]).union(
        set(icd_source.loc[keep_icd_source, "eid"])
    )


def read_selected_matrix(
    x_path: Path,
    columns: list[str],
    age_cutoff: float | None,
    outcome: str,
    acd_path: Path | None = None,
) -> pd.DataFrame:
    age_cols = [col for col in columns if col.startswith("21003-")]
    read_columns = dedupe_keep_order(["eid", *columns])
    df = pd.read_parquet(x_path, columns=read_columns)
    y_path = x_path.parent / "y.npy"

    if outcome == "alzheimers":
        if acd_path is None or not acd_path.exists():
            raise FileNotFoundError(f"Could not find ACD file for Alzheimer filtering: {acd_path}")
        y = np.load(y_path)
        if len(y) != len(df):
            raise ValueError(f"{y_path} length ({len(y)}) does not match {x_path} rows ({len(df)}).")
        alzheimer_eids = load_alzheimer_eids(acd_path, df["eid"])
        keep = (y == 0) | df["eid"].isin(alzheimer_eids).to_numpy()
        df = df.loc[keep].reset_index(drop=True)

    if age_cutoff is not None:
        if len(age_cols) != 1:
            raise ValueError(f"Expected one age column, found {age_cols}")
        df = df.loc[df[age_cols[0]] >= age_cutoff].reset_index(drop=True)
    return df.drop(columns=["eid"])


def categorical_group_key(column: str) -> str | None:
    if not CATEGORY_SUFFIX_RE.search(column):
        return None
    return column.rsplit("_", 1)[0]


def is_binary_like(series: pd.Series) -> bool:
    values = pd.to_numeric(series, errors="coerce").dropna().unique()
    if len(values) == 0:
        return False
    return set(values.tolist()).issubset({0, 1, 0.0, 1.0})


def detect_one_hot_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    candidate_groups: dict[str, list[str]] = {}
    for col in df.columns:
        key = categorical_group_key(col)
        if key is None:
            continue
        candidate_groups.setdefault(key, []).append(col)

    one_hot_groups: dict[str, list[str]] = {}
    for key, cols in candidate_groups.items():
        if len(cols) < 2:
            continue
        if not all(is_binary_like(df[col]) for col in cols):
            continue
        row_sums = df[cols].sum(axis=1, skipna=True)
        if row_sums.max() <= 1.000001 and row_sums.mean() > 0.5:
            one_hot_groups[key] = cols
    return one_hot_groups


def drop_reference_levels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = detect_one_hot_groups(df)
    dropped_rows = []
    cols_to_drop = []

    for group, cols in sorted(groups.items()):
        sorted_cols = sorted(cols)
        nan_cols = [col for col in sorted_cols if col.endswith("_nan")]
        reference = nan_cols[0] if nan_cols else sorted_cols[-1]
        cols_to_drop.append(reference)
        dropped_rows.append(
            {
                "one_hot_group": group,
                "reference_column": reference,
                "n_group_columns": len(cols),
                "group_columns": json.dumps(sorted_cols),
            }
        )

    reduced = df.drop(columns=cols_to_drop)
    return reduced, pd.DataFrame(dropped_rows)


def coerce_numeric(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric = df.apply(pd.to_numeric, errors="coerce")
    missing_summary = pd.DataFrame(
        {
            "feature": numeric.columns,
            "missing_fraction": numeric.isna().mean().values,
        }
    )
    return numeric, missing_summary


def remove_unusable_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    means = df.mean(axis=0, skipna=True)
    all_missing = means.isna()
    df = df.loc[:, ~all_missing]
    means = means.loc[~all_missing]
    df = df.fillna(means)

    std = df.std(axis=0, ddof=1)
    zero_variance = std <= 0
    dropped = pd.DataFrame(
        {
            "feature": std.index[zero_variance].tolist(),
            "drop_reason": "all_missing_or_zero_variance",
        }
    )
    df = df.loc[:, ~zero_variance]
    return df, dropped


def matrix_diagnostics(df: pd.DataFrame, rcond: float) -> dict[str, float | int | bool]:
    usable, _ = remove_unusable_features(df)
    n_rows, n_features = usable.shape
    if n_features == 0:
        return {
            "n_rows": n_rows,
            "n_features": 0,
            "rank": 0,
            "rank_deficiency": 0,
            "min_eigenvalue": np.nan,
            "max_eigenvalue": np.nan,
            "condition_number": np.nan,
        }

    centered = usable - usable.mean(axis=0)
    std = centered.std(axis=0, ddof=1)
    z = (centered / std).to_numpy(dtype=np.float32, copy=True)
    corr = (z.T @ z).astype(np.float64) / max(n_rows - 1, 1)
    corr = (corr + corr.T) / 2
    eigvals = np.linalg.eigvalsh(corr)
    max_eig = float(np.max(eigvals)) if eigvals.size else np.nan
    tol = max_eig * max(n_rows, n_features) * rcond if eigvals.size else 0
    rank = int(np.sum(eigvals > tol))
    positive = eigvals[eigvals > tol]
    min_pos = float(np.min(positive)) if positive.size else np.nan
    condition = float(max_eig / min_pos) if positive.size else np.inf
    return {
        "n_rows": n_rows,
        "n_features": n_features,
        "rank": rank,
        "rank_deficiency": int(n_features - rank),
        "min_eigenvalue": float(np.min(eigvals)) if eigvals.size else np.nan,
        "max_eigenvalue": max_eig,
        "condition_number": condition,
    }


def correlation_eigendecomposition(
    df: pd.DataFrame, rcond: float
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, int, int, float]:
    usable, _ = remove_unusable_features(df)
    n_rows, n_features = usable.shape
    if n_features == 0:
        return usable, np.array([]), np.empty((0, 0)), 0, 0, 0

    centered = usable - usable.mean(axis=0)
    std = centered.std(axis=0, ddof=1)
    z = (centered / std).to_numpy(dtype=np.float32, copy=True)
    corr = (z.T @ z).astype(np.float64) / max(n_rows - 1, 1)
    corr = (corr + corr.T) / 2
    eigvals, eigvecs = np.linalg.eigh(corr)
    max_eig = float(np.max(eigvals)) if eigvals.size else 0
    tol = max_eig * max(n_rows, n_features) * rcond
    rank = int(np.sum(eigvals > tol))
    rank_deficiency = int(n_features - rank)
    return usable, eigvals, eigvecs, rank, rank_deficiency, tol


def drop_linear_dependencies(
    df: pd.DataFrame, rcond: float, max_iterations: int = 25
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reduced = df.copy()
    dropped_rows = []

    for iteration in range(1, max_iterations + 1):
        usable, eigvals, eigvecs, rank, rank_deficiency, tol = correlation_eigendecomposition(
            reduced, rcond
        )
        if rank_deficiency == 0:
            break

        low_eigen_indices = np.where(eigvals <= tol)[0]
        columns_to_drop = []
        for eig_idx in low_eigen_indices:
            loadings = np.abs(eigvecs[:, eig_idx])
            ordered = np.argsort(loadings)[::-1]
            for col_idx in ordered:
                candidate = usable.columns[col_idx]
                if candidate not in columns_to_drop:
                    columns_to_drop.append(candidate)
                    break

        if not columns_to_drop:
            break

        for col in columns_to_drop:
            dropped_rows.append(
                {
                    "feature": col,
                    "drop_reason": "linear_dependency_after_reference_coding",
                    "iteration": iteration,
                    "rank_before_drop": rank,
                    "rank_deficiency_before_drop": rank_deficiency,
                }
            )
        reduced = reduced.drop(columns=columns_to_drop)

    return reduced, pd.DataFrame(dropped_rows)


def calculate_vif(df: pd.DataFrame, rcond: float) -> tuple[pd.DataFrame, dict[str, float | int | bool]]:
    usable, dropped = remove_unusable_features(df)
    n_rows, n_features = usable.shape
    if n_features == 0:
        return pd.DataFrame(), {
            "n_rows": n_rows,
            "n_features_for_vif": 0,
            "used_pseudoinverse": False,
            "rank": 0,
            "rank_deficiency": 0,
            "max_vif": np.nan,
            "n_vif_gt_5": 0,
            "n_vif_gt_10": 0,
            "n_dropped_unusable": len(dropped),
        }

    centered = usable - usable.mean(axis=0)
    std = centered.std(axis=0, ddof=1)
    z = (centered / std).to_numpy(dtype=np.float32, copy=True)
    corr = (z.T @ z).astype(np.float64) / max(n_rows - 1, 1)
    corr = (corr + corr.T) / 2

    eigvals = np.linalg.eigvalsh(corr)
    max_eig = float(np.max(eigvals)) if eigvals.size else np.nan
    tol = max_eig * max(n_rows, n_features) * rcond if eigvals.size else 0
    rank = int(np.sum(eigvals > tol))
    rank_deficiency = int(n_features - rank)

    used_pseudoinverse = False
    try:
        inv_corr = np.linalg.inv(corr)
    except np.linalg.LinAlgError:
        inv_corr = np.linalg.pinv(corr, rcond=rcond)
        used_pseudoinverse = True

    if rank_deficiency > 0:
        used_pseudoinverse = True
        inv_corr = np.linalg.pinv(corr, rcond=rcond)

    vif = np.diag(inv_corr)
    vif_df = pd.DataFrame(
        {
            "feature": usable.columns,
            "vif": vif,
        }
    ).sort_values("vif", ascending=False)

    metadata = {
        "n_rows": n_rows,
        "n_features_for_vif": n_features,
        "used_pseudoinverse": bool(used_pseudoinverse),
        "rank": rank,
        "rank_deficiency": rank_deficiency,
        "min_eigenvalue": float(np.min(eigvals)) if eigvals.size else np.nan,
        "max_eigenvalue": max_eig,
        "max_vif": float(np.nanmax(vif)) if len(vif) else np.nan,
        "mean_vif": float(np.nanmean(vif)) if len(vif) else np.nan,
        "median_vif": float(np.nanmedian(vif)) if len(vif) else np.nan,
        "n_vif_gt_5": int(np.sum(vif > 5)),
        "n_vif_gt_10": int(np.sum(vif > 10)),
        "n_dropped_unusable": int(len(dropped)),
    }
    return vif_df, metadata


def run_modality(
    modality: str,
    data_root: Path,
    output_dir: Path,
    age_cutoff: float | None,
    rcond: float,
    outcome: str,
    acd_path: Path | None,
) -> dict:
    modality_dir = data_root / modality
    x_path = modality_dir / "X.parquet"
    all_columns = parquet_columns(x_path)
    selected_columns = maximal_columns(data_root, modality, all_columns)
    df = read_selected_matrix(
        x_path,
        selected_columns,
        age_cutoff,
        outcome=outcome,
        acd_path=acd_path,
    )
    numeric, missing_summary = coerce_numeric(df)

    as_trained_diag = matrix_diagnostics(numeric, rcond=rcond)
    reference_coded, reference_columns = drop_reference_levels(numeric)
    vif_input, linear_dependency_columns = drop_linear_dependencies(
        reference_coded, rcond=rcond
    )
    vif_df, vif_meta = calculate_vif(vif_input, rcond=rcond)

    modality_output = output_dir / modality
    modality_output.mkdir(parents=True, exist_ok=True)
    missing_summary.to_csv(modality_output / "missingness_by_feature.csv", index=False)
    reference_columns.to_csv(
        modality_output / "one_hot_reference_columns.csv", index=False
    )
    linear_dependency_columns.to_csv(
        modality_output / "linear_dependency_columns.csv", index=False
    )
    vif_df.to_csv(modality_output / "vif_by_feature_reference_coded.csv", index=False)

    summary = {
        "outcome": outcome,
        "modality": modality,
        "experiment": "demographics_modality_lancet2024",
        "age_cutoff": age_cutoff,
        "n_rows_after_age_filter": len(numeric),
        "n_features_as_trained": len(selected_columns),
        "n_one_hot_reference_columns_dropped": len(reference_columns),
        "n_linear_dependency_columns_dropped": len(linear_dependency_columns),
        **{f"as_trained_{k}": v for k, v in as_trained_diag.items()},
        **vif_meta,
    }
    pd.DataFrame([summary]).to_csv(modality_output / "vif_summary.csv", index=False)
    return summary


def write_combined_summary(output_dir: Path) -> pd.DataFrame:
    summaries = []
    for summary_path in sorted(output_dir.glob("*/vif_summary.csv")):
        summaries.append(pd.read_csv(summary_path))
    if summaries:
        combined = pd.concat(summaries, ignore_index=True)
    else:
        combined = pd.DataFrame()
    combined.to_csv(output_dir / "ukbiobank_vif_summary.csv", index=False)
    return combined


def main() -> None:
    args = parse_args()
    data_root = resolve_path(args.data_root)
    base_output_dir = resolve_path(args.output_dir)
    if args.outcome != "dementia":
        base_output_dir = base_output_dir / args.outcome
    acd_path = resolve_path(args.acd_path)
    age_cutoff = None if args.age_cutoff == 0 else args.age_cutoff
    output_dir = (
        base_output_dir / f"agecutoff_{int(age_cutoff)}"
        if age_cutoff is not None
        else base_output_dir / "all_ages"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.summarize_only:
        for modality in args.modalities:
            print(f"Running UKB VIF diagnostics for {modality}")
            run_modality(
                modality=modality,
                data_root=data_root,
                output_dir=output_dir,
                age_cutoff=age_cutoff,
                rcond=args.rcond,
                outcome=args.outcome,
                acd_path=acd_path,
            )

    write_combined_summary(output_dir)
    print(f"Wrote UKB VIF diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
