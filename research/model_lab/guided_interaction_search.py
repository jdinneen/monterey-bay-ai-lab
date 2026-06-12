#!/usr/bin/env python3
"""Bounded exhaustive interaction search for Monterey Bay AI Lab.

This is the practical version of "test every permutation": rank candidate
features first, then exhaustively score pairs/triples within a declared bound.
It is additive research tooling and never mutates source data.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "model_lab" / "guided_interaction_search"
DEFAULT_LEAKAGE_NAME_TOKENS = (
    "advisory",
    "bacteria_result",
    "enterococcus",
    "e_coli",
    "exceed",
    "fecal_coliform",
    "label",
    "outcome",
    "pred",
    "prob",
    "result",
    "target",
    "total_coliform",
    "truth",
)


@dataclass(frozen=True)
class SearchConfig:
    target: str
    time_col: str | None = None
    max_rows: int = 100_000
    top_features: int = 200
    top_triple_features: int = 40
    max_pair_candidates: int = 20_000
    max_triple_candidates: int = 20_000
    min_valid_rows: int = 200
    random_state: int = 42
    exclude_columns: tuple[str, ...] = ()
    allow_leakage_named_features: bool = False


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    raise ValueError(f"unsupported input suffix: {path.suffix}")


def coerce_binary_target(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype("int8")
    numeric = pd.to_numeric(series, errors="coerce")
    unique = set(numeric.dropna().unique().tolist())
    if unique.issubset({0, 1, 0.0, 1.0}):
        return numeric.astype("float32")
    return (numeric > 0).astype("float32")


def is_leakage_named_feature(column: str, target: str) -> bool:
    name = column.strip().lower()
    target_name = target.strip().lower()
    if name == target_name:
        return True
    return any(token in name for token in DEFAULT_LEAKAGE_NAME_TOKENS)


def numeric_feature_frame(df: pd.DataFrame, target: str, time_col: str | None, cfg: SearchConfig) -> pd.DataFrame:
    excluded = {target, *cfg.exclude_columns}
    if time_col:
        excluded.add(time_col)
    cols: list[str] = []
    for col in df.columns:
        if col in excluded:
            continue
        if not cfg.allow_leakage_named_features and is_leakage_named_feature(str(col), target):
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().sum() < 10:
            continue
        if float(series.nunique(dropna=True)) <= 1:
            continue
        cols.append(str(col))
    out = pd.DataFrame({col: pd.to_numeric(df[col], errors="coerce") for col in cols})
    return out.replace([np.inf, -np.inf], np.nan)


def sample_rows(df: pd.DataFrame, cfg: SearchConfig) -> pd.DataFrame:
    if len(df) <= cfg.max_rows:
        return df.copy()
    if cfg.time_col and cfg.time_col in df.columns:
        ordered = df.sort_values(cfg.time_col)
        return ordered.iloc[-cfg.max_rows:].copy()
    return df.sample(cfg.max_rows, random_state=cfg.random_state).copy()


def split_indices(df: pd.DataFrame, y: pd.Series, cfg: SearchConfig) -> tuple[np.ndarray, np.ndarray]:
    valid_idx = np.flatnonzero(y.notna().to_numpy())
    if cfg.time_col and cfg.time_col in df.columns:
        ordered = df.iloc[valid_idx].sort_values(cfg.time_col).index.to_numpy()
        cut = max(1, int(len(ordered) * 0.8))
        return ordered[:cut], ordered[cut:]
    train_idx, test_idx = train_test_split(
        valid_idx,
        test_size=0.2,
        random_state=cfg.random_state,
        stratify=y.iloc[valid_idx] if y.iloc[valid_idx].nunique(dropna=True) == 2 else None,
    )
    return np.asarray(train_idx), np.asarray(test_idx)


def fill_feature(series: pd.Series, train_idx: np.ndarray) -> pd.Series:
    median = float(series.iloc[train_idx].median()) if series.iloc[train_idx].notna().any() else 0.0
    return series.fillna(median).astype("float32")


def univariate_scores(x: pd.DataFrame, y: pd.Series, train_idx: np.ndarray, test_idx: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    y_train = y.iloc[train_idx].astype(int)
    y_test = y.iloc[test_idx].astype(int)
    for col in x.columns:
        feature = fill_feature(x[col], train_idx)
        if feature.iloc[test_idx].nunique() <= 1:
            continue
        train_score = feature.iloc[train_idx].to_numpy()
        test_score = feature.iloc[test_idx].to_numpy()
        direction = 1.0
        try:
            ap_pos = average_precision_score(y_train, train_score)
            ap_neg = average_precision_score(y_train, -train_score)
            if ap_neg > ap_pos:
                direction = -1.0
        except ValueError:
            continue
        score = direction * test_score
        try:
            auc = roc_auc_score(y_test, score)
        except ValueError:
            auc = math.nan
        rows.append(
            {
                "feature": col,
                "ap": float(average_precision_score(y_test, score)),
                "roc_auc": None if math.isnan(auc) else float(auc),
                "direction": direction,
                "valid_rows": int(feature.notna().sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["ap", "feature"], ascending=[False, True]).reset_index(drop=True)


def interaction_values(x: pd.DataFrame, features: tuple[str, ...], train_idx: np.ndarray) -> pd.Series:
    vals = [fill_feature(x[f], train_idx) for f in features]
    out = vals[0].copy()
    for value in vals[1:]:
        out = out * value
    return out.astype("float32")


def bounded_combinations(features: list[str], order: int, max_candidates: int) -> Iterable[tuple[str, ...]]:
    count = 0
    for combo in itertools.combinations(features, order):
        if count >= max_candidates:
            break
        count += 1
        yield combo


def score_interactions(
    x: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    features: list[str],
    order: int,
    max_candidates: int,
) -> pd.DataFrame:
    y_train = y.iloc[train_idx].astype(int)
    y_test = y.iloc[test_idx].astype(int)
    rows: list[dict[str, object]] = []
    for combo in bounded_combinations(features, order, max_candidates):
        value = interaction_values(x, combo, train_idx)
        if value.iloc[test_idx].nunique() <= 1:
            continue
        train_score = value.iloc[train_idx].to_numpy()
        test_score = value.iloc[test_idx].to_numpy()
        try:
            ap_pos = average_precision_score(y_train, train_score)
            ap_neg = average_precision_score(y_train, -train_score)
        except ValueError:
            continue
        direction = -1.0 if ap_neg > ap_pos else 1.0
        score = direction * test_score
        try:
            auc = roc_auc_score(y_test, score)
        except ValueError:
            auc = math.nan
        rows.append(
            {
                "interaction": "*".join(combo),
                "order": order,
                "features": list(combo),
                "ap": float(average_precision_score(y_test, score)),
                "roc_auc": None if math.isnan(auc) else float(auc),
                "direction": direction,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["interaction", "order", "features", "ap", "roc_auc", "direction"])
    return pd.DataFrame(rows).sort_values(["ap", "interaction"], ascending=[False, True]).reset_index(drop=True)


def fit_probe(x: pd.DataFrame, y: pd.Series, train_idx: np.ndarray, test_idx: np.ndarray, features: list[str]) -> dict[str, object]:
    if not features:
        return {"status": "no_features"}
    frame = pd.DataFrame({f: fill_feature(x[f], train_idx) for f in features})
    clf = HistGradientBoostingClassifier(max_iter=80, learning_rate=0.06, random_state=42)
    clf.fit(frame.iloc[train_idx], y.iloc[train_idx].astype(int))
    proba = clf.predict_proba(frame.iloc[test_idx])[:, 1]
    out = {"ap": float(average_precision_score(y.iloc[test_idx].astype(int), proba))}
    try:
        out["roc_auc"] = float(roc_auc_score(y.iloc[test_idx].astype(int), proba))
    except ValueError:
        out["roc_auc"] = None
    return out


def run_search(input_path: Path, output_dir: Path, cfg: SearchConfig) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = sample_rows(read_table(input_path), cfg)
    if cfg.target not in df.columns:
        raise ValueError(f"target column not found: {cfg.target}")
    y = coerce_binary_target(df[cfg.target])
    valid = y.notna()
    df = df.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)
    if y.nunique(dropna=True) != 2:
        raise ValueError("target must contain both classes after filtering")
    x = numeric_feature_frame(df, cfg.target, cfg.time_col, cfg)
    if x.empty:
        raise ValueError("no numeric candidate features")
    train_idx, test_idx = split_indices(df, y, cfg)
    if len(test_idx) < cfg.min_valid_rows:
        raise ValueError(f"test split too small: {len(test_idx)} < {cfg.min_valid_rows}")

    uni = univariate_scores(x, y, train_idx, test_idx)
    ranked = uni["feature"].head(cfg.top_features).tolist()
    pair_scores = score_interactions(x, y, train_idx, test_idx, ranked, 2, cfg.max_pair_candidates)
    triple_features = ranked[: cfg.top_triple_features]
    triple_scores = score_interactions(x, y, train_idx, test_idx, triple_features, 3, cfg.max_triple_candidates)
    interaction_frames = [frame for frame in [pair_scores, triple_scores] if not frame.empty]
    if interaction_frames:
        interactions = pd.concat(interaction_frames, ignore_index=True)
        interactions = interactions.sort_values(["ap", "interaction"], ascending=[False, True]).reset_index(drop=True)
    else:
        interactions = pd.DataFrame(columns=["interaction", "order", "features", "ap", "roc_auc", "direction"])

    top_interaction_features = []
    for features in interactions.head(20).get("features", []):
        top_interaction_features.extend(features)
    probe_features = list(dict.fromkeys(ranked[: min(40, len(ranked))] + top_interaction_features))
    baseline_probe = fit_probe(x, y, train_idx, test_idx, ranked[: min(40, len(ranked))])
    interaction_probe = fit_probe(x, y, train_idx, test_idx, probe_features)

    uni.to_csv(output_dir / "univariate_scores.csv", index=False)
    interactions.to_json(output_dir / "interaction_scores.jsonl", orient="records", lines=True)
    interactions.head(200).to_csv(output_dir / "top_interactions.csv", index=False)

    best_uni_ap = float(uni["ap"].max()) if len(uni) else None
    best_interaction_ap = float(interactions["ap"].max()) if len(interactions) else None
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project": "Monterey Bay AI Lab",
        "value_gate": "DO_NOW: bounded exhaustive interaction search over ranked features with holdout scoring.",
        "input": str(input_path),
        "destructive": False,
        "rows": int(len(df)),
        "features_scanned": int(x.shape[1]),
        "leakage_named_features_allowed": bool(cfg.allow_leakage_named_features),
        "exclude_columns": list(cfg.exclude_columns),
        "ranked_features": int(len(ranked)),
        "pair_candidates_scored": int(len(pair_scores)),
        "triple_candidates_scored": int(len(triple_scores)),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "target_positive_rate": float(y.mean()),
        "best_univariate_ap": best_uni_ap,
        "best_interaction_ap": best_interaction_ap,
        "baseline_probe": baseline_probe,
        "interaction_probe": interaction_probe,
        "honesty_note": (
            "This is bounded exhaustive search, not all 2^N feature subsets. "
            "Survivors must still clear task-specific release gates."
        ),
        "artifacts": {
            "univariate_scores": str(output_dir / "univariate_scores.csv"),
            "interaction_scores": str(output_dir / "interaction_scores.jsonl"),
            "top_interactions": str(output_dir / "top_interactions.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bounded exhaustive interaction search.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--time-col", default=None)
    parser.add_argument("--max-rows", type=int, default=100_000)
    parser.add_argument("--top-features", type=int, default=200)
    parser.add_argument("--top-triple-features", type=int, default=40)
    parser.add_argument("--max-pair-candidates", type=int, default=20_000)
    parser.add_argument("--max-triple-candidates", type=int, default=20_000)
    parser.add_argument("--min-valid-rows", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--exclude-columns", nargs="*", default=[])
    parser.add_argument("--allow-leakage-named-features", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = SearchConfig(
        target=args.target,
        time_col=args.time_col,
        max_rows=args.max_rows,
        top_features=args.top_features,
        top_triple_features=args.top_triple_features,
        max_pair_candidates=args.max_pair_candidates,
        max_triple_candidates=args.max_triple_candidates,
        min_valid_rows=args.min_valid_rows,
        random_state=args.random_state,
        exclude_columns=tuple(args.exclude_columns),
        allow_leakage_named_features=args.allow_leakage_named_features,
    )
    summary = run_search(Path(args.input), Path(args.output_dir), cfg)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
