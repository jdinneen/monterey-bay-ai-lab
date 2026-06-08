#!/usr/bin/env python
"""Build an enforceable promotion matrix for MBARI forecast models.

The matrix compares completed neural/lakehouse runs against persistence and the
current XGBoost forecast-v2 baseline by target/horizon. It is intentionally
conservative: until XGBoost and neural runs share one externally supplied split
contract, neural winners are marked ``candidate_split_mismatch`` rather than
``promote``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from mbari_lakehouse import read_forecast_metrics
from ops.seasonal_naive import seasonal_naive_table


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMOTION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PromotionConfig:
    project_root: Path
    min_observations: int = 20
    allow_split_mismatch: bool = False


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_xgb_baseline(project_root: Path) -> pd.DataFrame:
    lakehouse = _read_xgb_lakehouse_baseline(project_root)
    if not lakehouse.empty:
        return lakehouse
    path = project_root / "mbari_forecast_v2_results" / "leaderboard.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    required = {"target", "horizon_h", "rmse", "skill_rmse_vs_persistence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"XGBoost leaderboard missing columns: {sorted(missing)}")
    out = df.copy()
    out["target"] = out["target"].astype(str)
    out["horizon_h"] = pd.to_numeric(out["horizon_h"], errors="raise").astype(int)
    out["xgb_rmse"] = pd.to_numeric(out["rmse"], errors="raise")
    # XGBoost skill is stored as a fraction; neural skill is stored as percent.
    out["xgb_skill_vs_persistence_pct"] = (
        pd.to_numeric(out["skill_rmse_vs_persistence"], errors="raise") * 100.0
    )
    out["xgb_beats_persistence"] = out["xgb_skill_vs_persistence_pct"] > 0
    out["xgb_run_id"] = (
        out["run_id"].astype(str) if "run_id" in out.columns else "mbari_forecast_v2_results"
    )
    out["xgb_split_id"] = (
        out["split_id"].astype(str) if "split_id" in out.columns else "forecast_v2_internal_folds"
    )
    return out[
        [
            "target",
            "horizon_h",
            "xgb_run_id",
            "xgb_split_id",
            "xgb_rmse",
            "xgb_skill_vs_persistence_pct",
            "xgb_beats_persistence",
        ]
    ]


def _read_xgb_lakehouse_baseline(project_root: Path) -> pd.DataFrame:
    df = read_forecast_metrics(project_root, include_partitions=True)
    if df.empty:
        return pd.DataFrame()
    required = {"run_id", "split_id", "unique_id", "horizon_h", "model_rmse", "skill_vs_persistence_pct", "model"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    is_xgb = df["model"].astype(str).str.lower().isin({"xgboost", "xgb", "forecast_v2_xgboost"})
    is_xgb = is_xgb | df["run_id"].astype(str).str.lower().str.startswith("xgb")
    if not is_xgb.any():
        return pd.DataFrame()
    out = df[is_xgb].copy()
    out["target"] = out["unique_id"].astype(str)
    out["horizon_h"] = pd.to_numeric(out["horizon_h"], errors="raise").astype(int)
    out["xgb_run_id"] = out["run_id"].astype(str)
    out["xgb_split_id"] = out["split_id"].astype(str)
    out["xgb_rmse"] = pd.to_numeric(out["model_rmse"], errors="raise")
    out["xgb_skill_vs_persistence_pct"] = pd.to_numeric(out["skill_vs_persistence_pct"], errors="raise")
    out["xgb_beats_persistence"] = out["xgb_skill_vs_persistence_pct"] > 0
    # Keep the latest row by run appearance for a target/horizon/split. This
    # allows one baseline row per shared split while avoiding stale duplicates.
    out = out.drop_duplicates(subset=["target", "horizon_h", "xgb_split_id"], keep="last")
    return out[
        [
            "target",
            "horizon_h",
            "xgb_run_id",
            "xgb_split_id",
            "xgb_rmse",
            "xgb_skill_vs_persistence_pct",
            "xgb_beats_persistence",
        ]
    ]


def _read_candidate_metrics(project_root: Path) -> pd.DataFrame:
    df = read_forecast_metrics(project_root, include_partitions=True)
    if df.empty:
        return pd.DataFrame()
    required = {
        "run_id",
        "split_id",
        "unique_id",
        "horizon_h",
        "model_rmse",
        "persistence_rmse",
        "skill_vs_persistence_pct",
        "n",
        "model",
        "loss",
        "drivers_enabled",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"lakehouse metrics missing columns: {sorted(missing)}")
    out = df.copy()
    if "model" in out.columns:
        model_text = out["model"].astype(str).str.lower()
        run_text = out["run_id"].astype(str).str.lower()
        out = out[~(model_text.isin({"xgboost", "xgb", "forecast_v2_xgboost"}) | run_text.str.startswith("xgb"))].copy()
    out["target"] = out["unique_id"].astype(str)
    out["horizon_h"] = pd.to_numeric(out["horizon_h"], errors="raise").astype(int)
    out["candidate_skill_vs_persistence_pct"] = pd.to_numeric(
        out["skill_vs_persistence_pct"], errors="raise"
    )
    out["candidate_rmse"] = pd.to_numeric(out["model_rmse"], errors="raise")
    out["candidate_n"] = pd.to_numeric(out["n"], errors="coerce").fillna(0).astype(int)
    out["candidate_model"] = out["model"].astype(str)
    out["candidate_loss"] = out["loss"].astype(str)
    out["candidate_drivers_enabled"] = out["drivers_enabled"].astype(bool)
    # Best-naive evidence (skill vs the better of persistence and seasonal-naive)
    # is backfilled into the gold metrics table; carry it through if present.
    if "skill_vs_best_naive_pct" in out.columns:
        out["candidate_skill_vs_best_naive_pct"] = pd.to_numeric(out["skill_vs_best_naive_pct"], errors="coerce")
    else:
        out["candidate_skill_vs_best_naive_pct"] = pd.NA
    if "seasonal_naive_rmse" in out.columns:
        out["candidate_seasonal_naive_rmse"] = pd.to_numeric(out["seasonal_naive_rmse"], errors="coerce")
    else:
        out["candidate_seasonal_naive_rmse"] = pd.NA
    result = out[
        [
            "target",
            "horizon_h",
            "run_id",
            "split_id",
            "candidate_model",
            "candidate_loss",
            "candidate_drivers_enabled",
            "candidate_rmse",
            "persistence_rmse",
            "candidate_seasonal_naive_rmse",
            "candidate_skill_vs_persistence_pct",
            "candidate_skill_vs_best_naive_pct",
            "candidate_n",
        ]
    ].rename(columns={"run_id": "candidate_run_id", "split_id": "candidate_split_id"})
    if not result.empty:
        model = result["candidate_model"].astype(str).str.lower()
        split = result["candidate_split_id"].astype(str)
        superseding = result[
            model.isin({"chronos", "timesfm"})
            & split.ne("foundation_zero_shot_benchmark")
        ][["target", "horizon_h", "candidate_model"]].drop_duplicates()
        if not superseding.empty:
            stale = result[
                result["candidate_split_id"].astype(str).eq("foundation_zero_shot_benchmark")
            ][["target", "horizon_h", "candidate_model"]].merge(
                superseding.assign(_superseded=True),
                on=["target", "horizon_h", "candidate_model"],
                how="left",
            )["_superseded"].notna().to_numpy()
            stale_index = result[result["candidate_split_id"].astype(str).eq("foundation_zero_shot_benchmark")].index[stale]
            result = result.drop(index=stale_index)
    return result


def _ensure_best_naive(cand: pd.DataFrame, project_root: Path) -> pd.DataFrame:
    """Attach authoritative best-naive skill, recomputed from gold predictions.

    The gate must not depend on the backfilled gold column alone: a recursive
    metrics read can deduplicate the enriched aggregate against per-run
    partitions that lack the column and null it out. So whenever the gold
    prediction partitions are available we recompute the best-naive table
    directly and overlay it (authoritative). When predictions are unavailable
    (e.g. unit fixtures), we keep whatever the metrics table already provided.
    """
    seasonal = seasonal_naive_table(project_root)
    if seasonal.empty:
        return cand
    key = seasonal.rename(
        columns={
            "run_id": "candidate_run_id",
            "split_id": "candidate_split_id",
            "unique_id": "target",
        }
    )[["candidate_run_id", "candidate_split_id", "target", "horizon_h", "skill_vs_best_naive_pct", "seasonal_naive_rmse"]]
    key["horizon_h"] = pd.to_numeric(key["horizon_h"], errors="coerce").astype(int)
    merged = cand.drop(
        columns=["candidate_skill_vs_best_naive_pct", "candidate_seasonal_naive_rmse"], errors="ignore"
    ).merge(key, on=["candidate_run_id", "candidate_split_id", "target", "horizon_h"], how="left")
    return merged.rename(
        columns={
            "skill_vs_best_naive_pct": "candidate_skill_vs_best_naive_pct",
            "seasonal_naive_rmse": "candidate_seasonal_naive_rmse",
        }
    )


def _status(row: pd.Series, cfg: PromotionConfig) -> tuple[str, str]:
    if pd.isna(row.get("xgb_rmse")):
        return "insufficient_data", "no XGBoost baseline for target/horizon"
    if int(row["candidate_n"]) < cfg.min_observations:
        return "insufficient_data", f"candidate has n={int(row['candidate_n'])}, below {cfg.min_observations}"

    # Honest-baseline gate: a candidate must beat the BETTER of persistence and
    # seasonal-naive (same-hour-yesterday). Persistence alone is a free lunch at
    # diurnal horizons (>=24h), so when seasonal-naive evidence is missing there
    # we fail closed rather than promote on persistence.
    horizon_h = int(row["horizon_h"])
    best_naive = row.get("candidate_skill_vs_best_naive_pct")
    has_best_naive = best_naive is not None and not pd.isna(best_naive)
    if has_best_naive:
        if float(best_naive) <= 0:
            return "reject", "candidate does not beat best-naive baseline (persistence/seasonal-naive)"
    elif horizon_h >= 24:
        return "insufficient_data", "seasonal-naive baseline unavailable at diurnal horizon (>=24h)"
    elif float(row["candidate_skill_vs_persistence_pct"]) <= 0:
        return "reject", "candidate does not beat persistence"

    if float(row["xgb_delta_skill_pct"]) < 0:
        return "reject", "candidate does not match or beat XGBoost skill"

    candidate_split_id = str(row.get("candidate_split_id") or "")
    xgb_split_id = str(row.get("xgb_split_id") or "")
    shared_split = bool(candidate_split_id and xgb_split_id and candidate_split_id == xgb_split_id)
    if shared_split:
        if float(row["xgb_skill_vs_persistence_pct"]) <= 0:
            return "promote", "candidate beats persistence and weak XGBoost on shared split"
        return "promote", "candidate beats persistence and XGBoost on shared split"
    if cfg.allow_split_mismatch:
        return "promote", "candidate beats persistence and XGBoost"
    if float(row["xgb_skill_vs_persistence_pct"]) <= 0:
        return "candidate_split_mismatch", "candidate beats persistence; XGBoost baseline is weak; shared split required"
    return "candidate_split_mismatch", "candidate beats persistence and XGBoost; shared split required"


def build_promotion_matrix(cfg: PromotionConfig) -> pd.DataFrame:
    xgb = _read_xgb_baseline(cfg.project_root)
    cand = _read_candidate_metrics(cfg.project_root)
    if cand.empty:
        return pd.DataFrame(
            columns=[
                "schema_version",
                "generated_at_utc",
                "target",
                "horizon_h",
                "candidate_model",
                "candidate_loss",
                "candidate_drivers_enabled",
                "candidate_run_id",
                "candidate_split_id",
                "xgb_run_id",
                "xgb_split_id",
                "candidate_n",
                "candidate_rmse",
                "persistence_rmse",
                "candidate_seasonal_naive_rmse",
                "xgb_rmse",
                "candidate_skill_vs_persistence_pct",
                "candidate_skill_vs_best_naive_pct",
                "xgb_skill_vs_persistence_pct",
                "xgb_delta_skill_pct",
                "status",
                "reason",
            ]
        )
    cand = _ensure_best_naive(cand, cfg.project_root)
    merged = cand.merge(
        xgb,
        left_on=["target", "horizon_h", "candidate_split_id"],
        right_on=["target", "horizon_h", "xgb_split_id"],
        how="left",
    )
    if merged["xgb_rmse"].isna().any():
        fallback = xgb.sort_values(["target", "horizon_h"]).drop_duplicates(["target", "horizon_h"], keep="last")
        missing = merged["xgb_rmse"].isna()
        fallback_join = merged.loc[missing, ["target", "horizon_h"]].merge(
            fallback, on=["target", "horizon_h"], how="left", suffixes=("", "_fallback")
        )
        for col in ["xgb_run_id", "xgb_split_id", "xgb_rmse", "xgb_skill_vs_persistence_pct", "xgb_beats_persistence"]:
            merged.loc[missing, col] = fallback_join[col].to_numpy()
    merged["shared_split"] = merged["candidate_split_id"].astype(str) == merged["xgb_split_id"].astype(str)
    merged["xgb_delta_skill_pct"] = (
        merged["candidate_skill_vs_persistence_pct"] - merged["xgb_skill_vs_persistence_pct"]
    )
    decisions = merged.apply(lambda row: _status(row, cfg), axis=1, result_type="expand")
    merged["status"] = decisions[0]
    merged["reason"] = decisions[1]
    merged.insert(0, "schema_version", PROMOTION_SCHEMA_VERSION)
    merged.insert(1, "generated_at_utc", _utc_now())
    cols = [
        "schema_version",
        "generated_at_utc",
        "target",
        "horizon_h",
        "candidate_model",
        "candidate_loss",
        "candidate_drivers_enabled",
        "candidate_run_id",
        "candidate_split_id",
        "xgb_run_id",
        "xgb_split_id",
        "candidate_n",
        "candidate_rmse",
        "persistence_rmse",
        "candidate_seasonal_naive_rmse",
        "xgb_rmse",
        "candidate_skill_vs_persistence_pct",
        "candidate_skill_vs_best_naive_pct",
        "xgb_skill_vs_persistence_pct",
        "xgb_delta_skill_pct",
        "shared_split",
        "status",
        "reason",
    ]
    for optional in ("candidate_seasonal_naive_rmse", "candidate_skill_vs_best_naive_pct"):
        if optional not in merged.columns:
            merged[optional] = pd.NA
    return merged[cols].sort_values(
        ["status", "target", "horizon_h", "xgb_delta_skill_pct"],
        ascending=[True, True, True, False],
    )


def summarize_matrix(matrix: pd.DataFrame) -> dict[str, Any]:
    if matrix.empty:
        return {
            "rows": 0,
            "status_counts": {},
            "promoted_row_count": 0,
            "unique_promoted_target_horizon_count": 0,
            "unique_promoted_model_cell_count": 0,
            "best_candidates": [],
        }
    status_counts = matrix["status"].value_counts().sort_index().to_dict()
    uniqueness = promotion_uniqueness_counts(matrix)
    eligible = matrix[matrix["status"].isin(["promote", "candidate", "candidate_split_mismatch"])].copy()
    if not eligible.empty:
        eligible = eligible.sort_values("xgb_delta_skill_pct", ascending=False)
    best_candidates = eligible.head(20).to_dict(orient="records")
    return {
        "rows": int(len(matrix)),
        "status_counts": {str(k): int(v) for k, v in status_counts.items()},
        **uniqueness,
        "best_candidates": best_candidates,
    }


def promotion_uniqueness_counts(matrix: pd.DataFrame) -> dict[str, int]:
    """Count promoted evidence without changing promotion policy.

    A raw promoted row is one promoted metric record. A target/horizon cell
    collapses duplicate records for the same forecast objective. A model cell
    keeps model/loss/driver identity while still collapsing duplicate run rows.
    """
    required = {
        "status",
        "target",
        "horizon_h",
        "candidate_model",
        "candidate_loss",
        "candidate_drivers_enabled",
    }
    if matrix.empty or not required.issubset(matrix.columns):
        return {
            "promoted_row_count": 0,
            "unique_promoted_target_horizon_count": 0,
            "unique_promoted_model_cell_count": 0,
        }

    promoted = matrix[matrix["status"] == "promote"].copy()
    if promoted.empty:
        return {
            "promoted_row_count": 0,
            "unique_promoted_target_horizon_count": 0,
            "unique_promoted_model_cell_count": 0,
        }

    target_horizon_cols = ["target", "horizon_h"]
    model_cell_cols = [
        "target",
        "horizon_h",
        "candidate_model",
        "candidate_loss",
        "candidate_drivers_enabled",
    ]
    return {
        "promoted_row_count": int(len(promoted)),
        "unique_promoted_target_horizon_count": int(promoted.drop_duplicates(target_horizon_cols).shape[0]),
        "unique_promoted_model_cell_count": int(promoted.drop_duplicates(model_cell_cols).shape[0]),
    }


def write_outputs(matrix: pd.DataFrame, project_root: Path, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "promotion_matrix.parquet"
    json_path = output_dir / "promotion_summary.json"
    reports_dir = project_root / "release_gate" / "reports"
    reports_parquet_path = reports_dir / "promotion_matrix.parquet"
    reports_json_path = reports_dir / "promotion_summary.json"
    md_path = reports_dir / "promotion_matrix.md"
    matrix.to_parquet(parquet_path, index=False)
    summary = summarize_matrix(matrix)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    reports_dir.mkdir(parents=True, exist_ok=True)
    matrix.to_parquet(reports_parquet_path, index=False)
    reports_json_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    md_path.write_text(make_markdown(matrix, summary), encoding="utf-8")
    return {
        "parquet": str(parquet_path),
        "json": str(json_path),
        "reports_parquet": str(reports_parquet_path),
        "reports_json": str(reports_json_path),
        "markdown": str(md_path),
    }


def make_markdown(matrix: pd.DataFrame, summary: dict[str, Any]) -> str:
    lines = [
        "# MBARI Promotion Matrix",
        "",
        f"- Generated: `{_utc_now()}`",
        f"- Rows: `{summary['rows']}`",
        f"- Raw promoted rows: `{summary.get('promoted_row_count', 0)}`",
        f"- Unique promoted target/horizon cells: `{summary.get('unique_promoted_target_horizon_count', 0)}`",
        f"- Unique promoted model cells: `{summary.get('unique_promoted_model_cell_count', 0)}`",
        "",
        "## Executive Summary: Shared-Split Promotions",
        "",
        "These rows beat persistence and the XGBoost baseline on a shared split. ",
        "Raw promoted rows can include duplicate runs for the same forecast objective; use the unique target/horizon and model-cell counts for breadth.",
        "",
    ]
    
    # Filter for the real wins: beating persistence and beating XGBoost on a shared split.
    wins = matrix[matrix["status"] == "promote"].copy()
    if not wins.empty:
        # Prioritize by the delta against the production baseline.
        wins = wins.sort_values("xgb_delta_skill_pct", ascending=False).head(15)
        show_cols = [
            "target",
            "horizon_h",
            "candidate_model",
            "candidate_skill_vs_persistence_pct",
            "candidate_skill_vs_best_naive_pct",
            "xgb_skill_vs_persistence_pct",
            "xgb_delta_skill_pct",
        ]
        show_cols = [c for c in show_cols if c in wins.columns]
        lines.append("### Top Shared-Split Promotions")
        lines.append("")
        lines.extend(_markdown_table(wins[show_cols]))
        lines.append("")
    else:
        lines.append("_No shared-split promotions currently available._")
        lines.append("")

    lines.extend([
        "## Operational Status Counts",
        "",
    ])
    if summary["status_counts"]:
        for status, count in summary["status_counts"].items():
            lines.append(f"- `{status}`: `{count}`")
    else:
        lines.append("- No candidate rows found.")
    
    lines.extend(["", "## Quality Review: Split-Mismatch Candidates", ""])
    lines.append("These rows beat persistence and XGBoost but require a shared-split rerun before promotion.")
    lines.append("")
    
    best = pd.DataFrame(summary.get("best_candidates", []))
    # Exclude already-promoted wins from this list to reduce noise.
    best = best[best["status"] == "candidate_split_mismatch"].head(10)
    
    if best.empty:
        lines.append("No split-mismatch candidates found.")
    else:
        show_cols = [
            "target",
            "horizon_h",
            "candidate_model",
            "candidate_skill_vs_persistence_pct",
            "xgb_skill_vs_persistence_pct",
            "status",
            "reason",
        ]
        lines.extend(_markdown_table(best[show_cols]))
    lines.extend(
        [
            "",
            "## Policy",
            "",
            "- `promote`: candidate beats the best-naive baseline (better of persistence and seasonal-naive) and XGBoost under allowed split policy.",
            "- `candidate_split_mismatch`: candidate beats best-naive and XGBoost, but shared split parity is not yet proven.",
            "- XGBoost leaderboards with `split_id` are promoted only when that `split_id` equals the candidate `split_id`.",
            "- `reject`: candidate fails the best-naive (persistence/seasonal-naive) or XGBoost comparison.",
            "- `insufficient_data`: missing baseline, too few scored observations, or no seasonal-naive evidence at a diurnal horizon (>=24h).",
            "- Best-naive note: persistence is a free lunch at diurnal horizons, so promotion requires beating same-hour-yesterday there.",
            "",
        ]
    )
    return "\n".join(lines)


def _markdown_table(df: pd.DataFrame) -> list[str]:
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        values = [str(row[col]) for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MBARI model promotion matrix.")
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PROJECT_ROOT / "lakehouse" / "gold" / "promotion_matrix",
    )
    parser.add_argument("--min-observations", type=int, default=20)
    parser.add_argument(
        "--allow-split-mismatch",
        action="store_true",
        help="allow promote status without a shared XGBoost/neural split; normally keep off",
    )
    parser.add_argument("--dry-run", action="store_true", help="print summary without writing outputs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = PromotionConfig(
        project_root=args.project_root.resolve(),
        min_observations=args.min_observations,
        allow_split_mismatch=args.allow_split_mismatch,
    )
    matrix = build_promotion_matrix(cfg)
    summary = summarize_matrix(matrix)
    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True, default=str))
        return 0
    paths = write_outputs(matrix, cfg.project_root, args.output_dir.resolve())
    print(json.dumps({"summary": summary, "paths": paths}, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
