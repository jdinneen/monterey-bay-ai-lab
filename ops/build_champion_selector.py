#!/usr/bin/env python3
"""Build a target/horizon champion selector from promotion-gated evidence."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


EXTRA_COLUMNS = [
    "fallback_model",
    "fallback_reason",
    "fallback_run_id",
    "fallback_split_id",
    "serving_status",
    "artifact_path",
    "requires_driver_table",
    "driver_policy_status",
    "duplicate_promoted_candidate_count",
    "promoted_candidate_count",
    "competing_promoted_models",
    "reason",
]


def _bool_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _fallback_model(row: pd.Series) -> str:
    xgb_run_id = str(row.get("xgb_run_id", "") or "").strip()
    xgb_split_id = str(row.get("xgb_split_id", "") or "").strip()
    xgb_rmse = row.get("xgb_rmse")
    if xgb_run_id and xgb_split_id and pd.notna(xgb_rmse):
        return "xgboost"
    return "persistence"


def _fallback_reason(row: pd.Series) -> str:
    if _fallback_model(row) == "xgboost":
        return "shared-split XGBoost baseline used for promotion comparison"
    return "persistence baseline used when no shared-split XGBoost fallback is available"


def _serving_status(row: pd.Series, project_root: Path | None = None) -> str:
    if _bool_value(row.get("candidate_drivers_enabled", False)):
        return "promotion_gated_driver_policy_required"
    artifact = str(row.get("artifact_path", "") or "")
    artifact_path = Path(artifact)
    if artifact and not artifact_path.is_absolute() and project_root is not None:
        artifact_path = project_root / artifact_path
    if artifact and artifact_path.exists():
        return "promotion_gated_candidate"
    return "needs_artifact"


def _driver_policy_status(row: pd.Series) -> str:
    if _bool_value(row.get("candidate_drivers_enabled", False)):
        return "requires_driver_policy_approval"
    return "no_driver_table_required"


def _artifact_path(run_id: Any) -> str:
    run = str(run_id or "").strip()
    if not run:
        return ""
    return f"lakehouse/gold/forecast_runs/run_id={run}"


def _empty_champions(required: set[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=sorted(required) + EXTRA_COLUMNS)


def build_champions(matrix: pd.DataFrame, project_root: Path | None = None) -> pd.DataFrame:
    if matrix.empty:
        return pd.DataFrame()
    required = {
        "status",
        "target",
        "horizon_h",
        "candidate_model",
        "candidate_loss",
        "candidate_drivers_enabled",
        "candidate_run_id",
        "candidate_split_id",
        "candidate_rmse",
        "candidate_skill_vs_persistence_pct",
        "xgb_delta_skill_pct",
    }
    missing = required - set(matrix.columns)
    if missing:
        raise ValueError(f"promotion matrix missing columns: {sorted(missing)}")
    promoted = matrix[matrix["status"] == "promote"].copy()
    if promoted.empty:
        return _empty_champions(required)
    promoted["horizon_h"] = pd.to_numeric(promoted["horizon_h"], errors="raise").astype(int)
    promoted["candidate_rmse"] = pd.to_numeric(promoted["candidate_rmse"], errors="raise")
    promoted["candidate_skill_vs_persistence_pct"] = pd.to_numeric(
        promoted["candidate_skill_vs_persistence_pct"], errors="raise"
    )
    promoted["xgb_delta_skill_pct"] = pd.to_numeric(promoted["xgb_delta_skill_pct"], errors="raise")
    # Best-naive skill (vs the better of persistence and seasonal-naive) is the
    # honest tiebreak; surface it and prefer it when the matrix provides it.
    if "candidate_skill_vs_best_naive_pct" in promoted.columns:
        promoted["candidate_skill_vs_best_naive_pct"] = pd.to_numeric(
            promoted["candidate_skill_vs_best_naive_pct"], errors="coerce"
        )
        sort_cols = ["target", "horizon_h", "candidate_skill_vs_best_naive_pct", "xgb_delta_skill_pct", "candidate_rmse"]
        ascending = [True, True, False, False, True]
    else:
        sort_cols = ["target", "horizon_h", "xgb_delta_skill_pct", "candidate_skill_vs_persistence_pct", "candidate_rmse"]
        ascending = [True, True, False, False, True]
    winners = promoted.sort_values(sort_cols, ascending=ascending).drop_duplicates(
        ["target", "horizon_h"], keep="first"
    )
    duplicate_counts = (
        promoted.groupby(["target", "horizon_h"], dropna=False)
        .size()
        .rename("promoted_candidate_count")
        .reset_index()
    )
    competing = (
        promoted.assign(
            _candidate_label=promoted.apply(
                lambda row: (
                    f"{row.get('candidate_model')}:{row.get('candidate_loss')}:"
                    f"drivers={_bool_value(row.get('candidate_drivers_enabled', False))}:"
                    f"run={row.get('candidate_run_id')}"
                ),
                axis=1,
            )
        )
        .groupby(["target", "horizon_h"], dropna=False)["_candidate_label"]
        .apply(lambda values: "; ".join(str(v) for v in values))
        .rename("competing_promoted_models")
        .reset_index()
    )
    winners = winners.merge(duplicate_counts, on=["target", "horizon_h"], how="left")
    winners = winners.merge(competing, on=["target", "horizon_h"], how="left")
    winners["fallback_model"] = winners.apply(_fallback_model, axis=1)
    winners["fallback_reason"] = winners.apply(_fallback_reason, axis=1)
    winners["fallback_run_id"] = winners.apply(
        lambda row: str(row.get("xgb_run_id", "") or "") if row["fallback_model"] == "xgboost" else "", axis=1
    )
    winners["fallback_split_id"] = winners.apply(
        lambda row: str(row.get("xgb_split_id", "") or row.get("candidate_split_id", "") or ""), axis=1
    )
    winners["artifact_path"] = winners["candidate_run_id"].map(_artifact_path)
    winners["requires_driver_table"] = winners["candidate_drivers_enabled"].map(_bool_value)
    winners["driver_policy_status"] = winners.apply(_driver_policy_status, axis=1)
    winners["serving_status"] = winners.apply(lambda row: _serving_status(row, project_root), axis=1)
    winners["duplicate_promoted_candidate_count"] = winners["promoted_candidate_count"]
    if "reason" not in winners.columns:
        winners["reason"] = ""
    return winners.reset_index(drop=True)


def summarize_champions(champions: pd.DataFrame) -> dict[str, Any]:
    if champions.empty:
        return {
            "champion_cells": 0,
            "targets": 0,
            "horizons": [],
            "models": {},
            "drivers_enabled_cells": 0,
            "driver_policy_required_cells": 0,
            "needs_artifact_cells": 0,
            "ready_cells": 0,
            "fallback_models": {},
        }
    return {
        "champion_cells": int(len(champions)),
        "targets": int(champions["target"].nunique()),
        "horizons": sorted(int(v) for v in champions["horizon_h"].dropna().unique()),
        "models": {str(k): int(v) for k, v in champions["candidate_model"].value_counts().sort_index().items()},
        "drivers_enabled_cells": int(champions["candidate_drivers_enabled"].astype(bool).sum()),
        "driver_policy_required_cells": int(
            (
                champions.get("driver_policy_status", pd.Series(dtype=str))
                == "requires_driver_policy_approval"
            ).sum()
        ),
        "needs_artifact_cells": int((champions.get("serving_status", pd.Series(dtype=str)) == "needs_artifact").sum()),
        "ready_cells": int(
            (champions.get("serving_status", pd.Series(dtype=str)) == "promotion_gated_candidate").sum()
        ),
        "fallback_models": {
            str(k): int(v) for k, v in champions.get("fallback_model", pd.Series(dtype=str)).value_counts().sort_index().items()
        },
    }


def _markdown_table(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return ["_No rows._"]
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return lines


def write_outputs(champions: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_champions(champions)
    parquet_path = output_dir / "champion_selector.parquet"
    json_path = output_dir / "champion_selector_summary.json"
    md_path = output_dir / "CHAMPION_SELECTOR.md"
    champions.to_parquet(parquet_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# MBARI Champion Selector",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`",
        "",
        "This selector includes only promotion-gated rows. It is target/horizon specific; it is not a global model ranking.",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, indent=2, sort_keys=True),
        "```",
        "",
        "## Champions",
        "",
    ]
    show_cols = [
        "target",
        "horizon_h",
        "candidate_model",
        "candidate_loss",
        "candidate_drivers_enabled",
        "candidate_skill_vs_persistence_pct",
        "candidate_skill_vs_best_naive_pct",
        "xgb_delta_skill_pct",
        "candidate_run_id",
        "candidate_split_id",
        "fallback_model",
        "fallback_reason",
        "fallback_run_id",
        "fallback_split_id",
        "serving_status",
        "artifact_path",
        "requires_driver_table",
        "driver_policy_status",
        "duplicate_promoted_candidate_count",
        "promoted_candidate_count",
        "competing_promoted_models",
        "reason",
    ]
    lines.extend(_markdown_table(champions[[c for c in show_cols if c in champions.columns]]))
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"parquet": str(parquet_path), "json": str(json_path), "markdown": str(md_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--matrix", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.project_root.resolve()
    matrix_path = args.matrix or root / "lakehouse" / "gold" / "promotion_matrix" / "promotion_matrix.parquet"
    output_dir = args.output_dir or root / "reports" / "champion_selector"
    matrix = pd.read_parquet(matrix_path)
    champions = build_champions(matrix, project_root=root)
    paths = write_outputs(champions, output_dir)
    print(json.dumps({"summary": summarize_champions(champions), "paths": paths}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
