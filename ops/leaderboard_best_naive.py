#!/usr/bin/env python3
"""Attach best-naive skill to a research run's leaderboard.

WHY THIS EXISTS
---------------
``mbal_neural_forecast.py`` writes ``leaderboard.csv`` with
``skill_vs_persistence_pct`` only. Persistence gets a "free lunch" at diurnal/
multi-day horizons (24/72/168h): the value 3-7 days ago is a terrible forecast,
so "beats persistence" there is meaningless. The honest bar is **best-naive** =
min(persistence, same-hour-yesterday). A driver A/B on 2026-06-09 produced
long-horizon cells that beat persistence yet lost to seasonal-naive by 30-600%;
the persistence-only leaderboard hid that. This tool recomputes best-naive skill
from a run's ``cv_predictions.parquet`` using the SAME scorer the gold metrics
use (``ops/seasonal_naive.score_predictions``) and writes an augmented
leaderboard, so no research run can be read as a win on the wrong baseline.

It is additive: it reads a finished run and writes a new CSV next to it. It does
not modify ``mbal_neural_forecast.py`` or any shared pipeline file.

Binding invariant: AGENTS.md "Persistence is NOT a sufficient baseline at diurnal
horizons ... compare against the better of persistence and seasonal-naive."
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ops.seasonal_naive import load_observed_panel, score_predictions  # noqa: E402

# Best-naive columns we graft onto the leaderboard, keyed by (unique_id, horizon_h).
_BN_COLS = [
    "seasonal_naive_rmse",
    "best_naive_rmse",
    "skill_vs_seasonal_naive_pct",
    "skill_vs_best_naive_pct",
    "n_best_naive",
]


def augment_leaderboard(
    leaderboard: pd.DataFrame,
    preds: pd.DataFrame,
    obs_val: pd.Series,
) -> pd.DataFrame:
    """Return ``leaderboard`` with best-naive columns merged in.

    Pure function (no IO) so it is unit-testable. Cells with no scorable
    best-naive subset are left NaN rather than dropped — the leaderboard keeps
    all its rows; it just gains the honest-baseline columns.
    """
    bn = score_predictions(preds, obs_val)
    keep = ["unique_id", "horizon_h"] + [c for c in _BN_COLS if c in bn.columns]
    bn = bn[keep].drop_duplicates(["unique_id", "horizon_h"])
    out = leaderboard.merge(bn, on=["unique_id", "horizon_h"], how="left")
    return out


def process_run(run_dir: Path, obs_val: pd.Series) -> pd.DataFrame:
    """Read a run dir, write ``leaderboard_best_naive.csv``, return the table."""
    lb_path = run_dir / "leaderboard.csv"
    cv_path = run_dir / "cv_predictions.parquet"
    if not lb_path.exists() or not cv_path.exists():
        raise FileNotFoundError(f"run dir missing leaderboard.csv / cv_predictions.parquet: {run_dir}")
    leaderboard = pd.read_csv(lb_path)
    preds = pd.read_parquet(cv_path)
    out = augment_leaderboard(leaderboard, preds, obs_val)
    out_path = run_dir / "leaderboard_best_naive.csv"
    out.to_csv(out_path, index=False)
    return out


def _summary(name: str, df: pd.DataFrame) -> str:
    scored = df[df["skill_vs_best_naive_pct"].notna()] if "skill_vs_best_naive_pct" in df else df.iloc[:0]
    beats = int((scored["skill_vs_best_naive_pct"] > 0).sum()) if len(scored) else 0
    return f"{name}: {beats}/{len(scored)} cells beat best-naive (of {len(df)} leaderboard rows)"


def main() -> int:
    ap = argparse.ArgumentParser(description="Attach best-naive skill to run leaderboard(s).")
    ap.add_argument("run_dirs", nargs="+", help="run directories (each with leaderboard.csv + cv_predictions.parquet)")
    a = ap.parse_args()

    obs = load_observed_panel(ROOT)
    if obs is None:
        print("ERROR: observed panel unavailable (nn_cache long/mask missing) — cannot score best-naive.", file=sys.stderr)
        return 2

    rc = 0
    for d in a.run_dirs:
        run_dir = Path(d)
        try:
            df = process_run(run_dir, obs)
            print(_summary(run_dir.name, df))
            print(f"  wrote {run_dir / 'leaderboard_best_naive.csv'}")
        except FileNotFoundError as e:
            print(f"SKIP {d}: {e}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
