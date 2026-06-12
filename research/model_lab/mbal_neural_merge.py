#!/usr/bin/env python3
"""
Merge all per-model neural leaderboards (nn_results/<model>/leaderboard.csv) into one unified
comparison vs persistence and the XGBoost baseline. Reads each model's own outdir -- safe to
run anytime, including while some models are still training (it just reports what's finished).

Correctness invariants (see audit 2026-06-08, AGENTS.md):
  - DROP smoke runs (summary.json smoke=true): they have n=3-8 points and are not comparable.
  - NEVER average full-history and bounded-history (tail_weeks) runs together — they are
    different evaluation regimes. Report each history class separately.
  - Headline aggregate is the MEDIAN across series, because skill = 1 - rmse/persistence has
    tiny denominators that produce huge negative outliers (down to ~-2000%) which make the
    arithmetic mean meaningless. Mean is emitted only as a clearly-labeled reference column.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import pandas as pd

# This module lives at <root>/research/model_lab/; fall back to the 2-levels-up project
# root (not the script dir) when MBAL_PROJECT_ROOT is unset, so nn_results resolves.
PROJECT_ROOT = Path(os.environ.get("MBAL_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
ROOT = PROJECT_ROOT / "nn_results"
XGB = PROJECT_ROOT / "mbal_forecast_v2_results_nodrivers" / "leaderboard.csv"
OUT = PROJECT_ROOT / "nn_results"
HORIZONS = [1, 6, 24, 72, 168]


def _run_meta(run_dir: Path) -> tuple[bool, object]:
    """(smoke, tail_weeks) from a run's summary.json; defaults (False, None)."""
    try:
        s = json.loads((run_dir / "summary.json").read_text())
        return bool(s.get("smoke", False)), s.get("tail_weeks")
    except Exception:
        return False, None


def _history_class(tail) -> str:
    return f"bounded{int(tail)}" if tail else "full"


def main() -> None:
    frames = []
    for lb in ROOT.glob("*/leaderboard.csv"):
        try:
            df = pd.read_csv(lb)
        except Exception as e:
            print(f"skip {lb}: {e}"); continue
        if not len(df) or "model" not in df.columns:
            continue
        smoke, tail = _run_meta(lb.parent)
        df["run_dir"] = lb.parent.name
        df["smoke"] = smoke
        df["tail_weeks"] = tail
        df["history_class"] = _history_class(tail)
        frames.append(df)
    if not frames:
        print("no finished model leaderboards yet."); return
    allm = pd.concat(frames, ignore_index=True)

    # Production rows only: smoke runs (n=3-8) are not comparable to full CV runs.
    prod = allm[~allm["smoke"].astype(bool)].copy()
    if prod.empty:
        print("only smoke runs present; nothing to publish."); return

    # XGBoost baseline (skill_rmse_vs_persistence in fraction) for reference
    xgb = None
    if XGB.exists():
        x = pd.read_csv(XGB)
        x = x[x.get("task", "forecast") == "forecast"] if "task" in x.columns else x
        xgb = x[["target", "horizon_h", "skill_rmse_vs_persistence"]].copy()
        xgb["skill_xgb_pct"] = xgb["skill_rmse_vs_persistence"] * 100
        xgb = xgb.rename(columns={"target": "unique_id"})[["unique_id", "horizon_h", "skill_xgb_pct"]]

    # Aggregate by model x history_class x horizon. Median is the robust headline; mean is reference.
    def agg(stat):
        return (prod.groupby(["model", "history_class", "horizon_h"])["skill_vs_persistence_pct"]
                .agg(stat).round(2).reset_index())
    med = agg("median")
    mean_ref = agg("mean").rename(columns={"skill_vs_persistence_pct": "mean_skill_outlier_sensitive"})

    # Primary file keeps its historic shape (model x horizon) but is now FULL-HISTORY,
    # no-smoke, MEDIAN — the honest production comparison consumed downstream.
    full_med = med[med["history_class"] == "full"]
    pivot = full_med.pivot(index="model", columns="horizon_h", values="skill_vs_persistence_pct")
    pivot.to_csv(OUT / "skill_by_model_horizon.csv")
    # Class-aware table (median + mean side by side) for full transparency.
    med.merge(mean_ref, on=["model", "history_class", "horizon_h"]).to_csv(
        OUT / "skill_by_model_history_horizon.csv", index=False)

    # Best model per (series,horizon): full-history production runs only (classes not comparable).
    full_prod = prod[prod["history_class"] == "full"]
    idx = full_prod.groupby(["unique_id", "horizon_h"])["model_rmse"].idxmin()
    best = full_prod.loc[idx, ["unique_id", "horizon_h", "model", "model_rmse", "skill_vs_persistence_pct"]]
    best.to_csv(OUT / "best_model_per_series_horizon.csv", index=False)

    # timing
    times = {}
    for sj in ROOT.glob("*/summary.json"):
        try:
            s = json.loads(sj.read_text()); times[s["model"]] = s.get("minutes")
        except Exception:
            pass

    def class_table(hclass):
        mp = med[med.history_class == hclass].pivot(index="model", columns="horizon_h",
                                                    values="skill_vs_persistence_pct")
        rows = []
        for m in mp.index:
            mr = mp.loc[m]
            cells = " | ".join(f"{mr.get(h):.1f}" if pd.notna(mr.get(h)) else "—" for h in HORIZONS)
            rows.append(f"| {m} | {cells} | {times.get(m,'?')} |")
        return rows

    L = ["# Monterey Bay AI Lab Neural Forecasting — Unified Leaderboard", "",
         f"Production models (smoke runs excluded): {sorted(prod['model'].unique())}", "",
         "Headline = **median** skill vs persistence across series (robust to near-zero-persistence",
         "outliers that make the mean unreliable). Mean is in skill_by_model_history_horizon.csv,",
         "labeled outlier-sensitive. Full-history and bounded-history (tail_weeks) runs are reported",
         "separately and never averaged together.", ""]

    full_rows = class_table("full")
    if full_rows:
        L += ["## Full-history runs — median skill vs persistence (%) by model × horizon", "",
              "| model | " + " | ".join(f"+{h}h" for h in HORIZONS) + " | train min |",
              "|" + "---|" * (len(HORIZONS) + 2)] + full_rows
        if xgb is not None:
            xs = xgb.groupby("horizon_h")["skill_xgb_pct"].median()
            cells = " | ".join(f"{xs.get(h, float('nan')):.1f}" if pd.notna(xs.get(h)) else "—" for h in HORIZONS)
            L.append(f"| xgboost(ref, median) | {cells} | — |")
        L.append("")

    for hclass in sorted(c for c in med["history_class"].unique() if c != "full"):
        rows = class_table(hclass)
        if not rows:
            continue
        L += [f"## {hclass} runs — NOT comparable to full-history — median skill (%)", "",
              "| model | " + " | ".join(f"+{h}h" for h in HORIZONS) + " | train min |",
              "|" + "---|" * (len(HORIZONS) + 2)] + rows + [""]

    L += ["## Best model per series × horizon (full-history, count of wins)", ""]
    for m, c in best["model"].value_counts().items():
        L.append(f"- {m}: {c}")
    L += ["", "Outputs: skill_by_model_horizon.csv (full-history median), "
          "skill_by_model_history_horizon.csv (all classes; median + mean), "
          "best_model_per_series_horizon.csv"]
    (OUT / "NEURAL_LEADERBOARD.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()

