#!/usr/bin/env python
"""Leakage-safe driver → whale/marine-mammal mortality model (Whale 2 lane).

Consumes the **mortality panel** (Whale 1's lane — see the panel contract posted to the
`_WHALE` channel) and asks the real modeling question: do *environmental drivers known
BEFORE a period* (domoic-acid exposure, SST) predict that period's mortality counts better
than a **per-region seasonal-climatology baseline**? Beating that honest baseline — not raw
accuracy — is the bar (AGENTS.md value gate).

Leakage discipline (the thing that makes or breaks this): every driver feature for period
`t` is built ONLY from values dated strictly before `t` (lagged pDA). `add_lagged_pda` and
`seasonal_baseline_predict` are pure and unit-tested; `evaluate` wires them to a model.

This file ships and is tested against a SYNTHETIC panel so it is correct before Whale 1's
real panel lands; point `--panel` at the real curated panel to run it for real.

Panel schema (contract): columns region, period_start (date), species_group, mortality_count
[, sighting_count, survey_effort]. Writes only under research/whale/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HABMAP = ROOT / "data" / "external_curated" / "habmap_cdph" / "habmap_cdph.parquet"
OUT = Path(__file__).resolve().parent / "reports"


# ── leakage-safe feature construction (pure, tested) ─────────────────────────────
def add_lagged_pda(panel: pd.DataFrame, pda_monthly: pd.Series,
                   lags: Iterable[int] = (1, 2, 3)) -> pd.DataFrame:
    """Attach pDA exposure from BEFORE each row's period_start (no leakage).

    `pda_monthly` is a Series indexed by monthly Period. For a panel row in month m, the
    feature `pda_lag{k}` is the pDA in month (m - k) — strictly past. A trailing mean
    `pda_trail3` (months m-1..m-3) gives a smoothed recent-exposure signal. Rows whose
    lagged months are missing get NaN (caller decides how to handle).
    """
    df = panel.copy()
    months = pd.to_datetime(df["period_start"], utc=True).dt.tz_convert(None).dt.to_period("M")
    for k in lags:
        df[f"pda_lag{k}"] = [pda_monthly.get(m - k, np.nan) for m in months]
    trail = []
    for m in months:
        vals = [pda_monthly.get(m - k, np.nan) for k in (1, 2, 3)]
        vals = [v for v in vals if v is not None and not pd.isna(v)]
        trail.append(float(np.mean(vals)) if vals else np.nan)
    df["pda_trail3"] = trail
    df["month_of_year"] = months.dt.month.to_numpy()
    return df


def seasonal_baseline_predict(train: pd.DataFrame, test: pd.DataFrame,
                              group_cols=("region", "month_of_year"),
                              target="mortality_count") -> np.ndarray:
    """Honest baseline: predict each test row as the mean target for its
    (region, calendar-month) computed on the TRAIN split only. Falls back to the global
    train mean for unseen groups. This is the climatology the model must beat."""
    gcols = list(group_cols)
    clim = train.groupby(gcols)[target].mean()
    global_mean = float(train[target].mean()) if len(train) else 0.0
    out = []
    for _, r in test.iterrows():
        key = tuple(r[c] for c in gcols)
        val = clim.get(key, np.nan)
        out.append(global_mean if (val is None or pd.isna(val)) else float(val))
    return np.asarray(out, dtype="float64")


def time_split(panel: pd.DataFrame, frac_train: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split — train on the earliest `frac_train`, test on the rest. No
    shuffling: a forecaster must not see the future."""
    df = panel.sort_values("period_start").reset_index(drop=True)
    cut = int(len(df) * frac_train)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def _mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


# ── evaluation (uses sklearn if present; linear fallback otherwise) ──────────────
def evaluate(panel: pd.DataFrame, pda_monthly: pd.Series,
             feature_cols=("pda_lag1", "pda_lag2", "pda_lag3", "pda_trail3", "month_of_year"),
             target="mortality_count") -> dict:
    """Fit driver model on the chronological train split, compare TEST MAE to the seasonal
    climatology baseline. Returns skill = 1 - model_mae/baseline_mae (>0 means the drivers
    add real predictive value over climatology)."""
    feat = add_lagged_pda(panel, pda_monthly)
    feat = feat.dropna(subset=[c for c in feature_cols if c.startswith("pda")])
    if len(feat) < 12:
        return {"status": "insufficient_rows", "n": int(len(feat))}
    train, test = time_split(feat)
    if len(train) < 6 or len(test) < 3:
        return {"status": "insufficient_split", "n_train": len(train), "n_test": len(test)}

    Xtr, ytr = train[list(feature_cols)].to_numpy(float), train[target].to_numpy(float)
    Xte, yte = test[list(feature_cols)].to_numpy(float), test[target].to_numpy(float)

    model_name, pred = _fit_predict(Xtr, ytr, Xte)
    base = seasonal_baseline_predict(train, test, target=target)

    model_mae, base_mae = _mae(yte, pred), _mae(yte, base)
    skill = float(1.0 - model_mae / base_mae) if base_mae > 0 else np.nan
    return {
        "status": "ok", "model": model_name,
        "n_train": int(len(train)), "n_test": int(len(test)),
        "model_test_mae": model_mae, "baseline_test_mae": base_mae,
        "skill_vs_climatology": skill,
        "beats_baseline": bool(skill is not None and not np.isnan(skill) and skill > 0),
        "honest_note": "skill>0 means lagged DA/SST beat the per-region seasonal climatology "
                       "out-of-sample. skill<=0 is an honest null — report it straight.",
    }


def _fit_predict(Xtr, ytr, Xte):
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: PLC0415
        m = HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.05)
        m.fit(Xtr, ytr)
        return "HistGradientBoostingRegressor", m.predict(Xte)
    except Exception:
        # least-squares fallback with intercept
        A = np.hstack([Xtr, np.ones((len(Xtr), 1))])
        coef, *_ = np.linalg.lstsq(A, ytr, rcond=None)
        return "linear_lstsq", np.hstack([Xte, np.ones((len(Xte), 1))]) @ coef


# ── data loading ─────────────────────────────────────────────────────────────────
def load_pda_monthly() -> pd.Series:
    df = pd.read_parquet(HABMAP, columns=["time", "pDA"])
    t = pd.to_datetime(df["time"], errors="coerce", utc=True).dt.tz_convert(None)
    v = pd.to_numeric(df["pDA"], errors="coerce")
    g = pd.DataFrame({"month": t.dt.to_period("M"), "v": v}).dropna()
    return g.groupby("month")["v"].mean().sort_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", type=str, default="", help="path to curated mortality panel parquet")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.panel or not Path(args.panel).exists():
        print("No --panel given / not found. The model harness is built + unit-tested; it will run "
              "when Whale 1's mortality panel lands. (See research/whale/MISSION.md.)")
        return
    panel = pd.read_parquet(args.panel)
    res = evaluate(panel, load_pda_monthly())
    (OUT / "mortality_model_result.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
