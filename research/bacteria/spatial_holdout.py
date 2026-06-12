#!/usr/bin/env python3
"""Leave-one-county-out (LOCO) spatial generalization for the bacteria benchmark.

The temporal split in operational_benchmark.py answers "predict a known beach's future."
A reviewer will demand the harder question: **predict a beach you have never trained
on.** This trains on every OTHER county (through 2021), calibrates on other-county
2020-21, and tests on the held-out county's 2022+ samples — strict spatial + temporal
holdout. The held-out county's own causal features (its prior labs, station memory,
local rainfall) are still available, so this measures whether the LEARNED patterns
transfer, not whether we memorized known sites.

Reuses the canonical leakage-safe pipeline from operational_benchmark.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow `python research/bacteria/spatial_holdout.py` (script run) to import the
# sibling package by putting the repo root on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from research.bacteria import operational_benchmark as ob  # noqa: E402


def _default_clf():
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, l2_regularization=1.0,
        class_weight="balanced", early_stopping=True, validation_fraction=0.1,
        random_state=42,
    )


def _sw_prev7_excluding(df: pd.DataFrame, county: str) -> pd.Series:
    """Recompute the statewide prior-7d exceedance rate with the held-out county's own
    samples removed, so its labels cannot leak into other counties' training rows."""
    day = df.groupby("sample_date")["exceed"].agg(all_sum="sum", all_n="count")
    cdf = df[df["county"] == county].groupby("sample_date")["exceed"].agg(c_sum="sum", c_n="count")
    j = day.join(cdf, how="left").fillna({"c_sum": 0, "c_n": 0})
    denom = (j["all_n"] - j["c_n"]).replace(0, np.nan)
    rate = ((j["all_sum"] - j["c_sum"]) / denom).sort_index()
    sw_prev7 = rate.shift(1).rolling(7, min_periods=1).mean()
    return df["sample_date"].map(sw_prev7)


def evaluate_county(df: pd.DataFrame, feats: list[str], county: str, clf_factory) -> dict | None:
    df = df.copy()
    df["sw_prev7"] = _sw_prev7_excluding(df, county)  # fix the cross-county boundary leak
    train = df[(df["county"] != county) & (df["sample_date"] <= "2019-12-31")]
    cal = df[(df["county"] != county) & (df["sample_date"] >= "2020-01-01")
             & (df["sample_date"] <= "2021-12-31")]
    test = df[(df["county"] == county) & (df["sample_date"] >= "2022-01-01")].copy()
    if test["exceed"].nunique() < 2 or train["exceed"].nunique() < 2:
        return None

    clf = clf_factory()
    # drop features constant in this held-out-county training set (sklearn>=1.9 HGBT binning
    # errors on a single-distinct-value feature); mirrors operational_benchmark's guard.
    ff = [c for c in feats if train[c].notna().sum() >= 2 and train[c].nunique(dropna=True) > 1]
    clf.fit(train[ff].astype(float), train["exceed"].to_numpy())
    raw = clf.predict_proba(test[ff].astype(float))[:, 1]
    p_cal = raw
    if cal["exceed"].nunique() > 1:
        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(clf.predict_proba(cal[ff].astype(float))[:, 1], cal["exceed"].to_numpy())
        p_cal = iso.predict(raw)

    base = float(train["exceed"].mean())
    y = test["exceed"].to_numpy()
    rows = {
        "model_calibrated": ob.score(y, p_cal, base),
        "baseline_prior_lab": ob.score(y, test["exc_prev"].fillna(base).to_numpy(), base),
        "baseline_station_memory": ob.score(y, test["station_prior_rate"].fillna(base).to_numpy(), base),
    }
    if "rain_3d" in test.columns:
        ab = (test["rain_3d"].fillna(0.0) >= ob.AB411_RAIN_MM).astype(float).to_numpy()
        rows["baseline_ab411_rain"] = ob.score(y, ab, base)

    m_ap = rows["model_calibrated"].get("ap")
    mem_ap = rows["baseline_station_memory"].get("ap")
    ab_ap = rows.get("baseline_ab411_rain", {}).get("ap")
    mem_ece = rows["baseline_station_memory"].get("ece")
    m_ece = rows["model_calibrated"].get("ece")
    return {
        "county": county, "n": int(len(test)), "events": int(y.sum()),
        "base_rate": round(float(y.mean()), 4), "models": rows,
        "model_beats_memory": (None if (m_ap is None or mem_ap is None) else bool(m_ap > mem_ap)),
        "model_beats_ab411": (None if (m_ap is None or ab_ap is None) else bool(m_ap > ab_ap)),
        "deploy_ready": bool(m_ece is not None and mem_ece is not None and m_ece <= mem_ece + 0.05),
    }


def run_loco(obs_path: Path, rain_dir=None, clf_factory=None, min_test_events: int = 150,
             max_counties: int | None = None, reveal_lag_days: int = 0, label: str = "any") -> dict:
    clf_factory = clf_factory or _default_clf
    df = ob.load_station_days(obs_path, label=label)
    df = ob.add_causal_features(df, reveal_lag_days=reveal_lag_days)
    feats = list(ob.FEATS)
    has_rain = False
    if rain_dir is not None:
        df = ob.add_rain_features(df, Path(rain_dir))
        if "rain_3d" in df.columns and df["rain_3d"].notna().any():
            feats += ob.RAIN_FEATS
            has_rain = True

    test_era = df[df["sample_date"] >= "2022-01-01"]
    ev = test_era.groupby("county")["exceed"].sum().sort_values(ascending=False)
    counties = [c for c, n in ev.items() if n >= min_test_events]
    if max_counties:
        counties = counties[:max_counties]

    per_county = [r for c in counties if (r := evaluate_county(df, feats, c, clf_factory)) is not None]
    # Aggregate across held-out counties (each county weighted equally — a fair
    # spatial-generalization summary, not dominated by San Diego's volume).
    m_aps = [r["models"]["model_calibrated"]["ap"] for r in per_county if "ap" in r["models"]["model_calibrated"]]
    beats_ab411 = [r["model_beats_ab411"] for r in per_county if r["model_beats_ab411"] is not None]
    beats_mem = [r["model_beats_memory"] for r in per_county if r["model_beats_memory"] is not None]
    agg = {
        "n_counties": len(per_county),
        "median_model_ap": round(float(np.median(m_aps)), 4) if m_aps else None,
        "counties_model_beats_ab411": f"{sum(beats_ab411)}/{len(beats_ab411)}" if beats_ab411 else "n/a",
        "counties_model_beats_memory": f"{sum(beats_mem)}/{len(beats_mem)}" if beats_mem else "n/a",
        "counties_deploy_ready": f"{sum(r['deploy_ready'] for r in per_county)}/{len(per_county)}",
    }
    return {"has_rain": has_rain, "min_test_events": min_test_events,
            "aggregate": agg, "per_county": per_county}


def to_markdown(res: dict) -> str:
    a = res["aggregate"]
    lines = [
        "# Leave-one-county-out spatial generalization",
        "",
        f"- rainfall used: {res['has_rain']} | min test events/county: {res['min_test_events']}",
        f"- counties held out: {a['n_counties']} | median calibrated model AP: {a['median_model_ap']}",
        f"- model beats AB411 rule: {a['counties_model_beats_ab411']} counties | "
        f"beats station-memory: {a['counties_model_beats_memory']} | "
        f"deploy-ready: {a['counties_deploy_ready']}",
        "",
        "| held-out county | n | events | base | model AP | model AUC | AB411 AP | memory AP | beats AB411 | beats memory | deploy |",
        "|---|--:|--:|--:|--:|--:|--:|--:|:-:|:-:|:-:|",
    ]
    for r in sorted(res["per_county"], key=lambda x: -x["events"]):
        m = r["models"]
        mc = m["model_calibrated"]
        ab = m.get("baseline_ab411_rain", {}).get("ap", "-")
        mem = m["baseline_station_memory"].get("ap", "-")
        lines.append(
            f"| {r['county']} | {r['n']} | {r['events']} | {r['base_rate']} | "
            f"{mc.get('ap','-')} | {mc.get('roc_auc','-')} | {ab} | {mem} | "
            f"{'Y' if r['model_beats_ab411'] else 'N'} | {'Y' if r['model_beats_memory'] else 'N'} | "
            f"{'Y' if r['deploy_ready'] else 'N'} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--obs", default=None)
    ap.add_argument("--rain-dir", default=None)
    ap.add_argument("--min-test-events", type=int, default=150)
    ap.add_argument("--max-counties", type=int, default=None)
    ap.add_argument("--reveal-lag-days", type=int, default=0)
    ap.add_argument("--label", default="any", choices=["any", "enterococcus"])
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)
    obs = Path(args.obs) if args.obs else ob.default_obs_path()
    if not obs.exists():
        print(f"[spatial_holdout] obs not found: {obs}")
        return 2
    res = run_loco(obs, rain_dir=args.rain_dir, min_test_events=args.min_test_events,
                   max_counties=args.max_counties, reveal_lag_days=args.reveal_lag_days, label=args.label)
    out_dir = Path(args.out_dir) if args.out_dir else (Path(__file__).resolve().parents[2] / "reports" / "operational_benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    # newline="\n": force LF on every OS so expected/ reproduces byte-identically off-Windows too.
    (out_dir / "spatial_holdout.json").write_text(json.dumps(res, indent=2), encoding="utf-8", newline="\n")
    (out_dir / "spatial_holdout.md").write_text(to_markdown(res), encoding="utf-8", newline="\n")
    print(to_markdown(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
