#!/usr/bin/env python3
"""Era-local (prequential) recalibration — fix the regime-shift calibration failures.

A single isotonic map fit on the 2020-21 validation era is STATIC: under a regime
shift it goes stale and actively harms calibration (San Diego post-2022 ECE 0.04->0.28;
San Francisco not deploy-ready). The operational reality is that lab results keep
arriving (with a ~1-2 day lag), so the calibrator can keep learning from recently
*revealed* labels — exactly the continual-learning setting.

This recalibrates the probability map prequentially: for each forward time block it
refits isotonic on ONLY the labels already revealed by then (sample_date <= block
start - lag_days), warm-started by the static 2020-21 map for the cold start. It never
refits features and never uses a label whose result has not yet returned.

Guardrail (and a unit test asserts it): the calibrator fit for a block contains no
sample whose label reveal-time is on/after that block's start.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from research.bacteria import operational_benchmark as ob  # noqa: E402


def _default_clf():
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_iter=600, learning_rate=0.05, l2_regularization=1.0,
        class_weight="balanced", early_stopping=True, validation_fraction=0.1,
        random_state=42,
    )


def _revealed_mask(dates: pd.Series, block_start: pd.Timestamp, lag_days: int) -> np.ndarray:
    """Samples whose label has been REVEALED before this block starts: sample taken at
    least ``lag_days`` before block_start (so its lab result has returned)."""
    return (dates <= (block_start - pd.Timedelta(days=lag_days))).to_numpy()


def prequential_recalibrate(dates, raw, y, static_iso, *, lag_days: int = 2,
                            block_freq: str = "M", min_calib: int = 300) -> np.ndarray:
    """Forward-walking isotonic recalibration using only revealed labels.

    ``static_iso`` (the 2020-21 map) is the cold-start fallback for blocks without
    enough revealed labels. Returns probabilities aligned to the input order.
    """
    from sklearn.isotonic import IsotonicRegression

    dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    raw = np.asarray(raw, dtype=float)
    y = np.asarray(y)
    out = np.asarray(static_iso.predict(raw) if static_iso is not None else raw, dtype=float).copy()
    blocks = dates.dt.to_period(block_freq)
    for b in pd.unique(blocks):
        block_mask = (blocks == b).to_numpy()
        revealed = _revealed_mask(dates, b.start_time, lag_days)
        if revealed.sum() >= min_calib and len(np.unique(y[revealed])) > 1:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(raw[revealed], y[revealed])
            out[block_mask] = iso.predict(raw[block_mask])
    return out


def run_online_recal(obs_path: Path, rain_dir=None, clf=None, *, lag_days: int = 2,
                     block_freq: str = "M", min_calib: int = 300) -> dict:
    df = ob.load_station_days(obs_path)
    df = ob.add_causal_features(df, reveal_lag_days=lag_days)
    feats = list(ob.FEATS)
    if rain_dir is not None:
        df = ob.add_rain_features(df, Path(rain_dir))
        if "rain_3d" in df.columns and df["rain_3d"].notna().any():
            feats += ob.RAIN_FEATS

    tr = df[df["sample_date"] <= "2019-12-31"]
    va = df[(df["sample_date"] >= "2020-01-01") & (df["sample_date"] <= "2021-12-31")]
    te = df[df["sample_date"] >= "2022-01-01"].copy()

    # drop features constant in the training set (sklearn>=1.9 HGBT binning errors on a
    # single-distinct-value feature); mirrors operational_benchmark's guard.
    feats = [c for c in feats if tr[c].notna().sum() >= 2 and tr[c].nunique(dropna=True) > 1]
    clf = clf or _default_clf()
    clf.fit(tr[feats].astype(float), tr["exceed"].to_numpy())
    te["raw"] = clf.predict_proba(te[feats].astype(float))[:, 1]

    from sklearn.isotonic import IsotonicRegression

    static = None
    if va["exceed"].nunique() > 1:
        static = IsotonicRegression(out_of_bounds="clip")
        static.fit(clf.predict_proba(va[feats].astype(float))[:, 1], va["exceed"].to_numpy())
    te["p_static"] = static.predict(te["raw"].to_numpy()) if static is not None else te["raw"]
    te = te.sort_values("sample_date").reset_index(drop=True)
    te["p_online"] = prequential_recalibrate(
        te["sample_date"], te["raw"].to_numpy(), te["exceed"].to_numpy(),
        static, lag_days=lag_days, block_freq=block_freq, min_calib=min_calib)

    strata = {
        "ALL": te,
        "EXCLUDE_SAN_DIEGO": te[te["county"] != "San Diego"],
        "SAN_DIEGO": te[te["county"] == "San Diego"],
        "SAN_FRANCISCO": te[te["county"] == "San Francisco"],
        "MONTEREY": te[te["county"].isin(ob.MONTEREY_COUNTIES)],
    }
    out = {"lag_days": lag_days, "block_freq": block_freq, "strata": {}}
    for sname, part in strata.items():
        if len(part) == 0 or part["exceed"].nunique() < 2:
            out["strata"][sname] = {"note": "empty/degenerate"}
            continue
        y = part["exceed"].to_numpy()
        base = float(y.mean())
        mem_ece = ob.score(y, part["station_prior_rate"].fillna(base).to_numpy(), base)["ece"]
        s_static = ob.score(y, part["p_static"].to_numpy(), base)
        s_online = ob.score(y, part["p_online"].to_numpy(), base)
        out["strata"][sname] = {
            "n": int(len(part)), "events": int(y.sum()),
            "static": {"ap": s_static["ap"], "ece": s_static["ece"]},
            "online": {"ap": s_online["ap"], "ece": s_online["ece"]},
            "station_memory_ece": mem_ece,
            "static_deploy_ready": bool(s_static["ece"] <= mem_ece + 0.05),
            "online_deploy_ready": bool(s_online["ece"] <= mem_ece + 0.05),
            "online_fixes_calibration": bool(s_online["ece"] < s_static["ece"]),
        }
    return out


def to_markdown(res: dict) -> str:
    lines = ["# Era-local (prequential) recalibration vs static",
             "", f"- lag_days={res['lag_days']} | block={res['block_freq']}", "",
             "| stratum | n | events | static ECE | **online ECE** | memory ECE | static deploy | **online deploy** |",
             "|---|--:|--:|--:|--:|--:|:-:|:-:|"]
    for s, d in res["strata"].items():
        if "static" not in d:
            lines.append(f"| {s} | - | - | - | - | - | - | - |")
            continue
        lines.append(f"| {s} | {d['n']} | {d['events']} | {d['static']['ece']} | "
                     f"{d['online']['ece']} | {d['station_memory_ece']} | "
                     f"{'Y' if d['static_deploy_ready'] else 'N'} | "
                     f"{'Y' if d['online_deploy_ready'] else 'N'} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--obs", default=None)
    ap.add_argument("--rain-dir", default=None)
    ap.add_argument("--lag-days", type=int, default=2)
    ap.add_argument("--block-freq", default="M")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)
    obs = Path(args.obs) if args.obs else ob.default_obs_path()
    if not obs.exists():
        print(f"[online_recalibration] obs not found: {obs}")
        return 2
    res = run_online_recal(obs, rain_dir=args.rain_dir, lag_days=args.lag_days, block_freq=args.block_freq)
    out_dir = Path(args.out_dir) if args.out_dir else (_REPO_ROOT / "reports" / "operational_benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    # newline="\n": force LF on every OS so expected/ reproduces byte-identically off-Windows too.
    (out_dir / "online_recalibration.json").write_text(json.dumps(res, indent=2), encoding="utf-8", newline="\n")
    (out_dir / "online_recalibration.md").write_text(to_markdown(res), encoding="utf-8", newline="\n")
    print(to_markdown(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
