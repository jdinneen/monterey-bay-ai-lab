#!/usr/bin/env python3
"""Gate the UPWELLING INDEX (CUTI/BEUTI) as a domoic-acid (DA) signal — the catalog's
PLANNED "strongest published DA driver", never previously fetched/tested.

Mechanism: upwelling (esp. BEUTI = nitrate flux) delivers the deep nitrate that fuels
Pseudo-nitzschia blooms -> particulate domoic acid. We add a leakage-safe windowed
upwelling feature (mean over the ~2 weeks BEFORE each pier visit, at the pier's latitude
band) and A/B it against the deployed DA headline (DA-history + Pseudo-nitzschia
precursor) on the same time-holdout + leave-one-station-out the lab already uses.

Reuses research/hab/da_forecast.py (load_panel / add_causal_features / fit_predict /
_scores) — does NOT edit it. Honest verdict: KEEP only if it lifts AP on the holdout AND
survives LOSO; a wash is a wash (the lab's prior is that DA drivers HURT).

    python -m research.hab.da_upwelling_gate
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "research" / "hab") not in sys.path:
    sys.path.insert(0, str(_ROOT / "research" / "hab"))
import da_forecast as da  # noqa: E402

UPW = _ROOT / "data" / "external_curated" / "upwelling_indices" / "upwelling_indices.parquet"
OUT = _ROOT / "reports" / "hab" / "da_upwelling_gate"


def add_upwelling(d: pd.DataFrame, window: int = 14) -> tuple[pd.DataFrame, list[str]]:
    """Add beuti_win / cuti_win: mean index over the `window` days strictly BEFORE each
    visit, at the pier's nearest 1-degree latitude band. Strictly causal."""
    if "latitude" not in d.columns:
        return d, []
    upw = pd.read_parquet(UPW)
    upw["date"] = pd.to_datetime(upw["date"], utc=True).dt.tz_localize(None).dt.normalize()
    d = d.copy()
    d["band"] = (pd.to_numeric(d["latitude"], errors="coerce").round().clip(32, 42)
                 .fillna(-999).astype("int64"))
    d["t"] = pd.to_datetime(d["time"]).dt.normalize()
    feats = []
    for idx in ("BEUTI", "CUTI"):
        col = f"{idx.lower()}_win"
        s = upw[upw["index_name"] == idx].copy()
        s["band"] = pd.to_numeric(s["latitude"], errors="coerce").round()
        s = s.dropna(subset=["band", "date", "value"]).copy()
        s["band"] = s["band"].astype("int64")
        s = s.sort_values(["band", "date"])
        # trailing window mean ENDING THE DAY BEFORE: shift(1) then rolling so a row at
        # date D summarizes [D-window, D-1] -> merge_asof at the visit date stays causal.
        s[col] = s.groupby("band")["value"].transform(
            lambda x: x.shift(1).rolling(window, min_periods=3).mean())
        merged = pd.merge_asof(
            d.sort_values("t"), s[["band", "date", col]].sort_values("date"),
            left_on="t", right_on="date", by="band", direction="backward",
            tolerance=pd.Timedelta(days=10))
        d[col] = pd.Series(merged[col].to_numpy(), index=merged.index).reindex(d.index)
        feats.append(col)
    return d.drop(columns=["band", "t"]), feats


def run(window: int = 14) -> dict:
    d = da.add_causal_features(da.load_panel())
    d, upw_feats = add_upwelling(d, window=window)
    if not upw_feats or d[upw_feats].notna().sum().sum() == 0:
        return {"error": "no upwelling features built (missing latitude or coverage)"}

    tr = d[d["year"] <= da.TRAIN_END]
    va = d[(d["year"] > da.TRAIN_END) & (d["year"] <= da.VALID_END)]
    te = d[d["year"] > da.VALID_END].copy()
    y = te["exceed"].to_numpy()
    base_rate = float(tr["exceed"].mean())
    base_feats = list(da.HEADLINE_FEATS)
    cand_feats = base_feats + upw_feats

    base_ap = da._scores(y, da.fit_predict(tr, va, te, base_feats), base_rate)["ap"]
    cand_ap = da._scores(y, da.fit_predict(tr, va, te, cand_feats), base_rate)["ap"]

    # leave-one-station-out (stations with >= 10 events), as in da_forecast
    ev = d.groupby("station")["exceed"].sum()
    base_oof, cand_oof, ys = [], [], []
    for st in ev[ev >= 10].index:
        tr_s = d[(d["station"] != st) & (d["year"] <= da.TRAIN_END)]
        va_s = d[(d["station"] != st) & (d["year"] > da.TRAIN_END) & (d["year"] <= da.VALID_END)]
        te_s = d[d["station"] == st].copy()
        if te_s.empty or tr_s["exceed"].nunique() < 2:
            continue
        ys.append(te_s["exceed"].to_numpy())
        base_oof.append(da.fit_predict(tr_s, va_s, te_s, base_feats))
        cand_oof.append(da.fit_predict(tr_s, va_s, te_s, cand_feats))
    loso = {}
    if ys:
        yy = np.concatenate(ys); br = float(yy.mean())
        loso = {"base_ap": da._scores(yy, np.concatenate(base_oof), br)["ap"],
                "cand_ap": da._scores(yy, np.concatenate(cand_oof), br)["ap"]}

    d_ap = round(cand_ap - base_ap, 4)
    d_loso = round(loso["cand_ap"] - loso["base_ap"], 4) if loso else None
    if d_ap < -0.005:
        verdict = "REJECT"
    elif d_ap < 0.005:
        verdict = "WASH"
    elif d_loso is not None and d_loso < 0:
        verdict = "REJECT"  # temporal win that fails LOSO = not real
    else:
        verdict = "KEEP"
    res = {"window_days": window, "n_test": int(len(te)), "events": int(y.sum()),
           "base_ap": round(base_ap, 4), "cand_ap": round(cand_ap, 4), "delta_ap": d_ap,
           "loso": loso, "loso_delta_ap": d_loso, "upwelling_feats": upw_feats,
           "verdict": verdict}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "verdict.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    return res


def main(argv=None) -> int:
    res = run()
    if "error" in res:
        print("ERROR:", res["error"]); return 1
    print(f"DA upwelling gate (window {res['window_days']}d, n_test {res['n_test']}, "
          f"{res['events']} events):")
    print(f"  base (DA+precursor) AP {res['base_ap']}  ->  +upwelling AP {res['cand_ap']}  "
          f"(dAP {res['delta_ap']:+.4f})")
    if res.get("loso"):
        print(f"  LOSO: base {res['loso']['base_ap']:.4f} -> cand {res['loso']['cand_ap']:.4f} "
              f"(dAP {res['loso_delta_ap']:+.4f})")
    print(f"  VERDICT: {res['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
