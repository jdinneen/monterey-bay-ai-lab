#!/usr/bin/env python3
"""Compare our DA forecast model against NOAA C-HARM at the CalHABMAP pier points.

C-HARM (California Harmful Algae Risk Mapping, NOAA CoastWatch) is the operational benchmark.
We use **C-HARM v3.1 nowcast** `particulate_domoic` = P(particulate DA > 500 ng/L), which is
EXACTLY our label (pDA >= 0.5 ng/mL). For each pier we sample the nearest valid ocean grid cell
(piers sit on land-masked cells) and the daily nowcast on each lab-visit date.

Honest framing of the matchup (stated, not hidden):
  * C-HARM v3.1 only covers 2022-11 -> 2026-06, so the comparison is on that overlap window
    (a subset of our >=2021 test set). Both models are scored on the IDENTICAL rows.
  * C-HARM nowcast uses SAME-DAY satellite+ROMS; our model uses only PRIOR-visit data (~1 week
    old). So C-HARM has an information advantage -- this is "can our cheap forecast match the
    agency's same-day nowcast", a deliberately hard bar for us.

Outputs (reports/hab/): charm_comparison.json, charm_comparison.md
Cache: data/external_raw/charm/<station>.parquet (C-HARM series; re-runs are fast).
"""
from __future__ import annotations

import io
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import da_forecast as F  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "external_raw" / "charm"
OUTDIR = ROOT / "reports" / "hab"
GRID = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/wvcharmV3_0day.csv"
CHARM_START, CHARM_END = "2022-11-01", "2026-06-08"
VAR = "particulate_domoic"


def _get(url: str, timeout: int = 400) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "mbal-datafetch/0.1"})
    return urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8", "replace")


def nearest_ocean_cell(lat: float, lon360: float, d: float = 0.2):
    """Nearest non-NaN C-HARM cell to a pier (piers fall on land-masked cells)."""
    url = (f"{GRID}?{VAR}%5B(last)%5D%5B({lat-d}):({lat+d})%5D%5B({lon360-d}):({lon360+d})%5D")
    df = pd.read_csv(io.StringIO(_get(url, 150)), skiprows=[1])
    df[VAR] = pd.to_numeric(df[VAR], errors="coerce")
    v = df.dropna(subset=[VAR]).copy()
    if v.empty:
        return None
    v["dist"] = np.hypot(v["latitude"] - lat, ((v["longitude"] - lon360 + 180) % 360 - 180))
    n = v.sort_values("dist").iloc[0]
    return float(n["latitude"]), float(n["longitude"])


def charm_series(station: str, lat: float, lon: float) -> pd.DataFrame:
    cache = CACHE / f"{station}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    cell = nearest_ocean_cell(lat, lon % 360)
    if cell is None:
        df = pd.DataFrame(columns=["charm_date", "charm_p"])
    else:
        clat, clon = cell
        url = f"{GRID}?{VAR}%5B({CHARM_START}):({CHARM_END})%5D%5B({clat})%5D%5B({clon})%5D"
        ts = pd.read_csv(io.StringIO(_get(url)), skiprows=[1])
        df = pd.DataFrame({
            "charm_date": pd.to_datetime(ts["time"], utc=True).dt.tz_localize(None).dt.normalize(),
            "charm_p": pd.to_numeric(ts[VAR], errors="coerce"),
        }).dropna()
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


def model_test_predictions() -> pd.DataFrame:
    """Our model's calibrated predictions on the held-out test rows (train<=2018)."""
    d = F.add_causal_features(F.load_panel())
    tr = d[d["year"] <= F.TRAIN_END]
    va = d[(d["year"] > F.TRAIN_END) & (d["year"] <= F.VALID_END)]
    te = d[d["year"] > F.VALID_END].copy()
    te["p_model"] = F.fit_predict(tr, va, te, F.HEADLINE_FEATS)
    return te[["station", "time", "exceed", "p_model", "latitude", "longitude"]]


def _scores(y, p):
    y = np.asarray(y); p = np.asarray(p)
    return {"n": int(len(y)), "events": int(y.sum()),
            "ap": round(float(average_precision_score(y, p)), 4) if 0 < y.sum() else None,
            "roc_auc": round(float(roc_auc_score(y, p)), 4) if 0 < y.sum() < len(y) else None}


def main() -> int:
    te = model_test_predictions()
    te["visit_date"] = te["time"].dt.normalize()
    rows = []
    for st, g in te.groupby("station"):
        lat, lon = g["latitude"].iloc[0], g["longitude"].iloc[0]
        cs = charm_series(st, lat, lon)
        if cs.empty:
            continue
        g = g.sort_values("visit_date")
        cs = cs.sort_values("charm_date")
        merged = pd.merge_asof(g, cs, left_on="visit_date", right_on="charm_date",
                               direction="nearest", tolerance=pd.Timedelta(days=3))
        rows.append(merged)
    allm = pd.concat(rows, ignore_index=True)
    common = allm[allm["charm_p"].notna()].copy()  # rows where BOTH models have a value

    overall = {
        "charm_dataset": "wvcharmV3_0day (C-HARM v3.1 nowcast, P[particulate DA > 500 ng/L])",
        "overlap_window": f"{CHARM_START}..{CHARM_END}",
        "n_common_rows": int(len(common)), "n_common_events": int(common["exceed"].sum()),
        "our_model": _scores(common["exceed"], common["p_model"]),
        "charm_nowcast": _scores(common["exceed"], common["charm_p"]),
    }
    # per-station head-to-head (where it has events)
    per = {}
    for st, g in common.groupby("station"):
        if g["exceed"].sum() == 0:
            continue
        ours = _scores(g["exceed"], g["p_model"]); ch = _scores(g["exceed"], g["charm_p"])
        per[st.replace("HABs-", "")] = {
            "n": ours["n"], "events": ours["events"],
            "our_ap": ours["ap"], "charm_ap": ch["ap"],
            "our_auc": ours["roc_auc"], "charm_auc": ch["roc_auc"],
            "we_win_ap": (ours["ap"] or 0) >= (ch["ap"] or 0),
        }
    wins = sum(v["we_win_ap"] for v in per.values())
    overall["per_station"] = per
    overall["headline"] = {
        "stations_compared": len(per), "we_beat_charm_ap": f"{wins}/{len(per)}",
        "our_ap": overall["our_model"]["ap"], "charm_ap": overall["charm_nowcast"]["ap"],
        "our_auc": overall["our_model"]["roc_auc"], "charm_auc": overall["charm_nowcast"]["roc_auc"],
        "charm_mean_prob": round(float(common["charm_p"].mean()), 3),
        "actual_event_rate": round(float(common["exceed"].mean()), 3),
    }

    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "charm_comparison.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    h = overall["headline"]
    md = [f"# DA forecast vs NOAA C-HARM (operational benchmark)", "",
          f"C-HARM dataset: `{overall['charm_dataset']}`. Overlap window "
          f"{overall['overlap_window']} ({h['stations_compared']} stations, "
          f"{overall['n_common_rows']} common visits, {overall['n_common_events']} events).", "",
          "C-HARM nowcast uses **same-day** satellite+ROMS; our model uses only **prior-visit** "
          "data (~1 week old). Both scored on identical rows. AP at a "
          f"{h['actual_event_rate']:.1%} event rate.", "",
          "| model | pooled AP | pooled ROC-AUC |", "|---|--:|--:|",
          f"| our forecast (prior-visit) | {h['our_ap']} | {h['our_auc']} |",
          f"| C-HARM v3.1 nowcast (same-day) | {h['charm_ap']} | {h['charm_auc']} |", "",
          f"C-HARM mean predicted probability = {h['charm_mean_prob']} vs actual event rate "
          f"{h['actual_event_rate']} (calibration sanity).", "",
          f"**Per-station: we match/beat C-HARM AP in {h['we_beat_charm_ap']} stations.**", "",
          "| station | n | events | our AP | C-HARM AP | our AUC | C-HARM AUC | we win |",
          "|---|--:|--:|--:|--:|--:|--:|---|"]
    for st, v in sorted(per.items(), key=lambda x: -(x[1]["events"])):
        md.append(f"| {st} | {v['n']} | {v['events']} | {v['our_ap']} | {v['charm_ap']} | "
                  f"{v['our_auc']} | {v['charm_auc']} | {'Y' if v['we_win_ap'] else 'n'} |")
    (OUTDIR / "charm_comparison.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(overall["headline"], indent=2))
    print(f"\nwrote {OUTDIR/'charm_comparison.json'}, charm_comparison.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
