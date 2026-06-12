#!/usr/bin/env python3
"""Cross-source consistency check — a real data-QA gate, not exploration.

Independent sources that measure the SAME physical quantity (sea-surface temp from MUR / CDIP /
NDBC / CenCOOS; pressure from NDBC / ISD / ERA5; wind across products; ...) must agree. This
checks that on every rebuild and FAILS loudly when they stop, catching the bugs that silently
corrupt a normalized lakehouse:
  * a timezone / join / resample break  -> the series stop CO-MOVING (correlation collapses);
  * a unit / scale / offset error (Kelvin-for-Celsius, hPa-for-Pa, x10)  -> the series still
    correlate perfectly but no longer AGREE IN ABSOLUTE VALUE.

The second is the one a correlation-only view (correlation_discovery.py) is blind to: Pearson is
invariant to scale and offset, so two products can correlate 0.99 and be 273 degrees apart. This
check tests both. It is gate-able: exit code is non-zero if any group FAILs.

Usage:  python research/data_lab/source_consistency_check.py [--output-dir reports/data_lab]
Outputs (reports/data_lab/): source_consistency.json, source_consistency.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import correlation_discovery as CD  # noqa: E402  (reuse CATALOG + _daily + build_matrix + deseason)

OUTDIR = CD.ROOT / "reports" / "data_lab"
MIN_OVERLAP = 180          # min overlapping days for a pair to be judged
CO_MOVE_FAIL = 0.30        # deseasonalized r below this = the pair stopped co-moving (likely a bug)
CO_MOVE_WARN = 0.60        # below this but above FAIL = weak, worth a look

# Groups of columns (source::var, from correlation_discovery.CATALOG) that MEASURE THE SAME THING,
# in ONE shared unit, so both co-movement AND absolute agreement are meaningful. `abs_bias`/`abs_mad`
# are the tolerances (in that unit) for the median signed difference and the median |difference|.
# Tolerances are set to catch UNIT / SCALE / GROSS-OFFSET bugs (Kelvin-for-Celsius ~273, km/h-for-m/s
# ~3.6x, MJ/m2/day-for-W/m2 ~10x, hPa-for-Pa ~100x), NOT the few-unit spread that statewide daily
# MEANS legitimately have across different station footprints. Members are stable, broad-coverage
# references; sparse rotating buoy-network means (cdip_sst, mbal moorings) are excluded -- a noisy
# aggregate is a poor consistency anchor, not a bug. era5 surface_pressure is excluded from
# sea-level pressure (surface != MSL, a real ~11 hPa physical offset, not an error).
GROUPS = [
    {"q": "sea_surface_temp", "unit": "degC", "abs_bias": 4.0, "abs_mad": 4.0, "absolute": True,
     "members": ["mursst::sst", "ndbc::watertemp", "cencoos_oa::temp", "roms::temp"]},
    {"q": "air_temp", "unit": "degC", "abs_bias": 4.0, "abs_mad": 4.0, "absolute": True,
     "members": ["isd::airtemp", "ndbc::airtemp", "era5::temp", "nasapower::temp"]},
    {"q": "sea_level_pressure", "unit": "hPa", "abs_bias": 4.0, "abs_mad": 4.0, "absolute": True,
     "members": ["ndbc::pres", "isd::slp"]},
    {"q": "wind_speed", "unit": "m/s", "abs_bias": 4.0, "abs_mad": 4.0, "absolute": True,
     "members": ["ndbc::wspd", "ndbc_cwind::wspd", "isd::wspd", "nasapower::wind", "era5::wind",
                 "gridmet::wind"]},
    {"q": "solar_radiation", "unit": "W/m2", "abs_bias": 50.0, "abs_mad": 60.0, "absolute": True,
     "members": ["gridmet::srad", "nasapower::solar", "era5::swrad"]},
    {"q": "salinity", "unit": "psu", "abs_bias": 1.5, "abs_mad": 2.0, "absolute": True,
     "members": ["cencoos_oa::sal", "roms::salt"]},
    {"q": "wave_height", "unit": "m", "abs_bias": 0.9, "abs_mad": 1.0, "absolute": True,
     "members": ["ndbc::wvht", "cdip_wave::hs"]},
    {"q": "precip", "unit": "(mixed)", "abs_bias": None, "abs_mad": None, "absolute": False,
     "members": ["gridmet::precip", "nasapower::precip", "era5::precip", "ghcnd::precip"]},
]


def _judge_pair(wide: pd.DataFrame, anom: pd.DataFrame, a: str, b: str, grp: dict) -> dict | None:
    aligned = wide[[a, b]].dropna()
    n = len(aligned)
    if n < MIN_OVERLAP:
        return None
    rr = float(wide[a].corr(wide[b]))
    rd = float(anom[a].corr(anom[b]))                       # deseasonalized = the strict co-move test
    rec = {"a": a, "b": b, "n": n, "pearson": round(rr, 3), "deseason_pearson": round(rd, 3)}
    # co-movement verdict (catches tz/join breaks)
    co = "PASS" if rd >= CO_MOVE_WARN else ("WARN" if rd >= CO_MOVE_FAIL else "FAIL")
    rec["co_move"] = co
    # absolute agreement (catches unit/scale/offset bugs that correlation cannot see)
    if grp["absolute"]:
        diff = (aligned[a] - aligned[b])
        bias = float(diff.median())
        mad = float(diff.abs().median())
        rec.update({"bias": round(bias, 3), "median_abs_diff": round(mad, 3),
                    "abs_tol_bias": grp["abs_bias"], "abs_tol_mad": grp["abs_mad"]})
        rec["absolute"] = ("PASS" if abs(bias) <= grp["abs_bias"] and mad <= grp["abs_mad"]
                           else "FAIL")
    else:
        rec["absolute"] = "NA"
    worst = ["PASS", "WARN", "NA", "FAIL"]
    rec["verdict"] = max([co, rec["absolute"]], key=worst.index)
    return rec


def run(output_dir: Path | None = None) -> dict:
    out_dir = Path(output_dir) if output_dir else OUTDIR
    wide, _ = CD.build_matrix()
    wide = wide[wide.index.notna()]
    anom = CD._deseasonalize(wide)

    groups_out, all_fail = [], 0
    for grp in GROUPS:
        present = [m for m in grp["members"] if m in wide.columns]
        pairs = []
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                rec = _judge_pair(wide, anom, present[i], present[j], grp)
                if rec:
                    pairs.append(rec)
        verdicts = [p["verdict"] for p in pairs]
        gv = "PASS"
        if "FAIL" in verdicts:
            gv = "FAIL"
        elif "WARN" in verdicts:
            gv = "WARN"
        elif not pairs:
            gv = "INSUFFICIENT_OVERLAP"
        all_fail += sum(v == "FAIL" for v in verdicts)
        groups_out.append({"quantity": grp["q"], "unit": grp["unit"], "absolute_checked": grp["absolute"],
                           "members_present": present, "n_pairs": len(pairs),
                           "group_verdict": gv, "pairs": pairs})

    overall = "FAIL" if any(g["group_verdict"] == "FAIL" for g in groups_out) else (
        "WARN" if any(g["group_verdict"] == "WARN" for g in groups_out) else "PASS")
    out = {
        "what": "cross-source consistency: same-quantity products must co-move AND agree in absolute units",
        "thresholds": {"min_overlap_days": MIN_OVERLAP, "co_move_fail": CO_MOVE_FAIL,
                       "co_move_warn": CO_MOVE_WARN},
        "overall": overall, "n_failing_pairs": all_fail,
        "groups": groups_out,
        "note": "co_move uses the DESEASONALIZED correlation (a tz/join break shows here even when "
                "both series share a seasonal cycle). absolute uses median signed bias + median "
                "|difference| in the group's unit (a Kelvin/hPa/x10 bug shows here even at r=0.99).",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "source_consistency.json").write_text(json.dumps(out, indent=2, default=str),
                                                     encoding="utf-8")
    (out_dir / "source_consistency.md").write_text(_render_md(out), encoding="utf-8")
    return out


def _render_md(out: dict) -> str:
    L = ["# Cross-source consistency check (data-QA gate)", "",
         f"What: {out['what']}.", "",
         f"**Overall: {out['overall']}** ({out['n_failing_pairs']} failing pairs). "
         f"co-move FAIL < {out['thresholds']['co_move_fail']} deseasonalized r; "
         f"absolute FAIL when median bias or |diff| exceeds the per-quantity unit tolerance.", "",
         "| quantity | unit | members | pairs | verdict |", "|---|---|--:|--:|:-:|"]
    for g in out["groups"]:
        L.append(f"| {g['quantity']} | {g['unit']} | {len(g['members_present'])} | {g['n_pairs']} "
                 f"| {g['group_verdict']} |")
    L += ["", "## Per-pair detail", ""]
    for g in out["groups"]:
        L.append(f"### {g['quantity']} ({g['unit']}) -- {g['group_verdict']}")
        if not g["pairs"]:
            L += ["- insufficient overlap", ""]
            continue
        L += ["| a | b | deseason r | co-move | bias | median \\|diff\\| | absolute | n |",
              "|---|---|--:|:-:|--:|--:|:-:|--:|"]
        for p in g["pairs"]:
            L.append(f"| `{p['a']}` | `{p['b']}` | {p['deseason_pearson']} | {p['co_move']} | "
                     f"{p.get('bias','-')} | {p.get('median_abs_diff','-')} | {p['absolute']} | {p['n']} |")
        L.append("")
    L += ["## How to read this", "",
          "- **co-move FAIL** -> two products that should track each other no longer do: suspect a "
          "timezone / join / resample regression in one of them.",
          "- **absolute FAIL** -> they co-move but disagree in value beyond tolerance: suspect a "
          "unit (Kelvin/Celsius, hPa/Pa), scale (x10), or offset bug.",
          "- **WARN** -> weaker than expected; not necessarily broken (e.g. wind at different "
          "anemometer heights, river vs ocean water temp). Tolerances are explicit above; recalibrate "
          "in `GROUPS` if a WARN is a known-real geophysical difference.",
          out["note"]]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default=str(OUTDIR))
    ap.add_argument("--warn-is-failure", action="store_true",
                    help="exit non-zero on WARN too (stricter gate)")
    args = ap.parse_args(argv)
    out = run(Path(args.output_dir))
    print(f"[consistency] overall={out['overall']} | {out['n_failing_pairs']} failing pairs across "
          f"{len(out['groups'])} quantity groups")
    print(f"[consistency] wrote {Path(args.output_dir) / 'source_consistency.md'} (+ .json)")
    bad = out["overall"] == "FAIL" or (args.warn_is_failure and out["overall"] == "WARN")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
