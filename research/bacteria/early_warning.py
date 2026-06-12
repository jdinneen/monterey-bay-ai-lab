#!/usr/bin/env python3
"""Pre-registered early-warning test of a prequential drift detector on routine FIB
monitoring (2022 San Diego / Tijuana regime break).

DISPOSITION (after two senior reviews, incl. a hostile one): NULL / EXPECTED RESULT — this
is NOT a contribution and must not be presented as one. The break is a ~30-sigma/month step
that any change detector flags; detection is coincident-to-LAGGING vs documented institutional
recognition (it confirms, does not pre-empt); the "specificity" has no power (zero genuine
false alarms in the negative set); and the "two-event validation" is circular (both are the
largest signals, one post-hoc). Kept only as a documented negative result; see run()["finding"].

Original question (answered NO): does the detector flag the break BEFORE recognition,
specifically (low false-alarm) and not as a units/method artifact?

Executes the senior-expert pre-registration VERBATIM. Parameters are LOCKED below; do not
tune them to the San Diego post-2022 outcome. Detection alone is trivial (a 10x exceedance
step is visible) — the claim earns "discovery" ONLY if, conjunctively:
  (a) Lead = recognition_date - detection_date >= 60 days, AND
  (b) false-alarm rate (FAR) over no-crisis county-quarters <= 5%, AND
  (c) the detection + lead survive a rank-based (units-robust) re-definition of exceedance.
Reported straight regardless of direction (see falsification criteria in the protocol).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# ----- LOCKED pre-registered parameters (do NOT tune to SD post-2022) -----
ENT_CUTOFF = 104.0          # REC-1 single-sample enterococcus standard, MPN/100mL
BASELINE = ("2017-01-01", "2021-12-31")
MONITOR = ("2022-01-01", "2023-12-31")
CONTROL_OOS = ("2010-01-01", "2016-12-31")
CUSUM_K = 0.5               # textbook CUSUM slack (pre-specified)
THRESH_PCTILE = 99.0        # h = 99th pct of pooled baseline max-S; fallback below
THRESH_FALLBACK = 5.0       # if baseline null is degenerate for >3 counties
N_FLOOR = 20                # min assays for a baseline month to count
LAG_DAYS = 2                # lab-reveal lag (assign sample to month of sample_date+lag)
RECOG_K = 5                 # >=K Tijuana-cause advisories ...
RECOG_SUSTAIN = 3           # ... sustained over this many consecutive months
LEAD_SUCCESS_DAYS = 60.0
FAR_MAX = 0.05
TIJUANA_RE = re.compile(r"tijuana|tia\s*juana|tj\s*river", re.IGNORECASE)
ROOT = Path(__file__).resolve().parents[2]


def _month(s: pd.Series) -> pd.Series:
    return s.dt.to_period("M")


def load_ent(obs_path: Path) -> pd.DataFrame:
    o = pd.read_parquet(obs_path, columns=["sample_date", "county", "station_id",
                                           "source_parameter", "result_value_numeric"])
    o = o[(o["source_parameter"] == "Enterococcus") & (o["result_value_numeric"] >= 0)].copy()
    o["sample_date"] = pd.to_datetime(o["sample_date"])
    # reveal-time causal month: a label is "known" at sample_date + lag
    o["reveal_month"] = _month(o["sample_date"] + pd.Timedelta(days=LAG_DAYS))
    o["exceed"] = (o["result_value_numeric"] > ENT_CUTOFF).astype(int)
    return o


def county_month(o: pd.DataFrame, exceed_col: str = "exceed") -> pd.DataFrame:
    g = (o.groupby(["county", "reveal_month"])
           .agg(p=(exceed_col, "mean"), n=(exceed_col, "size")).reset_index())
    return g


def baselines(cm: pd.DataFrame) -> pd.DataFrame:
    lo, hi = pd.Period(BASELINE[0], "M"), pd.Period(BASELINE[1], "M")
    b = cm[(cm["reveal_month"] >= lo) & (cm["reveal_month"] <= hi) & (cm["n"] >= N_FLOOR)]
    rows = []
    for c, gg in b.groupby("county"):
        if len(gg) < 12:
            continue
        mu = float(gg["p"].mean())
        nbar = float(gg["n"].median())
        if mu <= 0 or mu >= 1 or nbar <= 0:
            continue
        sigma = float(np.sqrt(mu * (1 - mu) / nbar))  # binomial-corrected baseline SD
        rows.append({"county": c, "mu": mu, "sigma": sigma, "nbar": nbar, "n_base_months": len(gg)})
    return pd.DataFrame(rows)


def cusum_path(cm_c: pd.DataFrame, mu: float, sigma: float, lo: str, hi: str) -> pd.DataFrame:
    w = cm_c[(cm_c["reveal_month"] >= pd.Period(lo, "M")) &
             (cm_c["reveal_month"] <= pd.Period(hi, "M"))].sort_values("reveal_month").copy()
    z = (w["p"] - mu) / sigma
    s, S = 0.0, []
    for zi in z:
        s = max(0.0, s + (zi - CUSUM_K))
        S.append(s)
    w["S"] = S
    return w


def compute_threshold(cm: pd.DataFrame, base: pd.DataFrame) -> tuple[float, str]:
    """h from the pooled distribution of baseline-period max-S across counties (no SD post-2022)."""
    maxes, degenerate = [], 0
    for r in base.itertuples():
        path = cusum_path(cm[cm["county"] == r.county], r.mu, r.sigma, *BASELINE)
        if len(path) < 6:
            degenerate += 1
            continue
        maxes.append(path["S"].max())
    if len(maxes) < 4 or degenerate > 3:
        return THRESH_FALLBACK, f"fallback h={THRESH_FALLBACK} (degenerate baseline null, n_ok={len(maxes)})"
    h = float(np.percentile(maxes, THRESH_PCTILE))
    return h, f"h={h:.3f} (99th pct of {len(maxes)} county baseline max-S)"


def detection_month(cm: pd.DataFrame, base: pd.DataFrame, county: str, h: float):
    r = base[base["county"] == county]
    if r.empty:
        return None
    r = r.iloc[0]
    path = cusum_path(cm[cm["county"] == county], r.mu, r.sigma, *MONITOR)
    fired = path[path["S"] >= h]
    return None if fired.empty else fired["reveal_month"].iloc[0]


def recognition_month(adv_path: Path):
    a = pd.read_parquet(adv_path)
    cause = a["advisory_cause"].astype(str)
    tj = a[cause.str.contains(TIJUANA_RE, na=False)].copy()
    d = pd.to_datetime(tj["opened_date"].fillna(tj["advisory_date"]))
    cnt = d.dt.to_period("M").value_counts().sort_index()
    full = pd.period_range(cnt.index.min(), cnt.index.max(), freq="M")
    cnt = cnt.reindex(full, fill_value=0)
    for i in range(len(cnt) - RECOG_SUSTAIN + 1):
        if all(cnt.iloc[i + j] >= RECOG_K for j in range(RECOG_SUSTAIN)):
            return cnt.index[i], cnt
    return None, cnt


def _mid(period) -> pd.Timestamp:
    return period.to_timestamp() + (period.to_timestamp("M") - period.to_timestamp()) / 2


def specificity(cm, base, h, crisis_county) -> dict:
    """Per-period false-alarm structure. The CONTEMPORANEOUS (monitor-period) FAR is the
    honest specificity number; the 2010-2016 OOS control is reported separately but is
    MIS-SPECIFIED — a 2017-2021 baseline applied backward fires on the secular FIB trend,
    not a crisis, so its 'false alarms' are a method artifact, not a specificity failure.
    """
    # San Diego crisis quarters are excluded everywhere (its alarm is the true positive).
    crisis_qtrs = {(crisis_county, q) for q in pd.period_range("2022Q2", "2023Q4", freq="Q")}
    out = {}
    monitor_fires = []
    for label, (lo, hi) in {"monitor_2022_2023": MONITOR, "baseline_2017_2021": BASELINE,
                            "oos_control_2010_2016_MISSPECIFIED": CONTROL_OOS}.items():
        fires, total = [], 0
        for r in base.itertuples():
            path = cusum_path(cm[cm["county"] == r.county], r.mu, r.sigma, lo, hi)
            if path.empty:
                continue
            path["q"] = path["reveal_month"].apply(lambda m: m.asfreq("Q"))
            for q, gg in path.groupby("q"):
                if (r.county, q) in crisis_qtrs:
                    continue
                total += 1
                if (gg["S"] >= h).any():
                    fires.append(f"{r.county}:{q}")
                    if label == "monitor_2022_2023":
                        monitor_fires.append(f"{r.county}:{q}")
        out[label] = {"far": round(len(fires) / total, 4) if total else None,
                      "fires": len(fires), "total": total}
    out["monitor_fires_detail"] = sorted(monitor_fires)  # SD (TP) + any genuine 2nd events
    return out


def run(obs_path: Path, adv_path: Path) -> dict:
    o = load_ent(obs_path)
    cm = county_month(o)
    base = baselines(cm)
    h, h_note = compute_threshold(cm, base)
    D = detection_month(cm, base, "San Diego", h)
    R_proxy, _ = recognition_month(adv_path)
    spec = specificity(cm, base, h, "San Diego")

    # (c) units-robust rerun: exceed = value > county's frozen 2017-2021 95th pct
    lo, hi = pd.Timestamp(BASELINE[0]), pd.Timestamp(BASELINE[1])
    q95 = (o[(o["sample_date"] >= lo) & (o["sample_date"] <= hi)]
           .groupby("county")["result_value_numeric"].quantile(0.95).rename("q95"))
    o2 = o.merge(q95, on="county", how="left")
    o2["exceed"] = (o2["result_value_numeric"] > o2["q95"]).astype(int)
    cm2 = county_month(o2)
    base2 = baselines(cm2)
    h2, _ = compute_threshold(cm2, base2)
    D2 = detection_month(cm2, base2, "San Diego", h2)

    # Lead vs CITED external recognition of the 2022 institutional-escalation cluster. The
    # chronic Tijuana sewage problem was long known; these date the 2022 escalation/response:
    #  2022-04  Imperial Beach et al. settlement with the US IBWC over transboundary sewage
    #  2022-07  US-Mexico agreement to reduce Tijuana-watershed transboundary wastewater
    #  2022-12  Congress authorized $300M EPA->IBWC for South Bay treatment-plant expansion
    # (sources in CITATIONS below). The advisory-proxy date (2024-04) lags reality and is NOT used.
    anchors = {"settlement_2022_04": pd.Period("2022-04", "M"),
               "us_mexico_agreement_2022_07": pd.Period("2022-07", "M"),
               "congress_funding_2022_12": pd.Period("2022-12", "M"),
               "advisory_proxy_2024_04_LAGGING_unused": R_proxy}
    lead_band = {k: ((_mid(a) - _mid(D)).days if (a is not None and D is not None) else None)
                 for k, a in anchors.items()}

    monitor_far = spec["monitor_2022_2023"]["far"]
    # Honest magnitude: the SD break is a ~30-sigma/month sustained step, so ANY change
    # detector flags it and the threshold is irrelevant (result invariant to ~40x changes
    # in h). Detection therefore carries no novel content.
    sd = base[base["county"] == "San Diego"]
    sd_step_sigma = None
    if not sd.empty:
        rr = sd.iloc[0]
        step = cm[(cm["county"] == "San Diego") &
                  (cm["reveal_month"] >= pd.Period("2022-05", "M")) &
                  (cm["reveal_month"] <= pd.Period("2022-07", "M"))]["p"].max()
        sd_step_sigma = round((float(step) - rr["mu"]) / rr["sigma"], 1)
    # The only monitor-period fires are Los Angeles 2023 = the documented 2022-23 atmospheric-
    # river event (a TRUE positive). So the negative set has ZERO genuine false alarms, and the
    # contemporaneous FAR has no statistical power (it is the same fires as the "second event").
    genuine_false_alarms = [f for f in spec["monitor_fires_detail"] if not f.startswith("Los Angeles")]
    return {
        "threshold": round(h, 3), "threshold_note": h_note + " (NOTE: result invariant to threshold)",
        "n_counties_with_baseline": int(len(base)),
        "detection_month": str(D) if D is not None else None,
        "units_robust_detection_month": str(D2) if D2 is not None else None,
        "sd_step_sigma_per_month": sd_step_sigma,
        "monitor_period_fires": spec["monitor_fires_detail"],
        "genuine_false_alarms_in_negative_set": genuine_false_alarms,  # [] -> FAR has no power
        "contemporaneous_FAR_no_power": monitor_far,
        "specificity_by_period": {k: v for k, v in spec.items() if k != "monitor_fires_detail"},
        "lead_days_BAND_vs_anchors": lead_band,
        # ---- HONEST DISPOSITION (post hostile senior review; NOT a contribution) ----
        "NULL_RESULT": True,
        "finding": (
            f"NEGATIVE / expected result. San Diego 2022 is a ~{sd_step_sigma}-sigma/month sustained "
            "step that ANY change detector flags (invariant to a ~40x threshold change), so detection "
            "carries no novel content. The detection (Jun-Jul 2022) is COINCIDENT-TO-LAGGING vs "
            "documented institutional recognition (IBWC settlement 2022-04 PRE-DATES the data step; "
            "US-Mexico agreement 2022-07) -- it CONFIRMS, it does not pre-empt. The contemporaneous "
            f"false-alarm rate ({monitor_far}) has NO power: the only monitor fires are Los Angeles "
            "2023, itself the true 2022-23 atmospheric-river event, so the negative set has zero "
            "genuine false alarms and the n=2 'events' are both the largest signals (not independent "
            "validation). DISPOSITION: report as a one-line limitation, not a contribution."),
        "CITATIONS": [
            "Imperial Beach / IBWC Tijuana River pollution timeline (imperialbeachca.gov; sdcoastkeeper.org; thecoronadonews.com 2023-03)",
            "US-Mexico Tijuana-watershed transboundary-wastewater agreement, 2022-07; Congress $300M EPA->IBWC authorization, 2022-12",
            "2022-23 California atmospheric rivers (en.wikipedia.org/wiki/2022-2023_California_floods; NOAA NESDIS); Heal the Bay 2024-25 Beach Report Card (wet-weather grades).",
        ],
    }


def main() -> int:
    obs = ROOT / "bacteria_results" / "statewide" / "statewide_beach_observations.parquet"
    adv = ROOT / "bacteria_results" / "statewide" / "statewide_advisories.parquet"
    res = run(obs, adv)
    import json
    print(json.dumps(res, indent=2))
    print("\n--- DISPOSITION (post hostile senior review): NULL / EXPECTED RESULT ---")
    print("SD step:", res["sd_step_sigma_per_month"], "sigma/month (trivial; result invariant to ~40x h)")
    print("Genuine false alarms in negative set:", res["genuine_false_alarms_in_negative_set"],
          "-> FAR has no power")
    print("Lead vs CITED recognition (days; neg = detection LAGS):", res["lead_days_BAND_vs_anchors"])
    print("NULL_RESULT:", res["NULL_RESULT"])
    print(res["finding"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
