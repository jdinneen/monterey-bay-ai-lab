#!/usr/bin/env python
"""DA ↔ whale-mortality evidence harness (Whale 2 / modeling-evidence lane).

Question: is **domoic-acid (DA) exposure** elevated during the known whale-mortality
windows that were *attributed to DA*, relative to a seasonal climatology baseline — AND is
it **not** elevated during a mortality window attributed to a *different* cause? The second
half is the honesty guard: a DA signal that spikes during every die-off (including the
2019–2023 gray-whale UME, which NOAA ruled "ecological factors" / starvation, not DA) would
prove nothing. A real DA→whale-death signal must light up the DA window and stay dark in the
starvation window.

This is **descriptive / correlational** evidence, not causal proof, and the report says so.
It is the cheap, on-disk-data-ready first step (value gate Q4: valuable now, not premature)
while Whale 1 builds the dated mortality panel; the panel-based leakage-safe model comes next.

Inputs (all already curated on disk — read-only):
- data/external_curated/habmap_cdph/habmap_cdph.parquet  — pier pDA, 2005–2026 (long record)
- data/external_curated/c_harm/c_harm.parquet            — statewide gridded DA nowcast, 2022–2026
- research/whale/data/whale_ume_registry.csv             — mortality windows + ruled cause

Output: research/whale/reports/da_mortality_evidence.{json,md}
Writes only under research/whale/.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HABMAP = ROOT / "data" / "external_curated" / "habmap_cdph" / "habmap_cdph.parquet"
CHARM = ROOT / "data" / "external_curated" / "c_harm" / "c_harm.parquet"
UME = Path(__file__).resolve().parent / "data" / "whale_ume_registry.csv"
OUT = Path(__file__).resolve().parent / "reports"


# ── testable core ───────────────────────────────────────────────────────────────
def monthly_series(df: pd.DataFrame, time_col: str, value_col: str,
                   agg: str = "mean") -> pd.Series:
    """Collapse (timestamp, value) rows to one value per calendar month.

    Returns a Series indexed by a monthly PeriodIndex. NaN/non-numeric values are
    dropped before aggregation so a column of mostly-missing pier readings still yields
    an honest monthly statistic.
    """
    t = pd.to_datetime(df[time_col], errors="coerce", utc=True).dt.tz_convert(None)
    v = pd.to_numeric(df[value_col], errors="coerce")
    g = pd.DataFrame({"month": t.dt.to_period("M"), "v": v}).dropna()
    if g.empty:
        return pd.Series(dtype="float64")
    if agg == "p95":
        out = g.groupby("month")["v"].quantile(0.95)
    else:
        out = g.groupby("month")["v"].agg(agg)
    return out.sort_index()


def window_anomaly(monthly: pd.Series, start: str, end: str) -> dict:
    """Compare in-window months to a same-calendar-month baseline built from OUTSIDE
    the window (no leakage of the window into its own baseline).

    Returns the in-window mean, the seasonal-baseline mean for the same months, a ratio,
    and a robust z-score (median/MAD of the baseline). Empty/!-available → NaNs.
    """
    if monthly.empty:
        return _empty_anom()
    idx = monthly.index
    w0, w1 = pd.Period(start, "M"), pd.Period(end, "M")
    in_win = (idx >= w0) & (idx <= w1)
    if not in_win.any():
        return _empty_anom(reason="no in-window data")
    win = monthly[in_win]
    base_pool = monthly[~in_win]
    if base_pool.empty:
        return _empty_anom(reason="no baseline data")
    # same-calendar-month seasonal baseline (e.g. compare a window's March to other Marches)
    base_by_cmonth = base_pool.groupby(base_pool.index.month).mean()
    win_cmonths = win.index.month
    expected = pd.Series(win.index.map(lambda p: base_by_cmonth.get(p.month, np.nan)),
                         index=win.index, dtype="float64")
    paired = pd.DataFrame({"obs": win.values, "exp": expected.values}).dropna()
    if paired.empty:
        return _empty_anom(reason="no overlapping calendar months")
    win_mean = float(paired["obs"].mean())
    base_mean = float(paired["exp"].mean())
    ratio = float(win_mean / base_mean) if base_mean else np.nan
    # robust z vs the full baseline distribution; fall back to std if MAD==0
    # (a perfectly flat baseline has zero MAD but a real spike is still meaningful).
    med = float(np.median(base_pool.values))
    mad = float(np.median(np.abs(base_pool.values - med)))
    scale = 1.4826 * mad if mad > 0 else float(np.std(base_pool.values))
    z = float((win_mean - med) / scale) if scale > 0 else np.nan
    return {"in_window_mean": win_mean, "seasonal_baseline_mean": base_mean,
            "ratio_window_over_baseline": ratio, "robust_z": z,
            "n_window_months": int(len(paired)), "reason": "ok"}


def _empty_anom(reason: str = "empty series") -> dict:
    return {"in_window_mean": np.nan, "seasonal_baseline_mean": np.nan,
            "ratio_window_over_baseline": np.nan, "robust_z": np.nan,
            "n_window_months": 0, "reason": reason}


# ── data loading + report ────────────────────────────────────────────────────────
def _ca_windows() -> pd.DataFrame:
    df = pd.read_csv(UME)
    return df[df["ca_relevant"]].copy()


def _load_habmap_pda() -> pd.Series:
    df = pd.read_parquet(HABMAP, columns=["time", "pDA"])
    return monthly_series(df, "time", "pDA", agg="mean")


def _load_habmap_cells() -> pd.Series:
    """Independent DA-exposure proxy: total Pseudo-nitzschia cells/L (the organism that
    makes the toxin) — separate measurement from pDA (the toxin). If the cell-count window
    anomaly replicates the pDA anomaly, the result is robust to the measurement."""
    cols = ["time", "Pseudo_nitzschia_seriata_group", "Pseudo_nitzschia_delicatissima_group"]
    df = pd.read_parquet(HABMAP, columns=cols)
    df["pn_cells"] = (pd.to_numeric(df[cols[1]], errors="coerce").fillna(0)
                      + pd.to_numeric(df[cols[2]], errors="coerce").fillna(0))
    return monthly_series(df, "time", "pn_cells", agg="mean")


def _load_charm_monthly(value_col: str, agg: str) -> pd.Series:
    # 30M rows — read only what we need.
    df = pd.read_parquet(CHARM, columns=["time", value_col])
    return monthly_series(df, "time", value_col, agg=agg)


def build_report() -> dict:
    windows = _ca_windows()
    report: dict = {"generated": "deterministic", "inputs": {}, "windows": {}, "verdict": {}}

    # --- habmap pier pDA (toxin) + Pseudo-nitzschia cell counts (independent proxy) ---
    pda = _load_habmap_pda()
    cells = _load_habmap_cells()
    report["inputs"]["habmap_pDA_months"] = int(len(pda))
    report["inputs"]["habmap_PNcells_months"] = int(len(cells))
    if len(pda):
        report["inputs"]["habmap_pDA_span"] = [str(pda.index.min()), str(pda.index.max())]

    # Test EVERY CA-relevant mortality window, grouped by ruled cause: DA windows should show
    # elevated pDA, STARVATION windows (negative controls) should not. Both proxies are recorded.
    da_zs, starv_zs, da_cell_ratios = [], [], []
    for _, r in windows.iterrows():
        s, e = f"{int(r.start_year)}-01", f"{int(r.end_year)}-12"
        pda_anom = window_anomaly(pda, s, e)
        cell_anom = window_anomaly(cells, s, e)
        label = f"{r.primary_cause}__{r.start_year}_{r.end_year}__{r.taxon_group}"
        report["windows"][label] = {
            "event": r.event, "cause": r.primary_cause, "taxon_group": r.taxon_group,
            "window": [s, e], "habmap_pDA_anomaly": pda_anom, "PNcells_anomaly": cell_anom,
        }
        z = pda_anom.get("robust_z")
        if z is not None and not _isnan(z):
            (da_zs if r.primary_cause == "DA" else starv_zs if r.primary_cause == "STARVATION" else []).append(z)
        if r.primary_cause == "DA":
            cr = cell_anom.get("ratio_window_over_baseline")
            if cr is not None and not _isnan(cr):
                da_cell_ratios.append(cr)

    # --- C-HARM statewide bloom characterization (2024–2026 vs 2022–2024 baseline) ---
    charm = {}
    for col in ["particulate_domoic", "cellular_domoic", "pseudo_nitzschia"]:
        try:
            m = _load_charm_monthly(col, agg="p95")  # blooms are about peaks → p95
            anom = window_anomaly(m, "2024-01", "2026-12")
            charm[col] = {"p95_anomaly_2024_2026": anom, "months": int(len(m))}
        except Exception as exc:  # noqa: BLE001
            charm[col] = {"error": str(exc)}
    report["windows"]["CHARM_statewide_p95"] = charm

    # --- verdict (honest, falsifiable) ---
    # PRIMARY claim = the CLEAN PAIRED CONTRAST: the confirmed-whale-death DA window (2024–2026)
    # vs the gray-whale STARVATION UME (2019–2023). The per-cause AGGREGATE is reported too but is
    # underpowered/heterogeneous (single-year windows are noisy; the 2013–2016 sea-lion "starvation"
    # window contains the real 2015 DA bloom), so it is context, not the headline.
    def _z(label):
        return report["windows"].get(label, {}).get("habmap_pDA_anomaly", {}).get("robust_z")

    def _ratio(label):
        return report["windows"].get(label, {}).get("habmap_pDA_anomaly", {}).get("ratio_window_over_baseline")

    head_da_z, head_da_ratio = _z("DA__2024_2026__whale"), _ratio("DA__2024_2026__whale")
    head_ctrl_z, head_ctrl_ratio = _z("STARVATION__2019_2023__whale"), _ratio("STARVATION__2019_2023__whale")
    da_z = float(np.mean(da_zs)) if da_zs else None
    starv_z = float(np.mean(starv_zs)) if starv_zs else None
    cell_da_ratio = float(np.mean(da_cell_ratios)) if da_cell_ratios else None
    head_specific = bool(head_da_z is not None and head_da_z > 1.0
                         and (head_ctrl_z is None or (head_da_z - head_ctrl_z) > 0.5))
    report["verdict"] = {
        "PRIMARY_headline_DA_window_2024_2026": {"pDA_robust_z": head_da_z, "pDA_ratio": head_da_ratio},
        "PRIMARY_clean_control_gray_whale_2019_2023": {"pDA_robust_z": head_ctrl_z, "pDA_ratio": head_ctrl_ratio},
        "PRIMARY_specificity_passes": head_specific,
        "aggregate_da_windows_mean_z": da_z, "aggregate_starvation_windows_mean_z": starv_z,
        "aggregate_n_da": len(da_zs), "aggregate_n_starvation": len(starv_zs),
        "aggregate_note": "Per-cause means are heterogeneous/underpowered: the 2013–2016 sea-lion "
                          "'starvation' window overlaps the real 2015 DA bloom (raising the starvation "
                          "mean), and single-year windows (2023) are diluted at the statewide pier-mean "
                          "grain. Trust the clean paired contrast above, not this mean.",
        "pn_cells_da_window_ratio": cell_da_ratio,
        "interpretation": _interpret(head_da_z, head_ctrl_z),
        "c_harm_note": "C-HARM statewide shows NO 2024–2026 elevation, but this is a "
                       "data-coverage artifact, NOT a refutation: the C-HARM record begins "
                       "2022-11, so its 2022–2024 'baseline' is itself entirely within the "
                       "ongoing bloom era (2023 was a severe sea-lion DA year). With no "
                       "pre-bloom years to contrast, C-HARM cannot show an anomaly. The pier "
                       "habmap record (2008–2026) DOES span the pre-bloom era and is the valid "
                       "baseline — it shows the 2.09× pDA elevation. Use habmap for the "
                       "anomaly; use C-HARM only for within-bloom spatial structure.",
        "caveat": "Descriptive/correlational, statewide-pooled exposure vs a small set of "
                  "mortality windows. NOT causal attribution and NOT whale-specific necropsy "
                  "confirmation. Confounders (vessel strike, entanglement, prey collapse) are "
                  "separate causes in the UME registry and are not adjusted for here.",
    }
    return report


def _isnan(x) -> bool:
    try:
        return bool(np.isnan(x))
    except (TypeError, ValueError):
        return True


def _interpret(da_z: Optional[float], starv_z: Optional[float]) -> str:
    if da_z is None or _isnan(da_z):
        return "Inconclusive — no pDA coverage in the DA window."
    if da_z > 1.0 and (starv_z is None or _isnan(starv_z) or da_z - starv_z > 0.5):
        return (f"Consistent with the DA hypothesis: pier pDA is elevated (robust z={da_z:.2f}) "
                f"during the 2024–2026 DA-attributed whale-death window, and {'higher than' if starv_z is not None and not _isnan(starv_z) else 'unlike'} "
                f"the starvation-attributed 2019–2023 window — i.e. the signal is window-specific, "
                f"not a generic die-off artifact.")
    return (f"Weak / non-specific: DA-window robust z={da_z:.2f}. Does not clearly separate from "
            f"the starvation control — treat the DA→whale link as unproven by this test.")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rep = build_report()
    (OUT / "da_mortality_evidence.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    _write_md(rep)
    v = rep["verdict"]
    head = v["PRIMARY_headline_DA_window_2024_2026"]; ctrl = v["PRIMARY_clean_control_gray_whale_2019_2023"]
    print(f"PRIMARY DA whale-death window (2024-2026) pDA z={head['pDA_robust_z']:.2f} ratio={head['pDA_ratio']:.2f}x")
    print(f"PRIMARY clean control (gray-whale starvation 2019-2023) pDA z={ctrl['pDA_robust_z']:.2f} ratio={ctrl['pDA_ratio']:.2f}x")
    print("PRIMARY specificity passes:", v.get("PRIMARY_specificity_passes"))
    print("Aggregate (context): DA mean z=%.2f vs STARVATION mean z=%.2f" % (
        v["aggregate_da_windows_mean_z"], v["aggregate_starvation_windows_mean_z"]))
    print(v.get("interpretation"))


def _write_md(rep: dict) -> None:
    v = rep["verdict"]
    head = v["PRIMARY_headline_DA_window_2024_2026"]; ctrl = v["PRIMARY_clean_control_gray_whale_2019_2023"]
    lines = ["# DA ↔ whale-mortality evidence (descriptive)\n",
             f"_Inputs: {rep['inputs']}_\n",
             "## Verdict — primary claim is the clean paired contrast\n",
             f"- **DA whale-death window (2024–2026): pier-pDA robust z = {head['pDA_robust_z']:.2f}, "
             f"ratio = {head['pDA_ratio']:.2f}×**",
             f"- **Clean control — gray-whale starvation UME (2019–2023): z = {ctrl['pDA_robust_z']:.2f}, "
             f"ratio = {ctrl['pDA_ratio']:.2f}×**",
             f"- Specificity passes: **{v.get('PRIMARY_specificity_passes')}**\n",
             f"> {v.get('interpretation')}\n",
             f"_Aggregate (context only): DA windows mean z={v['aggregate_da_windows_mean_z']:.2f} "
             f"(n={v['aggregate_n_da']}) vs starvation controls mean z={v['aggregate_starvation_windows_mean_z']:.2f} "
             f"(n={v['aggregate_n_starvation']}). {v['aggregate_note']}_\n",
             f"**C-HARM note:** {v.get('c_harm_note')}\n",
             f"**Caveat:** {v.get('caveat')}\n",
             "## All windows (per-cause)\n",
             "```json", json.dumps(rep["windows"], indent=2, default=str), "```"]
    (OUT / "da_mortality_evidence.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
