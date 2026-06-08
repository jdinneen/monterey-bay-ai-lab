#!/usr/bin/env python3
"""
Deep data-intimacy profiling for the MBARI M1 21-year record.

Characterizes the data to inform neural-forecaster choice: coverage/gaps, depth grid,
realtime<->archive boundary, autocorrelation structure, stationarity, diurnal + annual
seasonality (now resolvable over 21 years), cross-depth / temp-sal structure, and the
persistence + climatology reference errors per horizon (the bar any NN must beat).

Reuses the harness's matrix builder so the profile matches exactly what models will see.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

from mbari_forecast_v2 import load_source, build_hourly_matrix
from mbari_core import apply_physical_quality_filters

try:
    from statsmodels.tsa.stattools import adfuller, acf, pacf
    HAVE_SM = True
except Exception:
    HAVE_SM = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARQUET = PROJECT_ROOT / "mbari_history" / "opendap" / "m1_history.parquet"
OUT = PROJECT_ROOT / "mbari_data_profile"
OUT.mkdir(exist_ok=True)
REPORT: dict = {}


def main() -> None:
    raw = load_source("parquet", PARQUET, None)
    print(f"raw rows: {len(raw):,}")
    clean, _ = apply_physical_quality_filters(raw)

    # ---- raw structure ----
    REPORT["raw_rows"] = int(len(raw))
    REPORT["time_min"] = str(raw["time"].min())
    REPORT["time_max"] = str(raw["time"].max())
    if "station" in raw:
        REPORT["stations"] = raw["station"].astype(str).value_counts().to_dict()
    if "source" in raw:
        REPORT["sources"] = raw["source"].astype(str).value_counts().head(10).to_dict()
    depth_counts = raw.assign(depth_m=raw["z"].abs().round(1))["depth_m"].value_counts()
    REPORT["depth_grid"] = {f"{k:.1f}m": int(v) for k, v in depth_counts.head(25).items()}

    # ---- hourly matrix (what the models see) ----
    matrix, coverage = build_hourly_matrix(clean)
    matrix = matrix.sort_index()
    REPORT["matrix_hours"] = int(matrix.shape[0])
    REPORT["matrix_cols"] = int(matrix.shape[1])
    REPORT["matrix_span"] = [str(matrix.index.min()), str(matrix.index.max())]
    coverage.to_csv(OUT / "series_coverage.csv", index=False)

    # ---- gap analysis on hourly grid ----
    full = pd.date_range(matrix.index.min(), matrix.index.max(), freq="1h")
    present = matrix.reindex(full)
    # use the best-covered temp series as the "is the mooring reporting" proxy
    temp_cols = [c for c in matrix.columns if c.startswith("temp_d")]
    proxy = present[temp_cols].notna().any(axis=1) if temp_cols else present.notna().any(axis=1)
    missing = ~proxy
    # largest contiguous gaps
    gid = (missing != missing.shift()).cumsum()
    gaps = missing.groupby(gid).agg(["sum"]).rename(columns={"sum": "len"})
    gap_runs = [int(x) for x in missing.groupby(gid).sum() if x > 0]
    gap_runs_sorted = sorted(gap_runs, reverse=True)[:15]
    REPORT["hourly_grid_len"] = int(len(full))
    REPORT["pct_hours_with_any_temp"] = float(proxy.mean() * 100)
    REPORT["n_gaps"] = int(sum(1 for g in gap_runs if g > 0))
    REPORT["largest_gaps_hours"] = gap_runs_sorted
    REPORT["largest_gap_days"] = round(max(gap_runs_sorted) / 24, 1) if gap_runs_sorted else 0

    # ---- realtime/archive boundary: row density over time ----
    monthly = raw.set_index("time").resample("1MS").size()
    monthly.index = monthly.index.astype(str)
    (OUT / "monthly_rowcount.csv").write_text(monthly.to_csv())

    # ---- per-target seasonality + autocorrelation + stationarity ----
    focus = [c for c in ["temp_d10p0", "temp_d100p0", "sal_d10p0", "sal_d100p0",
                         "air_temperature", "air_pressure"] if c in matrix.columns]
    diag = {}
    for col in focus:
        s = matrix[col].interpolate(limit=6).dropna()
        if len(s) < 2000:
            continue
        d = {"n": int(len(s)), "mean": float(s.mean()), "std": float(s.std())}
        # diurnal + annual amplitude
        by_hour = s.groupby(s.index.hour).mean()
        by_doy = s.groupby(s.index.dayofyear).mean()
        d["diurnal_amp"] = float(by_hour.max() - by_hour.min())
        d["annual_amp"] = float(by_doy.max() - by_doy.min())
        d["annual_to_diurnal_ratio"] = float(d["annual_amp"] / (d["diurnal_amp"] + 1e-9))
        if HAVE_SM:
            ac = acf(s.values, nlags=168, fft=True)
            d["acf_1h"] = float(ac[1]); d["acf_24h"] = float(ac[24]); d["acf_168h"] = float(ac[168])
            # decorrelation time: first lag where acf < 1/e
            below = np.where(ac < np.exp(-1))[0]
            d["decorr_hours"] = int(below[0]) if len(below) else None
            try:
                d["adf_pvalue"] = float(adfuller(s.values[::6], maxlag=48, autolag=None)[1])  # subsample for speed
            except Exception:
                d["adf_pvalue"] = None
        diag[col] = d
    REPORT["series_diagnostics"] = diag

    # ---- cross-depth & temp-sal structure ----
    struct_cols = [c for c in matrix.columns if c.startswith(("temp_d", "sal_d"))]
    corr = matrix[struct_cols].corr()
    corr.to_csv(OUT / "cross_series_correlation.csv")

    # ---- persistence + climatology reference (the bar to beat) ----
    horizons = [1, 6, 24, 72, 168]
    ref_rows = []
    for col in focus:
        s = matrix[col]
        clim_hour = s.groupby(s.index.hour).transform("mean")
        for h in horizons:
            y = s.shift(-h)
            persist = s
            mask = y.notna() & persist.notna()
            if mask.sum() < 500:
                continue
            pe = float(np.sqrt(np.mean((y[mask] - persist[mask]) ** 2)))
            ce = float(np.sqrt(np.mean((y[mask] - clim_hour[mask]) ** 2)))
            ref_rows.append({"series": col, "horizon_h": h, "persistence_rmse": round(pe, 4),
                             "climatology_rmse": round(ce, 4), "n": int(mask.sum())})
    pd.DataFrame(ref_rows).to_csv(OUT / "baseline_reference_rmse.csv", index=False)
    REPORT["baseline_reference"] = ref_rows

    (OUT / "DATA_PROFILE.json").write_text(json.dumps(REPORT, indent=2, default=str))
    _write_md()
    print(f"Done. Profile -> {OUT/'DATA_PROFILE.md'}")


def _write_md() -> None:
    R = REPORT
    L = ["# MBARI M1 Data Intimacy Profile", "",
         f"- Raw rows: {R['raw_rows']:,}  |  span {R['time_min']} -> {R['time_max']}",
         f"- Hourly matrix: {R['matrix_hours']:,} hours x {R['matrix_cols']} cols",
         f"- Hours with any temp reading: {R['pct_hours_with_any_temp']:.1f}%  |  "
         f"{R['n_gaps']} gaps, largest = {R['largest_gap_days']} days",
         f"- Largest gaps (hours): {R['largest_gaps_hours'][:8]}", "",
         "## Depth grid (rows per depth)", ""]
    for k, v in list(R["depth_grid"].items())[:15]:
        L.append(f"- {k}: {v:,}")
    L += ["", "## Series diagnostics (interp<=6h)", "",
          "| series | n | std | diurnal_amp | annual_amp | ann/diur | acf_1h | acf_24h | decorr_h | adf_p |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for c, d in R.get("series_diagnostics", {}).items():
        L.append(f"| {c} | {d['n']:,} | {d['std']:.3g} | {d['diurnal_amp']:.3g} | {d['annual_amp']:.3g} | "
                 f"{d.get('annual_to_diurnal_ratio',0):.2f} | {d.get('acf_1h','?')} | {d.get('acf_24h','?')} | "
                 f"{d.get('decorr_hours','?')} | {d.get('adf_pvalue','?')} |")
    L += ["", "## Baseline reference RMSE (the bar any NN must beat)", "",
          "| series | h | persistence | climatology | n |", "|---|---|---|---|---|"]
    for r in R.get("baseline_reference", []):
        L.append(f"| {r['series']} | {r['horizon_h']} | {r['persistence_rmse']} | {r['climatology_rmse']} | {r['n']:,} |")
    (OUT / "DATA_PROFILE.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
