#!/usr/bin/env python3
"""Build a self-contained interactive Monterey Bay Signal Explorer (one offline .html).

Education-first: shows every signal the lab has and how they INTERPLAY -- upwelling->SST,
the daily tidal/solar heartbeat (the reason the forecast has a +6h sweet spot), and
rain->runoff->beach-bacteria. Plotly.js is embedded inline so the page opens anywhere with
no server and no internet. Regenerate any time data updates.

Tracked source: research/viz/build_explorer.py  ->  output: viz_output/monterey_bay_signal_explorer.html
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(os.environ.get("MBAL_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
NOAA = ROOT / "mbal_history" / "noaa"
OUT = ROOT / "viz_output" / "monterey_bay_signal_explorer.html"
OUT.parent.mkdir(parents=True, exist_ok=True)

C = dict(sst="#e4572e", surf_t="#f3a712", sub_t="#8c4843", sal="#2e86ab", cuti="#157f1f",
         beuti="#4c9f70", wave="#3066be", wind="#6f58c9", pres="#777", rain="#1f6feb",
         bact="#c1121f")


def _daily(s: pd.Series) -> pd.Series:
    """tz-aware/naive datetime-indexed series -> tz-naive daily-mean."""
    s = s.dropna()
    idx = pd.to_datetime(s.index)
    try:
        idx = idx.tz_convert("UTC").tz_localize(None)
    except (TypeError, AttributeError):
        idx = idx.tz_localize(None) if getattr(idx, "tz", None) is not None else idx
    s.index = idx
    return s.resample("D").mean()


def load_daily() -> pd.DataFrame:
    cols = {}
    # mooring panel (hourly) -> a few representative series
    panel = pd.read_parquet(ROOT / "nn_cache" / "long_v2_past_only_fill_origin_observed_full.parquet")
    panel["ds"] = pd.to_datetime(panel["ds"])
    for uid, name in [("temp_d1p0", "mooring_surface_temp_C"), ("temp_d100p0", "mooring_100m_temp_C"),
                      ("sal_d1p0", "mooring_surface_salinity_PSU")]:
        sub = panel[panel.unique_id == uid].set_index("ds")["y"]
        if len(sub):
            cols[name] = _daily(sub)
    # upwelling (daily)
    up = pd.read_parquet(NOAA / "noaa_upwelling.parquet")
    cols["upwelling_CUTI_37N"] = _daily(up["cuti_37n"])
    cols["upwelling_BEUTI_37N"] = _daily(up["beuti_37n"])
    # ndbc buoy (hourly)
    nd = pd.read_parquet(NOAA / "noaa_ndbc46042.parquet")
    cols["wave_height_m"] = _daily(nd["ndbc46042_wave_height_m"])
    cols["wind_speed_ms"] = _daily(nd["ndbc46042_wind_speed_ms"])
    # coops (hourly)
    co = pd.read_parquet(NOAA / "noaa_coops.parquet")
    cols["air_pressure_mb"] = _daily(co["coops_air_pressure_mb"])
    # MUR SST (may still be fetching)
    sst_p = NOAA / "noaa_sst.parquet"
    if sst_p.exists():
        sst = pd.read_parquet(sst_p)
        cols["satellite_SST_C"] = _daily(sst[sst.columns[0]])
    # rainfall (ASOS, hourly p01i inches)
    rain = pd.read_csv(ROOT / "bacteria_results" / "lovers_point" / "asos_rainfall.csv")
    rain["valid"] = pd.to_datetime(rain["valid"], errors="coerce")
    rain["p01i"] = pd.to_numeric(rain["p01i"].replace({"T": 0.001, "M": np.nan}), errors="coerce")
    rain = rain.dropna(subset=["valid"])
    rday = (rain.assign(d=rain["valid"].dt.tz_localize(None).dt.normalize())
                .groupby(["d", "station"])["p01i"].sum().groupby(level=0).mean())
    cols["rainfall_in_per_day"] = rday
    # bacteria daily exceedance rate (statewide)
    cols["bacteria_exceed_rate"] = bacteria_daily_rate()
    df = pd.DataFrame(cols).sort_index()
    df = df[(df.index >= "2004-01-01") & (df.index <= pd.Timestamp.today().normalize())]
    return df


def bacteria_daily_rate() -> pd.Series:
    b = pd.read_parquet(ROOT / "bacteria_results" / "statewide" / "statewide_beach_observations.parquet",
                        columns=["sample_date", "beach_name", "station_name", "property_id", "result_value_numeric"])
    b["sample_date"] = pd.to_datetime(b["sample_date"]).dt.tz_localize(None).dt.normalize()
    piv = b.pivot_table(index=["sample_date", "beach_name", "station_name"], columns="property_id",
                        values="result_value_numeric", aggfunc="max")
    ent = piv.get("prop_enterococcus"); fc = piv.get("prop_fecal_coliform"); tc = piv.get("prop_total_coliform")
    exceed = ((ent >= 104) | (fc >= 400) | (tc >= 10000) |
              ((tc >= 1000) & (fc > 0) & (tc > 0) & (fc / tc >= 0.1))).fillna(False)
    return exceed.groupby(level=0).mean()  # statewide fraction of site-days exceeding


def zscore(df):
    return (df - df.mean()) / df.std(ddof=0)


def fig_explorer(df):
    z = zscore(df)
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(x=z.index, y=z[col], name=col, mode="lines",
                                 line=dict(width=1.3), connectgaps=False,
                                 hovertemplate=f"{col}: %{{customdata:.2f}}<extra></extra>",
                                 customdata=df[col]))
    fig.update_layout(
        title="Signal Explorer — every signal, standardized (z-score) so you can compare shapes",
        height=560, hovermode="x unified", template="plotly_white",
        legend=dict(orientation="h", y=-0.25),
        xaxis=dict(rangeslider=dict(visible=True), range=["2020-01-01", str(df.index.max().date())]),
        yaxis_title="standardized value (σ from mean)")
    return fig


def fig_corr(df):
    cm = df.corr(min_periods=90)
    fig = go.Figure(go.Heatmap(z=cm.values, x=cm.columns, y=cm.columns, zmin=-1, zmax=1,
                               colorscale="RdBu", reversescale=True, zmid=0,
                               hovertemplate="%{x}<br>%{y}<br>r = %{z:.2f}<extra></extra>"))
    fig.update_layout(title="Interplay — daily correlation between signals (overlapping days only)",
                      height=620, template="plotly_white", xaxis=dict(tickangle=40))
    return fig


def fig_upwelling(df):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df.index, y=df["upwelling_CUTI_37N"], name="CUTI (upwelling)",
                             line=dict(color=C["cuti"])), secondary_y=False)
    yt = "satellite_SST_C" if "satellite_SST_C" in df else "mooring_surface_temp_C"
    fig.add_trace(go.Scatter(x=df.index, y=df[yt], name=yt, line=dict(color=C["sst"])), secondary_y=True)
    fig.update_layout(title="Story 1 — Upwelling cools the bay", height=420, template="plotly_white",
                      hovermode="x unified", xaxis=dict(range=["2021-01-01", str(df.index.max().date())]))
    fig.update_yaxes(title_text="CUTI (m²/s, upwelling →)", secondary_y=False)
    fig.update_yaxes(title_text="sea-surface temp (°C)", secondary_y=True)
    return fig


def fig_diurnal():
    panel = pd.read_parquet(ROOT / "nn_cache" / "long_v2_past_only_fill_origin_observed_full.parquet")
    panel["ds"] = pd.to_datetime(panel["ds"])
    fig = go.Figure()
    for uid, name, col in [("temp_d1p0", "surface temp (1 m)", C["surf_t"]),
                           ("temp_d10p0", "10 m temp", C["sub_t"])]:
        s = panel[panel.unique_id == uid].copy()
        if not len(s):
            continue
        hod = s.assign(h=s["ds"].dt.hour).groupby("h")["y"].mean()
        fig.add_trace(go.Scatter(x=hod.index, y=hod.values - hod.mean(), name=name, line=dict(color=col)))
    fig.add_vline(x=20, line_dash="dot", line_color="#999",
                  annotation_text="~local solar noon (20:00 UTC)", annotation_position="top")
    fig.update_layout(title="Story 2 — The daily heartbeat (mean cycle by hour of day)", height=400,
                      template="plotly_white", xaxis_title="hour of day (UTC)",
                      yaxis_title="temp anomaly from daily mean (°C)", hovermode="x unified")
    return fig


def fig_rain_bacteria(df):
    sub = df[["rainfall_in_per_day", "bacteria_exceed_rate"]].dropna()
    fig = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38],
                        subplot_titles=("Daily rainfall vs statewide beach-exceedance rate",
                                        "Lag correlation: rain leads bacteria"),
                        specs=[[{"secondary_y": True}, {}]])
    win = sub[sub.index >= "2018-01-01"]
    fig.add_trace(go.Bar(x=win.index, y=win["rainfall_in_per_day"], name="rainfall (in/day)",
                         marker_color=C["rain"], opacity=0.5), row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=win.index, y=win["bacteria_exceed_rate"], name="exceedance rate",
                             line=dict(color=C["bact"], width=1.3)), row=1, col=1, secondary_y=True)
    lags = range(0, 8)
    cors = [sub["rainfall_in_per_day"].corr(sub["bacteria_exceed_rate"].shift(-k)) for k in lags]
    fig.add_trace(go.Bar(x=list(lags), y=cors, name="corr(rain, bacteria@+k)", marker_color=C["bact"]),
                  row=1, col=2)
    fig.update_yaxes(title_text="rain (in/day)", secondary_y=False, row=1, col=1)
    fig.update_yaxes(title_text="exceedance rate", secondary_y=True, row=1, col=1)
    fig.update_xaxes(title_text="days rain leads bacteria", row=1, col=2)
    fig.update_yaxes(title_text="Pearson r", row=1, col=2)
    fig.update_layout(title="Story 3 — Rain → runoff → beach bacteria", height=430,
                      template="plotly_white", showlegend=True)
    return fig


SECTIONS = []


def add(title, html_text, fig):
    SECTIONS.append((title, html_text, fig))


def build():
    df = load_daily()
    n_sig = df.shape[1]
    span = f"{df.index.min().date()} → {df.index.max().date()}"
    add("Signal Explorer",
        "<p>Every signal the lab currently holds, on one timeline and standardized (z-score) so you "
        "can compare <i>shapes</i>, not units. <b>Click a legend entry to hide/show it</b>; drag the "
        "slider to zoom. Hover shows the real value. This is the front door: see what exists and how "
        "it moves before you pick signals for your own question.</p>", fig_explorer(df))
    add("How the signals interplay",
        "<p>Daily correlation between every pair of signals (overlapping days only). Warm = move "
        "together, cool = move oppositely. Look for the physically meaningful blocks: upwelling vs "
        "temperature (negative — upwelling cools), surface vs subsurface temperature, rainfall vs "
        "bacteria. Correlation isn't causation, but it tells you where to look.</p>", fig_corr(df))
    add("Story 1 · Upwelling cools the bay",
        "<p>When equatorward winds drive <b>upwelling</b> (CUTI rises), cold, nutrient-rich water "
        "comes up from depth and sea-surface temperature <b>drops</b>. This is the engine of Monterey "
        "Bay's productivity — and why temperature here isn't just a seasonal sine wave.</p>", fig_upwelling(df))
    add("Story 2 · The daily heartbeat",
        "<p>Averaged over all days, temperature swings on a <b>24-hour cycle</b> driven by solar heating "
        "and tides. This is the pedagogical payoff: it's exactly why our forecasts beat persistence best "
        "at <b>+6 h</b>, and why 'just repeat yesterday's value' (seasonal-naive) is so hard to beat at "
        "+24 h — the signal repeats daily, so a dumb daily baseline is strong.</p>", fig_diurnal())
    add("Story 3 · Rain → runoff → bacteria",
        "<p>Beach bacterial exceedances are <b>runoff-driven</b>: rain washes contamination into the surf. "
        "The right panel shows the correlation of rainfall with the statewide exceedance rate at a lag of "
        "0–7 days — the spike at short lags is the runoff signal a model can learn from (across 800+ CA "
        "sites, ~57k events).</p>", fig_rain_bacteria(df))

    parts = []
    for i, (title, text, fig) in enumerate(SECTIONS):
        body = fig.to_html(full_html=False, include_plotlyjs=("inline" if i == 0 else False),
                           config={"displayModeBar": True, "responsive": True})
        parts.append(f'<section><h2>{title}</h2>{text}<div class="chart">{body}</div></section>')
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Monterey Bay Signal Explorer</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;color:#1a2b3c;background:#f7f9fb}}
 header{{background:linear-gradient(120deg,#0b3d5c,#157f8f);color:#fff;padding:28px 32px}}
 header h1{{margin:0 0 6px;font-size:26px}} header p{{margin:0;opacity:.9;max-width:900px}}
 nav{{position:sticky;top:0;background:#0b3d5c;padding:8px 32px;z-index:10}}
 nav a{{color:#cfe8ef;margin-right:18px;text-decoration:none;font-size:14px}} nav a:hover{{color:#fff}}
 main{{max-width:1120px;margin:0 auto;padding:8px 24px 60px}}
 section{{background:#fff;border:1px solid #e4ebf1;border-radius:10px;padding:20px 24px;margin:22px 0;box-shadow:0 1px 3px rgba(0,0,0,.04)}}
 section h2{{margin:0 0 8px;color:#0b3d5c}} section p{{max-width:920px;line-height:1.5;color:#33485c}}
 .chart{{margin-top:10px}} footer{{color:#7b8a99;font-size:12px;padding:20px 32px}}
</style></head><body>
<header><h1>🌊 Monterey Bay Signal Explorer</h1>
<p>An interactive, offline view of the lab's coastal data — physical ocean, weather, satellite, and
beach water-quality signals — and how they interplay. {n_sig} signals · {span}.</p></header>
<nav>{''.join(f'<a href="#s{i}">{t.split("·")[-1].strip()}</a>' for i,(t,_,_) in enumerate(SECTIONS))}</nav>
<main>{''.join(p.replace("<section>", f'<section id="s{i}">') for i,p in enumerate(parts))}</main>
<footer>Generated by research/viz/build_explorer.py from the live lakehouse. Standardized overlays are
for shape comparison; story panels use real units. Correlation ≠ causation.</footer>
</body></html>"""
    OUT.write_text(page, encoding="utf-8")
    print(f"[viz] wrote {OUT}  ({OUT.stat().st_size/1e6:.1f} MB, {n_sig} signals, {span})"
          .encode("ascii", "replace").decode())


if __name__ == "__main__":
    build()
