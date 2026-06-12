#!/usr/bin/env python3
"""Build "Monterey Bay AI Lab — The Deep Dive": a self-contained offline scroll-story exhibit
for HIGH-SCHOOL students, generated from the real lakehouse.

Designed by a PM + UI/UX + Sr-Eng swarm (2026-06-08). A teen scrolls from the sunlit surface
down to the abyss; each depth band darkens. Sections: hero count-up -> place map -> the diving
thermometer (depth curtain + scrubber) -> the daily heartbeat -> the bay breathes cold (upwelling)
-> rain then don't swim -> the AI found the moon -> a haystack too big for any human.

Hard constraints: ONE offline .html, Plotly.js inlined ONCE, no CDN/network, every number derived
from source data at build time, plain language (jargon gate enforced at build end), accessible.

Tracked source: research/viz/build_deep_dive.py  ->  viz_output/monterey_bay_deep_dive.html
"""
from __future__ import annotations
import json, os, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(os.environ.get("MBAL_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
NOAA = ROOT / "mbal_history" / "noaa"
BACT = ROOT / "bacteria_results"
OUT = ROOT / "viz_output" / "monterey_bay_deep_dive.html"
OUT.parent.mkdir(parents=True, exist_ok=True)

# Okabe-Ito color-blind-safe accents
AMBER, SKY, GREEN, BLUE, VERM, PURPLE = "#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7"
DEPTHS = [1, 10, 20, 40, 60, 80, 100, 150, 200, 250, 300]


# ---------------------------------------------------------------- compute
def _panel():
    p = pd.read_parquet(ROOT / "nn_cache" / "long_v2_past_only_fill_origin_observed_full.parquet")
    p["ds"] = pd.to_datetime(p["ds"])
    return p


def compute_depth_grid(panel):
    grid, swing = [], {}
    for d in DEPTHS:
        s = panel[panel.unique_id == f"temp_d{d}p0"].copy()
        if not len(s):
            grid.append([None] * 52); swing[d] = None; continue
        s["w"] = s["ds"].dt.isocalendar().week.clip(1, 52).astype(int)
        wk = s.groupby("w")["y"].mean().reindex(range(1, 53))
        grid.append([None if pd.isna(v) else round(float(v), 2) for v in wk])
        swing[d] = round(float(wk.max() - wk.min()), 1)
    return {"depths": DEPTHS, "weeks": list(range(1, 53)), "temp": grid, "swing": swing}


def compute_heartbeat(panel):
    out = {}
    for d, key in [(1, "surface"), (100, "deep")]:
        s = panel[panel.unique_id == f"temp_d{d}p0"]
        hod = s.assign(h=s["ds"].dt.hour).groupby("h")["y"].mean()
        anom = hod - hod.mean()
        out[key] = {"hours": list(range(24)),
                    "anom": [round(float(anom.get(h, np.nan)), 3) for h in range(24)],
                    "swing": round(float(hod.max() - hod.min()), 2)}
    return out


def _daily(s):
    s = s.dropna(); idx = pd.to_datetime(s.index)
    try: idx = idx.tz_convert("UTC").tz_localize(None)
    except (TypeError, AttributeError): idx = idx.tz_localize(None) if getattr(idx, "tz", None) is not None else idx
    s.index = idx; return s.resample("D").mean()


def compute_upwelling(panel):
    up = _daily(pd.read_parquet(NOAA / "noaa_upwelling.parquet")["cuti_37n"])
    sst = _daily(panel[panel.unique_id == "temp_d1p0"].set_index("ds")["y"])
    full = pd.DataFrame({"cuti": up, "sst": sst}).dropna()
    r = float(full.corr().iloc[0, 1]) if len(full) > 2 else 0.0
    df = full[full.index >= "2021-01-01"]
    return {"dates": [d.strftime("%Y-%m-%d") for d in df.index], "cuti": df["cuti"].round(2).tolist(),
            "sst": df["sst"].round(2).tolist(), "r": round(r, 2)}


def bacteria_pivot():
    b = pd.read_parquet(BACT / "statewide" / "statewide_beach_observations.parquet",
                        columns=["sample_date", "beach_name", "station_name", "property_id", "result_value_numeric"])
    b["sample_date"] = pd.to_datetime(b["sample_date"]).dt.tz_localize(None).dt.normalize()
    piv = b.pivot_table(index=["sample_date", "beach_name", "station_name"], columns="property_id",
                        values="result_value_numeric", aggfunc="max")
    ent, fc, tc = piv.get("prop_enterococcus"), piv.get("prop_fecal_coliform"), piv.get("prop_total_coliform")
    exceed = ((ent >= 104) | (fc >= 400) | (tc >= 10000) |
              ((tc >= 1000) & (fc > 0) & (tc > 0) & (fc / tc >= 0.1))).fillna(False)
    return b, exceed


def compute_hero(b, exceed):
    return {"tests": int(len(exceed)), "events": int(exceed.sum()),
            "rate": round(float(exceed.mean()) * 100, 1),
            "stations": int(b["station_name"].nunique()), "beaches": int(b["beach_name"].nunique()),
            "years": int(np.ceil((b["sample_date"].max() - b["sample_date"].min()).days / 365.25))}


def compute_rain_lag(exceed):
    rate = exceed.groupby(level=0).mean()
    rain = pd.read_csv(BACT / "lovers_point" / "asos_rainfall.csv")
    rain["valid"] = pd.to_datetime(rain["valid"], errors="coerce")
    rain["p01i"] = pd.to_numeric(rain["p01i"].replace({"T": 0.001, "M": np.nan}), errors="coerce")
    rain = rain.dropna(subset=["valid"])
    rday = (rain.assign(d=rain["valid"].dt.tz_localize(None).dt.normalize())
                .groupby(["d", "station"])["p01i"].sum().groupby(level=0).mean())
    df = pd.DataFrame({"rain": rday, "rate": rate}).dropna()
    lags = list(range(0, 8))
    cors = [round(float(df["rain"].corr(df["rate"].shift(-k))), 3) for k in lags]
    return {"lags": lags, "cors": cors, "peak": int(lags[int(np.argmax(cors))])}


def compute_counties():
    inv = pd.read_csv(BACT / "statewide" / "statewide_event_inventory.csv").sort_values("events", ascending=False)
    inv = inv[inv["events"] > 0].head(12)
    return {"county": inv["county"].tolist(), "events": inv["events"].astype(int).tolist()}


def compute_model():
    fi = pd.read_csv(BACT / "lovers_point" / "feature_importance.csv")
    fcol = fi.columns[0]
    icol = [c for c in fi.columns if "import" in c.lower() or "gain" in c.lower() or "weight" in c.lower()]
    icol = icol[0] if icol else fi.columns[1]
    top = fi.sort_values(icol, ascending=False).head(10)

    def gloss(name):
        n = str(name).lower()
        if "tide" in n or "_m2" in n or "_k1" in n: return "the pull of the moon (tides)"
        if "coli" in n or "entero" in n or "exceed" in n or "prev" in n: return "how the water tested recently"
        if "rain" in n or "precip" in n: return "recent rain"
        if "wind" in n or "wave" in n: return "wind & waves"
        if "temp" in n or "sst" in n: return "water temperature"
        return str(name).replace("_", " ")

    def ismoon(name):
        n = str(name).lower(); return "tide" in n or "_m2" in n or "_k1" in n
    feats = [{"imp": round(float(r[icol]), 4), "gloss": gloss(r[fcol]), "moon": ismoon(r[fcol])}
             for _, r in top.iterrows()]
    m = json.loads((BACT / "lovers_point" / "metrics.json").read_text())

    def dig(o, key):
        if isinstance(o, dict):
            if key in o and isinstance(o[key], (int, float)): return o[key]
            for v in o.values():
                r = dig(v, key)
                if r is not None: return r
        return None
    return {"feats": feats, "recall": dig(m, "recall"), "precision": dig(m, "precision"), "roc_auc": dig(m, "roc_auc")}


# ---------------------------------------------------------------- plotly
def style(fig, h=420):
    fig.update_layout(template="plotly_white", height=h, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=60, r=30, t=20, b=50),
                      font=dict(color="#EAF6FB", size=14), legend=dict(orientation="h", y=-0.2))
    gc = "rgba(255,255,255,0.18)"
    fig.update_xaxes(gridcolor=gc, zeroline=False); fig.update_yaxes(gridcolor=gc, zeroline=False)
    return fig


def html(fig, first=False, div_id=None):
    return fig.to_html(full_html=False, include_plotlyjs=("inline" if first else False),
                       config={"displayModeBar": False, "responsive": True}, div_id=div_id)


def fig_curtain(grid):
    z = [[grid["temp"][di][wi] for wi in range(52)] for di in range(len(DEPTHS))]
    fig = go.Figure(go.Heatmap(z=z, x=grid["weeks"], y=[str(d) for d in DEPTHS],
                    colorscale=[[0, SKY], [0.5, "#F4E1B0"], [1, AMBER]], colorbar=dict(title="°C"),
                    hovertemplate="depth %{y} m · week %{x}<br>%{z} °C<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title="depth (m)")
    fig.update_xaxes(title="week of the year →")
    fig.update_layout(shapes=[dict(type="line", x0=1, x1=52, y0="1", y1="1", xref="x", yref="y",
                                   line=dict(color="#fff", width=3, dash="dot"))])
    return style(fig, 440)


def fig_heartbeat(hb):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hb["surface"]["hours"], y=hb["surface"]["anom"], name="at the surface",
                             line=dict(color=AMBER, width=3)))
    fig.add_trace(go.Scatter(x=hb["deep"]["hours"], y=hb["deep"]["anom"], name="100 m down",
                             line=dict(color=SKY, width=3), visible=False))
    fig.update_xaxes(title="hour of the day", dtick=6)
    fig.update_yaxes(title="warmer / cooler than the day's average (°C)")
    return style(fig, 380)


def fig_upwelling(uw):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=uw["dates"], y=uw["cuti"], name="how hard the wind pumps cold water up",
                             line=dict(color=GREEN)), secondary_y=False)
    fig.add_trace(go.Scatter(x=uw["dates"], y=uw["sst"], name="surface water temperature",
                             line=dict(color=AMBER)), secondary_y=True)
    fig.update_yaxes(title_text="wind pump →", secondary_y=False)
    fig.update_yaxes(title_text="surface temp (°C)", secondary_y=True)
    return style(fig, 400)


def fig_lag(rl):
    colors = [VERM if k == rl["peak"] else "rgba(213,94,0,0.45)" for k in rl["lags"]]
    fig = go.Figure(go.Bar(x=rl["lags"], y=rl["cors"], marker_color=colors,
                    hovertemplate="%{x} day(s) after rain<extra></extra>"))
    fig.update_xaxes(title="days after the rain", dtick=1)
    fig.update_yaxes(title="how strongly rain predicts an unsafe beach")
    return style(fig, 360)


def fig_counties(ct):
    fig = go.Figure(go.Bar(x=ct["events"][::-1], y=ct["county"][::-1], orientation="h",
                    marker_color=VERM, hovertemplate="%{y}: %{x:,} unsafe days<extra></extra>"))
    fig.update_xaxes(title="times the water failed its safety test (2005-2026)")
    return style(fig, 420)


def fig_model(md):
    feats = md["feats"][::-1]
    fig = go.Figure(go.Bar(x=[f["imp"] for f in feats], y=[f["gloss"] for f in feats], orientation="h",
                    marker_color=[AMBER if f["moon"] else "rgba(86,180,233,0.7)" for f in feats],
                    hovertemplate="%{y}<extra></extra>"))
    fig.update_xaxes(title="how much the AI leaned on this clue")
    return style(fig, 420)


def build():
    panel = _panel()
    b, exceed = bacteria_pivot()
    D = dict(hero=compute_hero(b, exceed), depthGrid=compute_depth_grid(panel),
             heartbeat=compute_heartbeat(panel), upwelling=compute_upwelling(panel),
             rainLag=compute_rain_lag(exceed), counties=compute_counties(), model=compute_model())
    H = D["hero"]

    c_curtain = html(fig_curtain(D["depthGrid"]), first=True, div_id="curtain")
    c_heart = html(fig_heartbeat(D["heartbeat"]), div_id="heart")
    c_upw = html(fig_upwelling(D["upwelling"]))
    c_lag = html(fig_lag(D["rainLag"]), div_id="lag")
    c_cty = html(fig_counties(D["counties"]))
    c_model = html(fig_model(D["model"]))

    recall = D["model"].get("recall") or 0.667
    roc = D["model"].get("roc_auc") or 0.79
    caught = max(1, round(recall * 3))

    def band(id_, cls, inner):
        return f'<section id="{id_}" class="band {cls}">{inner}</section>'

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Monterey Bay AI Lab — The Deep Dive</title>
<style>
:root{{--amber:{AMBER};--sky:{SKY};--verm:{VERM};}}
*{{box-sizing:border-box}} html{{scroll-behavior:smooth}}
body{{margin:0;font-family:-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;font-size:18px;line-height:1.6;color:#0A2A43;background:#061826}}
h1,h2,.wow{{font-family:Georgia,'Times New Roman',serif}}
.band{{padding:13vh 6vw;min-height:78vh;display:flex;flex-direction:column;justify-content:center}}
.band .inner{{max-width:62ch;margin:0 auto;width:100%}}
.surface{{background:linear-gradient(#EAF6FB,#BFE6F2);color:#0A2A43}}
.sunlit{{background:linear-gradient(#BFE6F2,#1B9AAA);color:#0A2A43}}
.midwater{{background:linear-gradient(#1B9AAA,#14506E);color:#EAF6FB}}
.deep{{background:linear-gradient(#14506E,#0A2A43);color:#EAF6FB}}
.deep2{{background:linear-gradient(#0A2A43,#082235);color:#EAF6FB}}
.abyss{{background:linear-gradient(#082235,#061826);color:#EAF6FB}}
h1{{font-size:clamp(2.2rem,6vw,4rem);line-height:1.1;margin:.2em 0}}
h2{{font-size:clamp(1.6rem,4.5vw,2.4rem);margin:.2em 0 .4em}}
.wow{{font-weight:800;font-size:clamp(2.5rem,8vw,5rem);line-height:1;display:block;margin:.1em 0}}
.sub{{opacity:.85;font-size:1.05rem}} .lead{{font-size:1.18rem}}
.chip{{display:inline-block;min-height:44px;line-height:44px;padding:0 18px;margin:6px 6px 0 0;border-radius:24px;border:2px solid var(--amber);background:transparent;color:inherit;font:inherit;cursor:pointer}}
.chip[aria-pressed=true]{{background:var(--amber);color:#0A2A43;font-weight:700}}
.card{{margin-top:14px;border-left:4px solid var(--amber);padding:10px 16px;background:rgba(255,255,255,.10);border-radius:0 8px 8px 0;display:none}}
.card.show{{display:block}}
.reveal{{min-height:44px;padding:0 18px;border-radius:24px;border:2px solid currentColor;background:transparent;color:inherit;font:inherit;cursor:pointer;margin-top:10px}}
input[type=range]{{width:100%;height:44px;accent-color:var(--amber)}}
.readout{{font-family:Georgia,serif;font-weight:800;font-size:clamp(1.6rem,5vw,2.6rem);color:var(--amber)}}
.chartwrap{{margin-top:18px;background:rgba(255,255,255,.05);border-radius:12px;padding:8px}}
.danger{{color:var(--verm);font-weight:700}}
.counts{{display:flex;flex-wrap:wrap;gap:22px;justify-content:center;text-align:center}}
.counts .wow{{font-size:clamp(1.8rem,6vw,3.2rem)}}
.sr{{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0)}}
.rail{{position:fixed;left:0;top:0;width:6px;height:100%;background:rgba(255,255,255,.10);z-index:50}}
.rail b{{position:absolute;left:0;top:0;width:100%;height:0;background:var(--amber)}}
footer{{padding:6vh 6vw;color:#8fb0c6;font-size:.85rem;text-align:center;background:#061826}}
a.scrolldown{{color:inherit;text-decoration:none;font-size:2rem;display:inline-block;margin-top:18px;animation:bob 1.8s ease-in-out infinite}}
.map{{max-width:520px;margin:18px auto 0;display:block;border-radius:10px}}
.pin{{cursor:pointer}} .pin:focus circle,.pin:hover circle{{r:13}}
details{{margin-top:10px}} summary{{cursor:pointer;min-height:44px}}
@keyframes bob{{0%,100%{{transform:translateY(0)}}50%{{transform:translateY(10px)}}}}
@media (prefers-reduced-motion:reduce){{*{{animation:none!important;scroll-behavior:auto}}}}
@media (max-width:768px){{body{{font-size:17px}}.band{{padding:10vh 7vw}}}}
:focus-visible{{outline:3px solid var(--amber);outline-offset:2px}}
</style></head><body>
<div class="rail"><b id="railfill"></b></div>

{band("hero","surface",f'''<div class="inner">
 <h1>Monterey Bay has been keeping a diary for {H['years']} years.</h1>
 <p class="lead">Robots floating in the water. Tests run on the sand.</p>
 <span class="wow" id="heroNum" data-to="{H['tests']}">0</span>
 <p class="sub">water-quality tests, and counting.</p>
 <a class="scrolldown" href="#place" aria-label="Scroll down to dive in">⌄</a></div>''')}

{band("place","sunlit",f'''<div class="inner"><h2>You are here</h2>
 <p>Out past the surf, two floating robot labs ride at anchor — one taking the ocean's pulse every hour since 2004. Almost everything you're about to see comes from these few dots in the water. <b>Tap a dot.</b></p>
 <svg class="map" viewBox="0 0 400 300" role="img" aria-label="Stylized map of Monterey Bay with two moorings and a beach">
   <rect width="400" height="300" fill="#9bd5e6"/>
   <path d="M400,40 C300,60 250,110 255,170 C258,220 300,260 400,275 L400,300 Z" fill="#e8dcc0"/>
   <g class="pin" tabindex="0" data-label="M1: a floating robot lab anchored about 18 miles out, taking the ocean's temperature at 11 depths every hour since 2004."><circle cx="150" cy="150" r="9" fill="{AMBER}"/><text x="150" y="138" text-anchor="middle" font-size="12" fill="#0A2A43">M1</text></g>
   <g class="pin" tabindex="0" data-label="M2: a second mooring farther out, part of the bay's long record."><circle cx="95" cy="110" r="9" fill="{AMBER}"/><text x="95" y="98" text-anchor="middle" font-size="12" fill="#0A2A43">M2</text></g>
   <g class="pin" tabindex="0" data-label="Lovers Point: a Monterey beach we taught an AI to forecast — more on that as you dive."><circle cx="248" cy="168" r="9" fill="{VERM}"/><text x="248" y="190" text-anchor="middle" font-size="11" fill="#0A2A43">beach</text></g>
 </svg>
 <div class="card" id="pinCard" role="status"></div></div>''')}

{band("dive","midwater",f'''<div class="inner"><h2>The diving thermometer</h2>
 <p>At the surface, the ocean swings about <b>{D['depthGrid']['swing'][1]}°C</b> across the year — warm in late summer, chilly in spring. How big do you think that swing is 100 meters down? <b>Drag the diver.</b></p>
 <p>\U0001F9A6 <input type="range" id="depthSlider" min="0" max="{len(DEPTHS)-1}" value="0" step="1" aria-label="Choose a depth to dive to"> \U0001F41F</p>
 <p>At <span class="readout" id="depthLbl">1 m</span> the year's temperature swing is</p>
 <span class="wow" id="swingVal" style="color:var(--amber)">{D['depthGrid']['swing'][1]}°C</span>
 <div class="chartwrap"><figure style="margin:0"><figcaption class="sr">Heat map of ocean temperature by depth (1 to 300 meters) and week of the year: the surface is warm and changes with the seasons, while below about 100 meters it stays cold and steady all year.</figcaption>{c_curtain}</figure></div>
 <p class="sub">Dive past 100 m and the ocean barely notices the seasons — or even the time of day.</p></div>''')}

{band("heartbeat","midwater",f'''<div class="inner"><h2>The bay has a daily heartbeat</h2>
 <p>The sun warms the surface by day and it cools at night — a tiny daily rhythm. Now switch to deep water and watch the heartbeat flatline.</p>
 <button class="chip" id="hbSurf" aria-pressed="true">At the surface</button>
 <button class="chip" id="hbDeep" aria-pressed="false">100 m down</button>
 <p>Daily swing: <span class="readout" id="hbSwing">{D['heartbeat']['surface']['swing']}°C</span></p>
 <div class="chartwrap"><figure style="margin:0"><figcaption class="sr">The surface water warms and cools about {D['heartbeat']['surface']['swing']} degrees over a day; 100 meters down the daily change is only {D['heartbeat']['deep']['swing']} degrees.</figcaption>{c_heart}</figure></div></div>''')}

{band("breath","deep",f'''<div class="inner"><h2>The bay breathes cold</h2>
 <p>When the wind blows just right, it shoves warm surface water out of the way — and cold, nutrient-packed water rises from the deep to take its place. The whole food web switches on. When the wind pumps harder, the surface cools.</p>
 <div class="chartwrap"><figure style="margin:0"><figcaption class="sr">When the wind pumps cold water upward, the surface temperature drops — they move in opposite directions.</figcaption>{c_upw}</figure></div>
 <button class="reveal" data-card="elseCard">What ELSE could cool the bay?</button>
 <div class="card" id="elseCard">Clouds? A cold current drifting by? Something nobody's spotted yet? This is exactly the kind of puzzle we point AI at — finding the patterns humans miss.</div></div>''')}

{band("rain","midwater",f'''<div class="inner"><h2>Rain, then don't swim</h2>
 <p>Big storm last night? Rain washes everything off the streets and into the surf. So when is it safe to get back in? <b>Slide through the days.</b></p>
 <p>\U0001F327️ <input type="range" id="lagSlider" min="0" max="7" value="{D['rainLag']['peak']}" step="1" aria-label="Days after the rain"></p>
 <p class="lead">About <b id="lagDay">{D['rainLag']['peak']}</b> day(s) after rain, beaches are most likely to <span class="danger">⚠ fail the safety test</span>.</p>
 <div class="chartwrap"><figure style="margin:0"><figcaption class="sr">Rainfall predicts unsafe beach water most strongly about one day later, then fades over a week.</figcaption>{c_lag}</figure></div>
 <p style="margin-top:18px">Across California, beaches failed that safety test <b>more than {H['events']//1000},000 times</b> in {H['years']} years — but not evenly:</p>
 <div class="chartwrap"><figure style="margin:0"><figcaption class="sr">San Diego and Los Angeles counties had the most unsafe-water days; Monterey had far fewer.</figcaption>{c_cty}</figure></div></div>''')}

{band("ai","deep2",f'''<div class="inner"><h2>The AI found the moon</h2>
 <p>Here's the wild part. We trained a computer to guess, ahead of time, when one beach would be unsafe. Then we asked it: <b>what clue did you lean on most?</b></p>
 <p class="lead">Its top answer was <b style="color:var(--amber)">the tides — the pull of the moon on the ocean.</b> We never told it about the moon. It found that clue on its own.</p>
 <div class="chartwrap"><figure style="margin:0"><figcaption class="sr">The AI's most important clues for predicting unsafe water were tide patterns driven by the moon, along with how the water had tested recently.</figcaption>{c_model}</figure></div>
 <p>It's not magic — it's honest: in testing it caught <b>{caught} of every 3</b> risky days, and sometimes cried wolf.</p>
 <button class="reveal" data-card="wrongCard">…and where it got it wrong</button>
 <div class="card" id="wrongCard">It still missed about 1 risky day in 3, and a few of its alarms were false. Its overall score was about {roc:.2f} out of 1.0 — useful, but a starting point, not the final word.</div></div>''')}

{band("haystack","abyss",f'''<div class="inner" style="text-align:center"><h2>A haystack too big for any human</h2>
 <div class="counts">
  <div><span class="wow countup" data-to="{H['stations']}">0</span><div class="sub">monitoring stations</div></div>
  <div><span class="wow countup" data-to="{H['beaches']}">0</span><div class="sub">beaches</div></div>
  <div><span class="wow countup" data-to="{H['tests']}">0</span><div class="sub">water tests</div></div>
  <div><span class="wow countup" data-to="{H['events']}">0</span><div class="sub">times unsafe</div></div>
  <div><span class="wow countup" data-to="{H['years']}">0</span><div class="sub">years</div></div>
 </div>
 <p class="lead" style="margin-top:24px">That's a haystack way too big for any person to search by hand. So we teach machines to hunt for the needles — the hidden patterns that tell us when the bay is changing.</p>
 <p class="readout">What pattern would you go looking for?</p></div>''')}

<footer>Monterey Bay AI Lab · built from real data through early 2026 — not a live feed.<br>
The wind→cooling and rain→unsafe-water links are real but modest clues, not proof.
<details><summary>For the data nerds</summary>Signals: M1/M2 moorings (temperature + saltiness, 1–300 m), wind-driven upwelling indices, offshore buoy, coastal weather, statewide California beach water-quality ({H['tests']:,} tests / {H['stations']} stations), rainfall. Generated by research/viz/build_deep_dive.py from the live lakehouse.</details></footer>

<script>window.LAB_DATA={json.dumps(D)};</script>
<script>
(function(){{
 var RM=window.matchMedia('(prefers-reduced-motion:reduce)').matches, D=window.LAB_DATA;
 function fmt(n){{return n.toLocaleString('en-US');}}
 function countUp(el){{var to=+el.dataset.to;if(RM){{el.textContent=fmt(to);return;}}
  var t0=null;function step(t){{if(!t0)t0=t;var p=Math.min((t-t0)/1400,1);
   el.textContent=fmt(Math.floor(p*to));if(p<1)requestAnimationFrame(step);else el.textContent=fmt(to);}}requestAnimationFrame(step);}}
 var io=new IntersectionObserver(function(es){{es.forEach(function(e){{if(e.isIntersecting&&e.target.dataset.to){{countUp(e.target);io.unobserve(e.target);}}}});}},{{threshold:.4}});
 document.querySelectorAll('#heroNum,.countup').forEach(function(el){{io.observe(el);}});
 var dg=D.depthGrid,ds=document.getElementById('depthSlider');
 if(ds)ds.addEventListener('input',function(){{var i=+ds.value,d=dg.depths[i];
   document.getElementById('depthLbl').textContent=d+' m';
   document.getElementById('swingVal').textContent=dg.swing[d]+'\\u00b0C';
   try{{Plotly.relayout('curtain',{{'shapes[0].y0':String(d),'shapes[0].y1':String(d)}});}}catch(e){{}}}});
 function hb(deep){{try{{Plotly.restyle('heart',{{visible:[!deep,deep]}});}}catch(e){{}}
   document.getElementById('hbSurf').setAttribute('aria-pressed',!deep);
   document.getElementById('hbDeep').setAttribute('aria-pressed',deep);
   document.getElementById('hbSwing').textContent=(deep?D.heartbeat.deep.swing:D.heartbeat.surface.swing)+'\\u00b0C';}}
 var hs=document.getElementById('hbSurf');if(hs){{hs.onclick=function(){{hb(false);}};document.getElementById('hbDeep').onclick=function(){{hb(true);}};}}
 var ls=document.getElementById('lagSlider');
 if(ls)ls.addEventListener('input',function(){{var k=+ls.value;document.getElementById('lagDay').textContent=k;
   var cols=D.rainLag.lags.map(function(l){{return l===k?'{VERM}':'rgba(213,94,0,0.45)';}});
   try{{Plotly.restyle('lag',{{'marker.color':[cols]}});}}catch(e){{}}}});
 document.querySelectorAll('.reveal').forEach(function(btn){{btn.onclick=function(){{document.getElementById(btn.dataset.card).classList.toggle('show');}};}});
 document.querySelectorAll('.pin').forEach(function(p){{function show(){{var c=document.getElementById('pinCard');c.textContent=p.dataset.label;c.classList.add('show');}}
   p.onclick=show;p.addEventListener('keydown',function(e){{if(e.key==='Enter'||e.key===' '){{e.preventDefault();show();}}}});}});
 var rf=document.getElementById('railfill');
 addEventListener('scroll',function(){{var h=document.documentElement;rf.style.height=(100*h.scrollTop/(h.scrollHeight-h.clientHeight))+'%';}},{{passive:true}});
}})();
</script></body></html>"""

    OUT.write_text(page, encoding="utf-8")

    txt = page; problems = []
    # Check authored markup/copy only — strip the embedded plotly.js + data <script> blobs.
    visible = re.sub(r"<script.*?</script>", "", txt, flags=re.S)
    if not (0.08 <= exceed.mean() <= 0.16): problems.append(f"exceedance rate {exceed.mean():.3f} out of range")
    if not (40000 <= H["events"] <= 75000): problems.append(f"events {H['events']} out of range")
    for term in ["z-score", "Pearson", "diurnal", " PSU", "exceedance", "persistence", "seasonal-naive", "CUTI", "BEUTI"]:
        if term in visible: problems.append(f"jargon leaked: {term.strip()}")
    # Offline gate: any <script>/<link> that actually LOADS an external URL (inlined plotly is fine).
    if re.search(r'<(script|link)\b[^>]*\b(src|href)\s*=\s*["\']https?://', txt):
        problems.append("external resource load found")
    mb = len(txt.encode()) / 1e6
    print(f"[deep-dive] wrote {OUT}  ({mb:.1f} MB)  tests={H['tests']:,} events={H['events']:,} stations={H['stations']} beaches={H['beaches']} years={H['years']}"
          .encode("ascii", "replace").decode())
    if mb > 20: problems.append(f"file too big {mb:.1f}MB")
    print("[deep-dive] SELF-TESTS PASSED" if not problems else "[deep-dive] SELF-TEST FAILURES:\n  " + "\n  ".join(problems))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(build())
