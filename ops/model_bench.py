#!/usr/bin/env python3
"""Comprehensive, REPEATABLE model bench for the Monterey Bay AI Lab.

Runs EVERY available model family on the bacteria-exceedance task across the full
driver set (every ingested source the leakage-safe seam can causally join), plus the
marine-HAB / domoic-acid task — then aggregates ONE honest leaderboard with per-stratum
metrics, baseline verdicts, and a learnings log. It is a *bench*, not a promotion:
producing a number here does not promote anything (promotion stays gated by release_gate/).

It LEVERAGES existing infrastructure — it does not reimplement models or metrics:
  * `research.bacteria.operational_benchmark.run(clf=..., <driver dirs>, label=...)`
    — the leakage-safe train(<=2019)/calibrate(2020-21)/test(>=2022) seam that already
    computes AP / ROC-AUC / ECE per region stratum + the operational verdict + baselines.
  * `research.model_lab.model_suite._bacteria_candidates()` — the production tree-model
    factories (HGBT / XGBoost / LightGBM / CatBoost).
  * `research/hab/da_forecast.py` — the HAB next-visit pDA-exceedance eval.

Driver sets:
  rain  = rainfall only (the documented baseline driver)
  all   = rainfall + first-flush discharge + waves + tide + oceanography (SST/HF-radar)
          — every driver group the seam can join, from the trusted + curated sources.

    python -m ops.model_bench                      # bacteria (all models x rain/all) + HAB
    python -m ops.model_bench --skip-completed     # resume (per-cell cache)
    python -m ops.model_bench --models hgbt,xgboost --drivers all
    python -m ops.model_bench --no-hab
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

OUT = _REPO / "reports" / "model_bench"
CELLS = OUT / "cells"
LEARNINGS = OUT / "bench_learnings.jsonl"
HEADLINE_STRATUM = "EXCLUDE_SAN_DIEGO"   # the honest, San-Diego-excluded headline stratum

# Driver directories the seam can causally join (trusted + curated sources).
DRIVER_DIRS = {
    "rain_dir": "bacteria_results/rainfall",
    "discharge_dir": "bacteria_results/discharge",
    "cdip_dir": "bacteria_results/cdip_waves",
    "tide_stages_dir": "bacteria_results/tide_stages",
    "ocean_dir": "data/external_curated",   # holds mur_sst/ + hf_radar/ subdirs
    "air_dir": "data/external_curated",     # holds open_meteo_archive/ (gated air-temp KEEP)
}
DRIVER_SETS = {
    "rain": ["rain_dir"],
    "all": ["rain_dir", "discharge_dir", "cdip_dir", "tide_stages_dir", "ocean_dir", "air_dir"],
}


# ---------------------------------------------------------------- model zoo
def _extra_factories() -> dict:
    """Model families beyond the registered tree candidates, for breadth. NaN-unsafe
    sklearn models are wrapped in a median-impute pipeline so the seam can fit them."""
    out: dict[str, callable] = {}

    def _rf():
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                                    n_jobs=-1, random_state=42))

    def _et():
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        return make_pipeline(SimpleImputer(strategy="median"),
                             ExtraTreesClassifier(n_estimators=500, class_weight="balanced",
                                                  n_jobs=-1, random_state=42))

    def _logit():
        from sklearn.linear_model import LogisticRegression
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(max_iter=2000, class_weight="balanced"))

    out["random_forest"] = _rf
    out["extra_trees"] = _et
    out["logistic"] = _logit
    return out


def model_zoo() -> dict:
    """name -> (factory|None, status). Registered tree families + extra sklearn families."""
    from research.model_lab.model_suite import _bacteria_candidates
    zoo: dict[str, tuple] = {}
    name_map = {"bacteria_hgbt_isotonic": "hgbt", "bacteria_xgboost": "xgboost",
                "bacteria_lightgbm": "lightgbm", "bacteria_catboost": "catboost"}
    for mid, (factory, state) in _bacteria_candidates().items():
        zoo[name_map.get(mid, mid)] = (factory, state)
    for name, factory in _extra_factories().items():
        zoo[name] = (factory, "ok")
    return zoo


# ---------------------------------------------------------------- one cell
def _extract(res: dict, stratum: str) -> dict:
    s = (res.get("strata", {}) or {}).get(stratum, {}) or {}
    models = s.get("models", {}) or {}
    cal = models.get("model_hgbt_calibrated", {}) or {}
    raw = models.get("model_hgbt", {}) or {}
    verdict = s.get("operational_verdict", {}) or {}
    base = {k: v for k, v in models.items() if k.startswith("baseline")}
    best_base_ap = max((b.get("ap") or 0.0) for b in base.values()) if base else None
    return {
        "ap_calibrated": cal.get("ap"), "ap_raw": raw.get("ap"),
        "roc_auc": cal.get("roc_auc") or raw.get("roc_auc"),
        "ece": cal.get("ece") or cal.get("ece_calibrated"),
        "n_test": s.get("n_test") or s.get("test_rows"),
        "events_test": s.get("events_test"),
        "best_baseline_ap": best_base_ap,
        "beats_ab411": verdict.get("model_beats_ab411"),
        "beats_vb_mlr": verdict.get("model_beats_vb_mlr"),
        "beats_operational_ranking": verdict.get("model_beats_operational_ranking"),
        "deploy_ready": verdict.get("calibrated_deploy_ready"),
    }


def run_cell(model: str, driverset: str, *, label: str, reveal_lag: int,
             skip_completed: bool) -> dict:
    from research.bacteria import operational_benchmark as ob
    cell_path = CELLS / f"{model}__{driverset}.json"
    if skip_completed and cell_path.exists():
        prev = json.loads(cell_path.read_text(encoding="utf-8"))
        if prev.get("status") == "ok":
            return {**prev, "note": "cached"}

    zoo = model_zoo()
    factory, state = zoo.get(model, (None, "unknown_model"))
    rec = {"model": model, "drivers": driverset, "label": label, "status": state}
    if factory is None:
        CELLS.mkdir(parents=True, exist_ok=True)
        cell_path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        return rec

    dirs = {k: str(_REPO / DRIVER_DIRS[k]) for k in DRIVER_SETS[driverset]
            if (_REPO / DRIVER_DIRS[k]).exists()}
    obs = ob.default_obs_path()
    t0 = time.monotonic()
    try:
        res = ob.run(obs, clf=factory(), reveal_lag_days=reveal_lag, label=label, **dirs)
        metrics = {st: _extract(res, st) for st in
                   ("EXCLUDE_SAN_DIEGO", "ALL", "SAN_DIEGO_ONLY", "MONTEREY")}
        rec.update({"status": "ok", "seconds": round(time.monotonic() - t0, 1),
                    "headline": metrics[HEADLINE_STRATUM], "strata": metrics,
                    "drivers_joined": sorted(dirs.keys())})
    except Exception as e:  # one model must never crash the bench
        rec.update({"status": "errored", "error": f"{type(e).__name__}: {e}",
                    "seconds": round(time.monotonic() - t0, 1)})
    CELLS.mkdir(parents=True, exist_ok=True)
    cell_path.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
    return rec


# ---------------------------------------------------------------- HAB task
def run_hab() -> dict:
    script = _REPO / "research" / "hab" / "da_forecast.py"
    if not script.exists():
        return {"status": "missing", "note": "research/hab/da_forecast.py not found"}
    t0 = time.monotonic()
    proc = subprocess.run([sys.executable, str(script)], cwd=str(_REPO),
                          capture_output=True, text=True, timeout=1800)
    out = {"status": "ok" if proc.returncode == 0 else "errored",
           "seconds": round(time.monotonic() - t0, 1), "returncode": proc.returncode}
    res_json = _REPO / "reports" / "hab" / "da_forecast.json"
    if res_json.exists():
        try:
            d = json.loads(res_json.read_text(encoding="utf-8"))
            h = d.get("headline", {}) or {}
            out["headline"] = {
                "ap": h.get("model_ap"), "roc_auc": h.get("model_roc_auc"),
                "beats_baselines": h.get("model_beats_baselines"),
                "best_baseline_ap": h.get("best_baseline_ap"),
                "loso_beats_seasonal": h.get("loso_model_beats_seasonal"),
            }
            # honest note: does adding drivers help or hurt the sparse-event HAB model?
            th = d.get("test_timeheldout", {}) or {}
            base_ap = (th.get("model_DA+precursor", {}) or {}).get("ap")
            drv_ap = (th.get("model_DA+precursor+drivers", {}) or {}).get("ap")
            if base_ap is not None and drv_ap is not None:
                out["driver_effect"] = ("drivers HELP" if drv_ap - base_ap >= 0.005
                                        else "drivers WASH/HURT") + f" ({drv_ap - base_ap:+.4f} AP)"
        except Exception as e:
            out["note"] = f"could not parse da_forecast.json: {e}"
    else:
        out["note"] = (proc.stderr or proc.stdout or "")[-400:]
    return out


# ---------------------------------------------------------------- aggregate
def aggregate(cells: list[dict], hab: dict | None) -> dict:
    ok = [c for c in cells if c.get("status") == "ok"]
    ranked = sorted(ok, key=lambda c: (c.get("headline", {}).get("ap_calibrated") or -1),
                    reverse=True)
    # driver lift: best 'all' AP vs best 'rain' AP, per model
    lift = {}
    for model in sorted({c["model"] for c in ok}):
        by = {c["drivers"]: c.get("headline", {}).get("ap_calibrated") for c in ok
              if c["model"] == model}
        if by.get("rain") is not None and by.get("all") is not None:
            lift[model] = round(by["all"] - by["rain"], 4)
    return {
        "task": "bacteria_exceedance (enterococcus, EXCLUDE_SAN_DIEGO)",
        "n_cells": len(cells), "n_ok": len(ok),
        "errored": [{"model": c["model"], "drivers": c["drivers"], "error": c.get("error")}
                    for c in cells if c.get("status") == "errored"],
        "unavailable": [c["model"] for c in cells if c.get("status") == "unavailable_dependency"],
        "leaderboard": [{
            "model": c["model"], "drivers": c["drivers"],
            **{k: c.get("headline", {}).get(k) for k in
               ("ap_calibrated", "roc_auc", "ece", "best_baseline_ap",
                "beats_ab411", "beats_vb_mlr", "deploy_ready")}} for c in ranked],
        "driver_lift_all_minus_rain": lift,
        "hab": hab,
    }


def render_md(agg: dict) -> str:
    L = ["# Model Bench — comprehensive run (bacteria + HAB)", "",
         f"Task: **{agg['task']}**. Cells: {agg['n_ok']}/{agg['n_cells']} ok. "
         "Bench only — running here does NOT promote (release_gate/ decides that).", "",
         "## Bacteria leaderboard (headline stratum: San-Diego-excluded, calibrated)", "",
         "| model | drivers | AP(cal) | ROC-AUC | ECE | best baseline AP | beats AB411 | beats VB-MLR | deploy-ready |",
         "|---|---|--:|--:|--:|--:|:-:|:-:|:-:|"]
    for r in agg["leaderboard"]:
        def f(x, n=4):
            return f"{x:.{n}f}" if isinstance(x, (int, float)) else "—"
        L.append(f"| `{r['model']}` | {r['drivers']} | {f(r['ap_calibrated'])} | "
                 f"{f(r['roc_auc'])} | {f(r['ece'])} | {f(r['best_baseline_ap'])} | "
                 f"{r['beats_ab411']} | {r['beats_vb_mlr']} | {r['deploy_ready']} |")
    L += ["", "## Driver lift (AP with all drivers − AP rain-only), per model", ""]
    lift = agg["driver_lift_all_minus_rain"]
    if lift:
        for m, d in sorted(lift.items(), key=lambda kv: kv[1] or 0, reverse=True):
            tag = "KEEP" if (d or 0) >= 0.005 else ("WASH" if abs(d or 0) < 0.005 else "HURTS")
            L.append(f"- `{m}`: {d:+.4f}  [{tag}]")
    else:
        L.append("- (need both rain and all cells per model to compute lift)")
    if agg.get("unavailable"):
        L += ["", f"_Unavailable (missing dep): {', '.join(agg['unavailable'])}_"]
    if agg.get("errored"):
        L += ["", "## Errored cells (fix these)"]
        for e in agg["errored"]:
            L.append(f"- `{e['model']}`/{e['drivers']}: {e['error']}")
    hab = agg.get("hab") or {}
    L += ["", "## HAB task (marine domoic-acid, next-visit pDA exceedance)", ""]
    if hab.get("headline"):
        h = hab["headline"]
        L.append(f"- AP {h.get('ap')} · ROC-AUC {h.get('roc_auc')} · best baseline AP "
                 f"{h.get('best_baseline_ap')} · beats baselines: {h.get('beats_baselines')} "
                 f"· LOSO beats seasonal-naive: {h.get('loso_beats_seasonal')}")
        if hab.get("driver_effect"):
            L.append(f"- driver effect on HAB: **{hab['driver_effect']}** "
                     "(sparse 52-event panel — extra drivers can overfit)")
    else:
        L.append(f"- status: {hab.get('status')} {hab.get('note','')}")
    L += ["", "_Honest reads: AP(cal)=Average Precision after isotonic calibration; the bar is the "
          "best operational baseline (AB411 rain rule / Virtual-Beach MLR / station memory). A driver "
          "group that moves AP < 0.005 is a WASH._"]
    return "\n".join(L)


def _log_learnings(agg: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    top = agg["leaderboard"][0] if agg["leaderboard"] else {}
    rec = {"ts": "bench", "best_model": top.get("model"), "best_drivers": top.get("drivers"),
           "best_ap": top.get("ap_calibrated"), "driver_lift": agg["driver_lift_all_minus_rain"],
           "errored": [e["model"] for e in agg["errored"]],
           "hab_ap": (agg.get("hab") or {}).get("headline", {}).get("ap")}
    with LEARNINGS.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")


# ---------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=None, help="comma list (default: all in zoo)")
    ap.add_argument("--drivers", default="rain,all", help="comma subset of {rain,all}")
    ap.add_argument("--label", default="enterococcus")
    ap.add_argument("--reveal-lag", type=int, default=2)
    ap.add_argument("--skip-completed", action="store_true")
    ap.add_argument("--no-hab", action="store_true")
    args = ap.parse_args(argv)

    zoo = model_zoo()
    models = [m.strip() for m in args.models.split(",")] if args.models else list(zoo)
    driversets = [d.strip() for d in args.drivers.split(",") if d.strip() in DRIVER_SETS]
    print(f"[bench] {len(models)} models x {len(driversets)} driver-sets = "
          f"{len(models) * len(driversets)} cells")
    cells = []
    for model in models:
        for ds in driversets:
            print(f"  [run] {model} / {ds} ...", flush=True)
            r = run_cell(model, ds, label=args.label, reveal_lag=args.reveal_lag,
                         skip_completed=args.skip_completed)
            hl = r.get("headline", {})
            print(f"      -> {r['status']}"
                  + (f" AP(cal)={hl.get('ap_calibrated')} AUC={hl.get('roc_auc')}"
                     if r['status'] == 'ok' else f" {r.get('error','')}"), flush=True)
            cells.append(r)

    hab = None if args.no_hab else run_hab()
    if hab:
        print(f"  [hab] {hab.get('status')} {hab.get('headline','')}")

    agg = aggregate(cells, hab)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "leaderboard.json").write_text(json.dumps(agg, indent=2, default=str), encoding="utf-8")
    (OUT / "leaderboard.md").write_text(render_md(agg) + "\n", encoding="utf-8")
    _log_learnings(agg)
    print(f"\n[bench] wrote {OUT / 'leaderboard.md'}  (ok {agg['n_ok']}/{agg['n_cells']})")
    if agg["leaderboard"]:
        b = agg["leaderboard"][0]
        print(f"[bench] best: {b['model']}/{b['drivers']} AP(cal)={b['ap_calibrated']} "
              f"AUC={b['roc_auc']}")
    return 0 if agg["n_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
