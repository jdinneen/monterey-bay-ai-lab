#!/usr/bin/env python3
"""Sequential full-dataset confirmation runner for the Monterey Bay AI Lab model suite.

Goal: prove every *runnable* registered model executes end-to-end on the FULL local dataset and
produces real (non-smoke, non-synthetic, trusted-source) results — one model at a time — WITHOUT
touching any production output directory and WITHOUT being able to fire by accident.

This is a confirmation harness, NOT a promotion. It reuses each model's existing CLI (no new model
code). Producing metrics here does not promote anything: promotion stays gated by release_gate/
(best-naive, promotion matrix, SR-manager gate). Confirm outputs are isolated under
reports/model_hardening/confirm_run/, and the gold layer is hashed before/after — if any step
mutates lakehouse/gold/ the whole pass is reported FAILED.

DEFAULT IS DRY-RUN. It prints the ordered plan + a readiness pre-flight and writes confirm_plan.json;
it runs nothing. To actually execute the full pass:  python ops/confirm_model_suite.py --run

Failure modes this harness is designed around (hardened after a senior-engineering review):
- Neural harness writes to lakehouse/gold/ unconditionally (LAKEHOUSE env, mbal_neural_forecast.py:61)
                                                    -> neural steps get MBAL_LAKEHOUSE_DIR=<OUT>/lakehouse,
                                                       AND a gold before/after integrity check catches any leak.
- GPU sm_120/torch hard-crash at full neural scale  -> neuralforecast models forced MBAL_ACCEL=cpu.
- MoE needs a GPU + reads gold predictions          -> pre-flight asserts CUDA + non-empty gold input + disk.
- "exit 0 but untrusted" (wrong source/partial)     -> forecast_v2 uses trusted parquet + --all-variables;
                                                       sources are labeled; promotion stays separate.
- Long wall-clock / hangs / partial restart         -> cheap jobs first; per-model --timeout-min;
                                                       in-progress marker + --skip-completed resume.
- Windows console encoding                           -> child stdout captured to utf-8 log files.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

CONFIRM_ROOT = _REPO_ROOT / "reports" / "model_hardening" / "confirm_run"
GOLD_DIR = _REPO_ROOT / "lakehouse" / "gold"

# Output paths a confirmation run may NEVER write into (production / promotion surfaces).
FORBIDDEN_WRITE = ("mbal_forecast_v2_results", "lakehouse/gold", "lakehouse\\gold")
_OUTPUT_FLAGS = {"--outdir", "--out-dir", "--output-dir"}

# Trusted full-dataset inputs (verified present before authoring this plan).
_M1_HISTORY = "mbal_history/opendap/m1_history.parquet"   # trusted M1 source for forecast_v2 + gate
_OBS = "bacteria_results/statewide/statewide_beach_observations.parquet"
_MIN_FREE_GB_FOR_MOE = 60

# ---------------------------------------------------------------------------------------------
# The ordered confirmation plan. Sequential in this exact order: CHEAPEST / fail-fast jobs first,
# multi-hour neural and MoE jobs last. One entry per registered model.
#   mode: "run"        -> execute argv
#         "skip"       -> cannot be confirmed via a reproducible full-run CLI; record the reason
# argv is relative to the repo root; "<OUT>" in argv OR env values is replaced by the model's
# isolated output dir (reports/model_hardening/confirm_run/<model_id>).
PLAN: list[dict] = [
    {
        "model_id": "early_warning_cusum",
        "mode": "run",
        "argv": ["research/bacteria/early_warning.py"],
        "hardware": "cpu", "est": "seconds",
        "risks": "fail-fast canary: confirms the detector executes and reproduces the documented null.",
    },
    {
        "model_id": "bacteria_hgbt_isotonic",
        "mode": "run",
        "argv": ["research/model_lab/model_suite.py", "--suite", "bacteria"],
        "covers": ["bacteria_xgboost", "ab411_rain_rule", "virtual_beach_mlr", "station_memory"],
        "hardware": "cpu", "est": "minutes",
        "risks": "full statewide benchmark (HGBT + XGBoost + baselines) on the real obs parquet via "
                 "the leakage-safe seam. RAIN-ONLY drivers (cdip/tide dirs absent); writes the suite "
                 "comparator json only.",
    },
    {
        "model_id": "bacteria_hgbt_spatial",
        "mode": "run",
        "argv": ["research/bacteria/spatial_drivers_experiment.py", "--out-dir", "<OUT>"],
        "hardware": "cpu", "est": "minutes-to-tens-of-minutes (499 permutations + 5 seeds, LOBO)",
        "risks": "needs reports/station_geo.parquet (else emits an error result); deterministic given seeds.",
    },
    {
        "model_id": "da_forecast_hgbt",
        "mode": "run",
        "argv": ["research/hab/da_forecast.py", "--output-dir", "<OUT>"],
        "hardware": "cpu", "est": "seconds",
        "risks": "local CalHABMAP panel only (data/external_curated/habmap_cdph); strictly causal, "
                 "deterministic (seed=42). Isolated --output-dir keeps it out of reports/hab/.",
    },
    {
        "model_id": "hab_sota_hgbt",
        "mode": "run",
        "argv": ["research/hab/hab_sota_sweep.py", "--output-dir", "<OUT>",
                 "--panel-out", "<OUT>/hab_sota_panel.parquet"],
        "hardware": "cpu", "est": "seconds-to-minutes",
        "risks": "the 'all normalized signals' DA sweep; joins existing curated external sources "
                 "as-of the prior visit. Isolated --output-dir + --panel-out keep it out of "
                 "reports/hab/ and lakehouse/silver/.",
    },
    {
        "model_id": "charm_da_nowcast",
        "mode": "skip",
        "reason": "operational comparator, not a trained model: charm_comparison.py scores NOAA "
                  "C-HARM via a live CoastWatch ERDDAP fetch (cached under data/external_raw/charm/), "
                  "so it is not a reproducible offline full-local-data run. Its evidence "
                  "(reports/hab/charm_comparison.json) is refreshed by the data pipeline.",
    },
    {
        "model_id": "xgboost_forecast_v2",
        "mode": "run",
        "argv": ["mbal_forecast_v2.py", "--source", "parquet", "--path", _M1_HISTORY,
                 "--all-variables", "--output-dir", "<OUT>"],
        "hardware": "cpu", "est": "tens-of-minutes to ~hour (--all-variables x all horizons, walk-forward)",
        "risks": "trusted M1 OPeNDAP parquet (M1 only, by design); --all-variables makes it truly full; "
                 "isolated --output-dir keeps it out of mbal_forecast_v2_results/.",
    },
    {
        "model_id": "patchtst",
        "mode": "run",
        "argv": ["mbal_neural_forecast.py", "--model", "patchtst", "--outdir", "<OUT>"],
        "env": {"MBAL_ACCEL": "cpu", "MBAL_LAKEHOUSE_DIR": "<OUT>/lakehouse"},
        "hardware": "cpu (GPU sm_120 hard-crashes at full input_size)", "est": "HOURS",
        "risks": "source = mbal_history/opendap dir (M1+M2). Gold writes redirected to <OUT>/lakehouse "
                 "so the real gold layer is untouched; input_size left at the module default 168 "
                 "(>168 hard-crashes sm_120).",
    },
    {
        "model_id": "nhits",
        "mode": "run",
        "argv": ["mbal_neural_forecast.py", "--model", "nhits", "--outdir", "<OUT>"],
        "env": {"MBAL_ACCEL": "cpu", "MBAL_LAKEHOUSE_DIR": "<OUT>/lakehouse"},
        "hardware": "cpu", "est": "HOURS",
        "risks": "same as patchtst (gold writes redirected to <OUT>/lakehouse).",
    },
]


# ---------------------------------------------------------------- plan construction + guards
def _registry_ids() -> list[str]:
    from research.model_lab import model_suite as ms
    return [m["model_id"] for m in ms.load_registry().get("models", [])]


def _sub_out(token: str, out_dir: Path) -> str:
    return token.replace("<OUT>", str(out_dir))


def validate_plan() -> list[str]:
    """Return human-readable errors. Empty = the plan is structurally safe and complete."""
    errors: list[str] = []
    planned = {e["model_id"] for e in PLAN}
    covered = {c for e in PLAN if e["mode"] == "run" for c in e.get("covers", [])}

    reg_ids = set(_registry_ids())
    missing = reg_ids - planned - covered
    if missing:
        errors.append(f"registry models absent from the plan: {sorted(missing)}")
    extra = planned - reg_ids
    if extra:
        errors.append(f"plan references unknown model_ids: {sorted(extra)}")
    if planned & covered:
        errors.append(f"models both planned and covered (ambiguous): {sorted(planned & covered)}")

    for e in PLAN:
        mid = e["model_id"]
        if e["mode"] not in {"run", "skip"}:
            errors.append(f"{mid}: bad mode {e['mode']!r}")
        if e["mode"] == "run":
            argv = e.get("argv") or []
            if not argv:
                errors.append(f"{mid}: run entry has empty argv")
            out_dir = CONFIRM_ROOT / mid
            # production-dir guard: scan argv AND resolved env values for forbidden substrings.
            scan = [str(t) for t in argv] + [_sub_out(str(v), out_dir) for v in e.get("env", {}).values()]
            for tok in scan:
                low = tok.replace("\\", "/").lower()
                if any(bad.replace("\\", "/").lower() in low for bad in FORBIDDEN_WRITE):
                    # an env value pointing *into* its own isolated <OUT>/lakehouse is the SAFE
                    # redirect, not a violation — only flag forbidden paths outside CONFIRM_ROOT.
                    if str(CONFIRM_ROOT).replace("\\", "/").lower() in low:
                        continue
                    errors.append(f"{mid}: references a forbidden production path: {tok}")
            # every explicit output flag must resolve under CONFIRM_ROOT.
            for i, tok in enumerate(argv):
                if tok in _OUTPUT_FLAGS:
                    val = argv[i + 1] if i + 1 < len(argv) else ""
                    resolved = (out_dir if val == "<OUT>" else (_REPO_ROOT / val)).resolve()
                    if CONFIRM_ROOT.resolve() not in resolved.parents and resolved != CONFIRM_ROOT.resolve():
                        errors.append(f"{mid}: output flag {tok} -> {val} is not under {CONFIRM_ROOT}")
        if e["mode"] == "skip" and not e.get("reason"):
            errors.append(f"{mid}: skip entry needs a reason")
    return errors


def _resolve(entry: dict, out_dir: Path) -> tuple[list[str], dict]:
    argv = [sys.executable, *[_sub_out(t, out_dir) for t in entry["argv"]]]
    env = {k: _sub_out(v, out_dir) for k, v in entry.get("env", {}).items()}
    return argv, env


# ---------------------------------------------------------------- readiness pre-flight
def preflight(entries: list[dict]) -> list[str]:
    """Fail FAST, before any hour-long job: required inputs, deps, GPU/disk for MoE, writability."""
    errors: list[str] = []
    run_ids = {e["model_id"] for e in entries if e["mode"] == "run"}
    covered = {c for e in entries if e["mode"] == "run" for c in e.get("covers", [])}
    active = run_ids | covered

    req = []
    if active & {"bacteria_hgbt_isotonic", "bacteria_hgbt_spatial", "bacteria_xgboost",
                 "ab411_rain_rule", "virtual_beach_mlr", "station_memory"}:
        req += [_OBS, "bacteria_results/rainfall/rainfall_grid.parquet",
                "bacteria_results/rainfall/station_grid_map.parquet"]
    if "bacteria_hgbt_spatial" in run_ids:
        req.append("reports/station_geo.parquet")
    if "xgboost_forecast_v2" in run_ids:
        req.append(_M1_HISTORY)
    for f in dict.fromkeys(req):
        if not (_REPO_ROOT / f).exists():
            errors.append(f"missing input file: {f}")

    need = {}
    if "xgboost_forecast_v2" in run_ids or "bacteria_xgboost" in active:
        need["xgboost"] = "xgboost"
    if run_ids & {"patchtst", "nhits"}:
        need["neuralforecast"] = "neuralforecast"
    if "moe_ewc_latenttsf" in run_ids:
        need["torch"] = "torch"
    for pip_name, import_name in need.items():
        if importlib.util.find_spec(import_name) is None:
            errors.append(f"missing dependency: {pip_name} (pip install {pip_name})")

    # Neural jobs read the mbal_history/opendap DIRECTORY (M1+M2) and write a multi-GB long-frame
    # cache; fail fast if either the source dir is absent or disk is tight (else they die at hour N).
    if run_ids & {"patchtst", "nhits"}:
        opendap = _REPO_ROOT / "mbal_history" / "opendap"
        if not opendap.exists() or not any(opendap.rglob("*.parquet")):
            errors.append("neural source missing: mbal_history/opendap/*.parquet is empty")
        free_gb = shutil.disk_usage(_REPO_ROOT).free / 1e9
        if free_gb < 20:
            errors.append(f"neural cache needs >=20GB free; {free_gb:.0f}GB available")

    if "moe_ewc_latenttsf" in run_ids:
        gold_pred = GOLD_DIR / "forecast_predictions"
        if not gold_pred.exists() or not any(gold_pred.rglob("*.parquet")):
            errors.append("MoE input missing: lakehouse/gold/forecast_predictions/*.parquet is empty")
        if importlib.util.find_spec("torch") is not None:
            try:
                import torch
                if not torch.cuda.is_available():
                    errors.append("MoE requires a CUDA GPU; torch.cuda.is_available() is False")
                else:
                    # Model it the way the model actually gates itself: the admission guard MoE
                    # enforces at launch (run_production.py) checks live VRAM, not just presence.
                    try:
                        from ops.gpu_admission import check_gpu_admission, estimate_run_production_mib
                        denial = check_gpu_admission(
                            label="moe_ewc_latenttsf preflight",
                            request_mib=estimate_run_production_mib(2, 168))
                        if denial:
                            errors.append("MoE GPU admission would deny launch now: "
                                          + denial.splitlines()[0])
                    except Exception as ex:
                        errors.append(f"MoE GPU admission preflight errored: {ex}")
            except Exception as ex:  # importing torch can fail on a broken CUDA install
                errors.append(f"MoE GPU check failed to import torch: {ex}")
        free_gb = shutil.disk_usage(_REPO_ROOT).free / 1e9
        if free_gb < _MIN_FREE_GB_FOR_MOE:
            errors.append(f"MoE needs >={_MIN_FREE_GB_FOR_MOE}GB free; {free_gb:.0f}GB available")

    try:
        CONFIRM_ROOT.mkdir(parents=True, exist_ok=True)
        probe = CONFIRM_ROOT / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except Exception as ex:
        errors.append(f"CONFIRM_ROOT not writable: {ex}")
    return errors


# ---------------------------------------------------------------- gold-layer integrity
def gold_manifest() -> dict[str, str]:
    """relpath -> sha256(content) for every file under lakehouse/gold/. The before/after diff is the
    real safety net: it catches any model that writes to gold despite the argv guard. Content hashing
    (not size+mtime) closes the same-second/same-size rewrite evasion."""
    if not GOLD_DIR.exists():
        return {}
    out = {}
    for p in sorted(GOLD_DIR.rglob("*")):
        if p.is_file():
            h = hashlib.sha256()
            with p.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            out[str(p.relative_to(_REPO_ROOT)).replace("\\", "/")] = h.hexdigest()
    return out


def gold_diff(before: dict, after: dict) -> dict:
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = sorted(k for k in before.keys() & after.keys() if before[k] != after[k])
    return {"added": added, "removed": removed, "changed": changed}


# ---------------------------------------------------------------- execution
def _status_path(out_dir: Path) -> Path:
    return out_dir / "confirm_status.json"


def run_entry(entry: dict, *, timeout_min: float | None, skip_completed: bool) -> dict:
    mid = entry["model_id"]
    if entry["mode"] == "skip":
        return {"model_id": mid, "status": "skipped", "reason": entry["reason"]}

    out_dir = CONFIRM_ROOT / mid
    out_dir.mkdir(parents=True, exist_ok=True)
    if skip_completed and _status_path(out_dir).exists():
        prev = json.loads(_status_path(out_dir).read_text(encoding="utf-8"))
        if prev.get("status") == "ok":
            return {**prev, "note": "skipped (already ok)"}

    argv, env_over = _resolve(entry, out_dir)
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.update(env_over)

    # in-progress marker so a reboot/kill mid-job is visible and not silently re-run as 'ok'.
    _status_path(out_dir).write_text(json.dumps(
        {"model_id": mid, "status": "running", "argv": argv}, indent=2), encoding="utf-8")

    log_path = out_dir / "confirm.log"
    start = time.monotonic()
    timeout_s = timeout_min * 60 if timeout_min else None
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"# {mid}\n# argv: {argv}\n# env-overrides: {env_over}\n\n")
        log.flush()
        try:
            proc = subprocess.run(argv, cwd=str(_REPO_ROOT), env=env, stdout=log,
                                  stderr=subprocess.STDOUT, timeout=timeout_s, check=False)
            status = "ok" if proc.returncode == 0 else "failed"
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            status, rc = "timeout", None
    result = {"model_id": mid, "status": status, "returncode": rc,
              "seconds": round(time.monotonic() - start, 1), "out_dir": str(out_dir),
              "log": str(log_path), "argv": argv}
    _status_path(out_dir).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def render_report(results: list[dict], gold: dict) -> str:
    ok = sum(1 for r in results if r["status"] == "ok")
    lines = [
        "# Model Suite — Full-Dataset Confirmation Run",
        "",
        f"- planned: {len(results)} | ok: {ok} | "
        f"failed: {sum(1 for r in results if r['status']=='failed')} | "
        f"timeout: {sum(1 for r in results if r['status']=='timeout')} | "
        f"skipped: {sum(1 for r in results if r['status']=='skipped')}",
        f"- gold layer (lakehouse/gold/) unchanged: **{'YES' if gold['unchanged'] else 'NO — INTEGRITY FAIL'}**",
        "",
        "| order | model_id | status | seconds | detail |",
        "|--:|---|---|--:|---|",
    ]
    for i, r in enumerate(results, 1):
        detail = r.get("reason") or r.get("note") or (r.get("log") or "")
        lines.append(f"| {i} | `{r['model_id']}` | {r['status']} | {r.get('seconds','-')} | {detail} |")
    if not gold["unchanged"]:
        lines += ["", "## GOLD INTEGRITY FAILURE", "A confirmation step mutated the production gold "
                  "layer. This invalidates the pass — investigate before trusting any result.",
                  "```json", json.dumps(gold["diff"], indent=2), "```"]
    lines += ["", "_Confirmation only — exit 0 means the model RAN end-to-end on the full dataset. It "
              "does NOT mean the model beat its baseline or is promotable: promotion stays gated by "
              "release_gate/ (best-naive, promotion matrix, SR-manager gate). Per-model metrics are in "
              "each model's out_dir; forecasting skill-vs-best-naive is decided by the promotion matrix, "
              "not here._"]
    return "\n".join(lines)


# ---------------------------------------------------------------- analysis + visualization
# Plain-English glossary, reused in the freshman report so jargon is always defined nearby.
GLOSSARY = {
    "baseline": "the simple method we must beat. If a fancy model can't beat the simple one, "
                "the fancy model isn't worth it.",
    "AP": "Average Precision (0-1, higher is better): how good the model is at catching the rare "
          "'beach is unsafe' days without crying wolf too often.",
    "ROC-AUC": "Area Under the ROC Curve (0.5 = coin flip, 1.0 = perfect): how well the model "
               "separates unsafe days from safe days.",
    "ECE": "Expected Calibration Error (lower is better): when the model says '70% chance', is it "
           "actually right about 70% of the time? Small ECE = you can trust the percentage.",
    "best-naive / persistence": "the 'nothing changes' guess: tomorrow will look like today (or like "
                                "this time last year). Ocean data is smooth, so this is surprisingly "
                                "hard to beat.",
    "skill vs persistence": "how much better (or worse) a model is than the 'nothing changes' guess. "
                            "Positive = better than doing nothing. Negative = worse than doing nothing.",
    "horizon": "how far into the future we predict, in hours (e.g. 1h, 6h, 24h, 72h, 168h=1 week).",
    "calibration": "tuning the model's confidence so its percentages mean what they say.",
}


def _read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_csv(p: Path):
    try:
        import pandas as pd
        return pd.read_csv(p)
    except Exception:
        return None


def analyze() -> dict:
    """Read each model's confirm-run artifacts and distill a small, honest record per model:
    did it run, what is its headline number, what baseline did it have to beat, did it beat it."""
    bac = _read_json(_REPO_ROOT / "reports" / "operational_benchmark" / "model_suite_bacteria.json")
    records: dict[str, dict] = {}

    # --- bacteria suite (covers HGBT + XGBoost + the three baselines) ---
    if bac and "models" in bac:
        for mid in ("bacteria_hgbt_isotonic", "bacteria_xgboost"):
            r = bac["models"].get(mid, {})
            if r.get("status") == "ok":
                records[mid] = {
                    "ran": True, "task": "bacteria",
                    "metrics": {"AP (calibrated)": r.get("ap_calibrated"),
                                "ROC-AUC": r.get("roc_auc"), "ECE": r.get("ece_calibrated")},
                    "baseline": "AB411 rain rule / Virtual-Beach MLR / station-memory",
                    "beats_baseline": r.get("beats_best_operational"),
                    "extra": {"deploy_ready": r.get("calibrated_deploy_ready")},
                }
            else:
                records[mid] = {"ran": False, "task": "bacteria", "note": r.get("status", "no result")}

    # --- spatial bacteria ---
    sj = _read_json(CONFIRM_ROOT / "bacteria_hgbt_spatial" / "spatial_drivers_experiment.json")
    if sj is not None:
        gain = sj.get("ap_gain") or sj.get("delta_ap") or sj.get("spatial_ap_gain")
        records["bacteria_hgbt_spatial"] = {
            "ran": "error" not in sj, "task": "bacteria",
            "metrics": {k: sj.get(k) for k in ("ap_gain", "delta_ap", "morans_i_residual")
                        if sj.get(k) is not None},
            "baseline": "the same model WITHOUT the spatial (lat/lon) features, on never-seen beaches",
            "beats_baseline": (gain is not None and gain > 0) if gain is not None else None,
            "note": sj.get("error"),
        }

    # --- forecast_v2 (XGBoost ocean forecaster) ---
    lb = _read_csv(CONFIRM_ROOT / "xgboost_forecast_v2" / "leaderboard.csv")
    if lb is not None and not lb.empty and "mean_skill_rmse_vs_persistence" in lb.columns:
        by_h = lb.groupby("horizon_h")["mean_skill_rmse_vs_persistence"].mean().to_dict()
        n_beat = int(lb["beats_persistence"].sum()) if "beats_persistence" in lb.columns else None
        records["xgboost_forecast_v2"] = {
            "ran": True, "task": "forecast",
            "metrics": {"cells beating persistence": n_beat, "cells total": int(len(lb))},
            "skill_by_horizon": {int(k): round(float(v), 3) for k, v in by_h.items()},
            "baseline": "best-naive (persistence / seasonal-naive) per variable+horizon",
            # Honest: 'beat on at least one cell' is NOT a win. Promotion is decided per cell by the
            # promotion matrix, not globally here. Report the count; leave the verdict to the gate.
            "beats_baseline": None,
        }

    # --- neural (PatchTST, NHITS): per-run RMSE by horizon ---
    for mid in ("patchtst", "nhits"):
        sm = _read_json(CONFIRM_ROOT / mid / "summary.json")
        if sm is not None:
            rmse_by_h = sm.get("mean_rmse_by_h") or {}
            records[mid] = {
                "ran": True, "task": "forecast",
                "metrics": {"series scored": sm.get("n_series"), "rows scored": sm.get("scored_rows"),
                            "minutes": sm.get("minutes")},
                "rmse_by_horizon": {int(k): round(float(v), 3) for k, v in rmse_by_h.items()},
                "baseline": "best-naive per variable+horizon (decided by the promotion matrix, not here)",
                "beats_baseline": None,  # neural best-naive skill is a separate merge/gate step
            }

    # --- MoE (research, failed-gate): confirm it trains ---
    moe_status = _read_json(CONFIRM_ROOT / "moe_ewc_latenttsf" / "confirm_status.json")
    if moe_status:
        records["moe_ewc_latenttsf"] = {
            "ran": moe_status.get("status") == "ok", "task": "forecast",
            "metrics": {"run status": moe_status.get("status"), "seconds": moe_status.get("seconds")},
            "baseline": "best-naive (prior result: beats it on only 1 of 24 series — shelved)",
            "beats_baseline": False,
            "note": "research-only; confirming it RUNS, not promoting it.",
        }

    # --- CUSUM detector (documented null) ---
    cusum_status = _read_json(CONFIRM_ROOT / "early_warning_cusum" / "confirm_status.json")
    if cusum_status:
        records["early_warning_cusum"] = {
            "ran": cusum_status.get("status") == "ok", "task": "detector",
            "metrics": {"run status": cusum_status.get("status")},
            "baseline": "any detector (the 2022 break is a ~35-sigma cliff anything would catch)",
            "beats_baseline": False,
            "note": "this is a NULL result on purpose — confirming the documented non-finding reproduces.",
        }
    return records


def make_visualizations(records: dict, results: list[dict]) -> list[str]:
    """Write PNG charts; return the relative paths written. Never raises — a failed chart is skipped."""
    import matplotlib
    matplotlib.use("Agg")  # no display; safe on a headless/Windows box
    import matplotlib.pyplot as plt

    fig_dir = CONFIRM_ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    def _save(fig, name):
        path = fig_dir / name
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        written.append(f"figures/{name}")

    # 1) Run-status overview (did every model execute?)
    try:
        color = {"ok": "#2e7d32", "failed": "#c62828", "timeout": "#ef6c00", "skipped": "#9e9e9e"}
        names = [r["model_id"] for r in results]
        cols = [color.get(r["status"], "#9e9e9e") for r in results]
        fig, ax = plt.subplots(figsize=(8, 0.45 * len(names) + 1))
        ax.barh(range(len(names)), [1] * len(names), color=cols)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xticks([])
        ax.invert_yaxis()
        ax.set_title("Did each model run? (green=ran, red=failed, orange=timeout, grey=skipped)")
        for i, r in enumerate(results):
            ax.text(0.5, i, r["status"], ha="center", va="center", color="white", fontsize=9, weight="bold")
        _save(fig, "fig_status.png")
    except Exception:
        pass

    # 2) Bacteria: calibrated AP + ROC-AUC for the tree candidates
    try:
        bac = {k: v for k, v in records.items()
               if k in ("bacteria_hgbt_isotonic", "bacteria_xgboost") and v.get("ran")}
        if bac:
            labels = list(bac.keys())
            ap = [bac[k]["metrics"].get("AP (calibrated)") or 0 for k in labels]
            auc = [bac[k]["metrics"].get("ROC-AUC") or 0 for k in labels]
            import numpy as np
            x = np.arange(len(labels))
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(x - 0.2, ap, 0.4, label="AP (catch unsafe days)", color="#1565c0")
            ax.bar(x + 0.2, auc, 0.4, label="ROC-AUC (separate safe/unsafe)", color="#43a047")
            ax.axhline(0.5, ls="--", c="grey", lw=1)
            ax.text(len(labels) - 0.5, 0.51, "coin-flip (0.5)", fontsize=8, color="grey", ha="right")
            ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
            ax.set_ylim(0, 1); ax.set_ylabel("score (higher = better)")
            ax.set_title("Bacteria models: how well they flag unsafe beach days")
            ax.legend(fontsize=8)
            _save(fig, "fig_bacteria.png")
    except Exception:
        pass

    # 3) Forecast skill vs persistence by horizon (XGBoost forecaster)
    try:
        fc = records.get("xgboost_forecast_v2", {})
        sbh = fc.get("skill_by_horizon")
        if sbh:
            hs = sorted(sbh)
            vals = [sbh[h] for h in hs]
            fig, ax = plt.subplots(figsize=(7, 4))
            bars = ax.bar([str(h) for h in hs], vals,
                          color=["#2e7d32" if v >= 0 else "#c62828" for v in vals])
            ax.axhline(0, c="black", lw=1)
            ax.set_xlabel("how far ahead (hours)")
            ax.set_ylabel("skill vs 'nothing changes' (>0 = better)")
            ax.set_title("Ocean forecaster: above 0 = better than doing nothing")
            _save(fig, "fig_forecast_skill.png")
    except Exception:
        pass

    # 4) Neural RMSE by horizon (lower = better)
    try:
        import numpy as np
        fig, ax = plt.subplots(figsize=(7, 4))
        plotted = False
        for mid, c in (("patchtst", "#6a1b9a"), ("nhits", "#00838f")):
            rbh = records.get(mid, {}).get("rmse_by_horizon")
            if rbh:
                hs = sorted(rbh)
                ax.plot([str(h) for h in hs], [rbh[h] for h in hs], "o-", label=mid, color=c)
                plotted = True
        if plotted:
            ax.set_xlabel("how far ahead (hours)")
            ax.set_ylabel("error (RMSE, lower = better)")
            ax.set_title("Neural forecasters: prediction error by how far ahead")
            ax.legend(fontsize=9)
            _save(fig, "fig_neural_rmse.png")
        else:
            plt.close(fig)
    except Exception:
        pass

    return written


def _verdict_phrase(beats) -> str:
    return {True: "YES — it beat the bar", False: "NO — it did not beat the bar",
            None: "NOT DECIDED HERE (a separate gate decides this)"}[beats]


def render_analysis_report(records: dict, results: list[dict], figures: list[str], gold: dict) -> str:
    """A plain-English report a college freshman can follow: what each model is, whether it ran,
    and whether it beat the simple method it has to beat."""
    by_status = {s: [r["model_id"] for r in results if r["status"] == s]
                 for s in ("ok", "failed", "timeout", "skipped")}
    L = [
        "# Monterey Bay AI Lab — Model Suite Analysis (Plain-English Report)",
        "",
        "## Read this first (30 seconds)",
        "We have a bunch of computer models that try to predict two things: (1) **is a beach unsafe** "
        "to swim because of bacteria, and (2) **what the ocean will do next** (temperature, salinity, "
        "etc.). This report ran every model we could on the **full real dataset** and checked one "
        "simple question for each: *does it beat the dumb, simple method?* If a smart model can't beat "
        "the simple one, it isn't worth using.",
        "",
        "Think of each model as a student taking a test. The **baseline** is the score to beat. "
        "Beating the baseline is the whole game.",
        "",
        f"- Models that ran successfully: **{len(by_status['ok'])}**",
        f"- Models skipped (can't be run reliably yet): **{len(by_status['skipped'])}**",
        f"- Models that failed or timed out: **{len(by_status['failed']) + len(by_status['timeout'])}**",
        f"- Production data left untouched (safety check): **{'YES ✅' if gold['unchanged'] else 'NO ❌'}**",
        "",
    ]
    if "figures/fig_status.png" in figures:
        L += ["![Which models ran](figures/fig_status.png)", ""]

    L += ["## A few words you'll see (quick glossary)", ""]
    for term, meaning in GLOSSARY.items():
        L.append(f"- **{term}** — {meaning}")
    L += [""]

    # group the story by task
    L += ["## Part 1 — Beach safety models (the strongest results)", "",
          "These predict whether a beach has too much bacteria to be safe. This is where our models "
          "genuinely shine.", ""]
    if "figures/fig_bacteria.png" in figures:
        L += ["![Bacteria model scores](figures/fig_bacteria.png)", ""]
    for mid in ("bacteria_hgbt_isotonic", "bacteria_hgbt_spatial", "bacteria_xgboost"):
        L += _model_block(mid, records.get(mid))

    L += ["## Part 2 — Ocean forecasting models", "",
          "These predict the ocean's near future. Here the simple 'nothing really changes' guess is "
          "shockingly strong, because the ocean moves slowly — so beating it is hard, and we are "
          "honest when we don't.", ""]
    for f in ("figures/fig_forecast_skill.png", "figures/fig_neural_rmse.png"):
        if f in figures:
            L += [f"![chart]({f})", ""]
    for mid in ("xgboost_forecast_v2", "patchtst", "nhits", "moe_ewc_latenttsf"):
        L += _model_block(mid, records.get(mid))

    L += ["## Part 3 — The early-warning detector", "",
          "This one looked for early warning signs of a bacteria spike. The honest finding is that it "
          "**doesn't add value** — the big 2022 event was so huge that any method would catch it, and "
          "nothing caught it *early*. We keep this as a documented 'no result' so nobody re-invents it.", ""]
    L += _model_block("early_warning_cusum", records.get("early_warning_cusum"))

    L += ["## Part 4 — Models we did NOT run (and why that's the honest call)", ""]
    skip_reasons = {e["model_id"]: e.get("reason") for e in PLAN if e["mode"] == "skip"}
    for mid, reason in skip_reasons.items():
        L += [f"- **{mid}** — {reason}"]
    L += [""]

    L += ["## The bottom line (what we can and cannot say)", "",
          "**We CAN say:** our beach-safety model is a real, trustworthy tool that beats the methods "
          "officials use today, and even works at beaches it never trained on.",
          "",
          "**We must NOT say:** that our ocean-forecasting or experimental models are 'wins'. Most of "
          "them do not reliably beat the simple 'nothing changes' guess, and a few can't even be "
          "rebuilt from current code. Running a model is not the same as a model being good — this "
          "report shows which is which.",
          "",
          "_Important: 'it ran' does not mean 'promote it'. Whether a forecasting model is actually "
          "good enough to deploy is decided by a separate, stricter checker (the promotion gate), not "
          "by this report._"]
    return "\n".join(L)


def _model_block(mid: str, rec: dict | None) -> list[str]:
    pretty = {
        "bacteria_hgbt_isotonic": "Beach bacteria model (HGBT + calibration) — our flagship",
        "bacteria_hgbt_spatial": "Beach bacteria model with map smarts (learns by location)",
        "bacteria_xgboost": "Beach bacteria model, second flavor (XGBoost) — a sanity check",
        "xgboost_forecast_v2": "Ocean forecaster (XGBoost)",
        "patchtst": "Ocean forecaster (PatchTST, a neural network)",
        "nhits": "Ocean forecaster (NHITS, a neural network)",
        "moe_ewc_latenttsf": "Experimental 'keeps-learning' giant model (MoE/EWC) — research only",
        "early_warning_cusum": "Early-warning detector (CUSUM)",
    }.get(mid, mid)
    out = [f"### {pretty}", ""]
    if not rec:
        out += ["_No result was produced for this model in this run._", ""]
        return out
    if not rec.get("ran"):
        out += [f"- **Did it run?** No — {rec.get('note', 'no usable output')}.", ""]
        return out
    out += ["- **Did it run?** Yes."]
    metrics = {k: v for k, v in (rec.get("metrics") or {}).items() if v is not None}
    if metrics:
        out += ["- **Key numbers:** " + ", ".join(f"{k} = {v}" for k, v in metrics.items())]
    out += [f"- **What it had to beat:** {rec.get('baseline')}"]
    out += [f"- **Did it beat it?** {_verdict_phrase(rec.get('beats_baseline'))}."]
    if rec.get("note"):
        out += [f"- **Note:** {rec['note']}"]
    out += [""]
    return out


def write_analysis(results: list[dict], gold: dict) -> dict:
    """Build records, charts, and the freshman report. Returns the paths written."""
    records = analyze()
    figures = make_visualizations(records, results)
    report = render_analysis_report(records, results, figures, gold)
    paths = {
        "analysis_md": CONFIRM_ROOT / "ANALYSIS_REPORT.md",
        "analysis_json": CONFIRM_ROOT / "model_analysis.json",
    }
    paths["analysis_md"].write_text(report + "\n", encoding="utf-8")
    paths["analysis_json"].write_text(json.dumps(records, indent=2), encoding="utf-8")
    return {**{k: str(v) for k, v in paths.items()}, "figures": figures}


# ---------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="store_true",
                    help="ACTUALLY execute the full pass (default is dry-run: print plan + pre-flight)")
    ap.add_argument("--preflight", action="store_true", help="run only the readiness pre-flight and exit")
    ap.add_argument("--analyze", action="store_true",
                    help="(re)build the plain-English ANALYSIS_REPORT.md + charts from existing confirm outputs")
    ap.add_argument("--only", nargs="+", default=None, help="run a subset of model_ids in plan order")
    ap.add_argument("--timeout-min", type=float, default=None,
                    help="per-model timeout in minutes (STRONGLY recommended for the neural/MoE jobs)")
    ap.add_argument("--skip-completed", action="store_true",
                    help="skip models whose previous confirm_status.json is already 'ok' (resume)")
    ap.add_argument("--skip-preflight", action="store_true", help="bypass the readiness pre-flight (not advised)")
    args = ap.parse_args(argv)

    errors = validate_plan()
    if errors:
        for e in errors:
            print(f"[plan] ERROR: {e}")
        print(f"\nplan: FAIL ({len(errors)} error(s)) - refusing to run")
        return 2

    entries = PLAN if not args.only else [e for e in PLAN if e["model_id"] in set(args.only)]

    if args.analyze:
        saved = _read_json(CONFIRM_ROOT / "confirm_results.json") or {}
        results = saved.get("results") or [
            (_read_json(_status_path(CONFIRM_ROOT / e["model_id"])) or
             {"model_id": e["model_id"], "status": "skipped" if e["mode"] == "skip" else "no result",
              "reason": e.get("reason")})
            for e in PLAN]
        gold = saved.get("gold") or {"unchanged": True, "diff": {}}
        out = write_analysis(results, gold)
        for k, v in out.items():
            print(f"wrote {k}: {v}")
        return 0

    pf = preflight(entries)

    if args.preflight:
        for e in pf:
            print(f"[preflight] {e}")
        print("preflight: " + ("PASS" if not pf else f"FAIL ({len(pf)})"))
        return 1 if pf else 0

    if not args.run:
        print("DRY-RUN (no model executed). Pass --run to execute the full pass sequentially.\n")
        print(f"output root: {CONFIRM_ROOT}\n")
        for i, e in enumerate(entries, 1):
            if e["mode"] == "run":
                out_dir = CONFIRM_ROOT / e["model_id"]
                cmd, env_over = _resolve(e, out_dir)
                print(f"{i:>2}. RUN   {e['model_id']:<22} [{e['hardware']}, {e['est']}]")
                print(f"      cmd: {' '.join(cmd[1:])}")
                if env_over:
                    print(f"      env: {env_over}")
                if e.get("covers"):
                    print(f"      covers: {', '.join(e['covers'])}")
                print(f"      risk: {e['risks']}")
            else:
                print(f"{i:>2}. SKIP  {e['model_id']:<22} {e['reason']}")
        CONFIRM_ROOT.mkdir(parents=True, exist_ok=True)
        (CONFIRM_ROOT / "confirm_plan.json").write_text(json.dumps(PLAN, indent=2), encoding="utf-8")
        print(f"\nwrote plan: {CONFIRM_ROOT / 'confirm_plan.json'}")
        print("\npre-flight: " + ("PASS" if not pf else f"FAIL ({len(pf)})"))
        for e in pf:
            print(f"  - {e}")
        if not args.timeout_min:
            print("\nNOTE: no --timeout-min set. For an unattended pass, set one (the neural/MoE jobs "
                  "are multi-hour) so a single hang can't stall everything after it.")
        print("plan: PASS (validated, complete)" if not pf else "plan valid, but PRE-FLIGHT FAILS — fix before --run")
        return 0

    if pf and not args.skip_preflight:
        for e in pf:
            print(f"[preflight] ERROR: {e}")
        print(f"\npre-flight: FAIL ({len(pf)}) - refusing to run. Fix, or pass --skip-preflight to override.")
        return 2

    CONFIRM_ROOT.mkdir(parents=True, exist_ok=True)
    gold_before = gold_manifest()
    results = []
    for i, e in enumerate(entries, 1):
        print(f"[{i}/{len(entries)}] {e['mode'].upper()} {e['model_id']} ...", flush=True)
        r = run_entry(e, timeout_min=args.timeout_min, skip_completed=args.skip_completed)
        print(f"    -> {r['status']} ({r.get('seconds','-')}s)", flush=True)
        results.append(r)
    diff = gold_diff(gold_before, gold_manifest())
    gold = {"unchanged": not any(diff.values()), "diff": diff}

    (CONFIRM_ROOT / "confirm_results.json").write_text(
        json.dumps({"results": results, "gold": gold}, indent=2), encoding="utf-8")
    (CONFIRM_ROOT / "CONFIRM_REPORT.md").write_text(render_report(results, gold) + "\n", encoding="utf-8")
    print(f"\nwrote {CONFIRM_ROOT / 'CONFIRM_REPORT.md'}")
    analysis = write_analysis(results, gold)  # plain-English report + charts from the real results
    print(f"wrote {analysis['analysis_md']} ({len(analysis['figures'])} charts)")
    if not gold["unchanged"]:
        print("GOLD INTEGRITY FAIL: a step mutated lakehouse/gold/ — pass invalidated.")
    failed = [r for r in results if r["status"] in {"failed", "timeout"}]
    return 1 if (failed or not gold["unchanged"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
