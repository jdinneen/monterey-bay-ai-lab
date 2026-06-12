#!/usr/bin/env python3
"""Model suite: one legible, fail-closed view of every model in the lab.

Reads `model_registry.yaml` (the single source of truth) and answers the questions that keep
coming up: is it neural / MoE / tree? are its dependencies available? does it beat a baseline?
is it calibrated? is it CLAIMABLE? what blocks it? It renders `MODEL_SUITE.md` from the registry alone (no data
needed), and — for the bacteria task — can actually run registered tabular candidates through the
existing leakage-safe `operational_benchmark.run(clf=...)` seam to bound them against the
incumbent HGBT and the operational baselines.

Design choices (per docs/VALUE_GATE.md):
- The registry is metadata + the existing `clf` injection seam. NO new adapter framework.
- New tabular candidates are gated, optional, and reversible. A missing optional dependency is
  reported as `unavailable_dependency`, never a crash — and no heavy deps are added to the project.
- FAIL-CLOSED: a model is claimable only if the registry says so AND its evidence file exists.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

REGISTRY_PATH = Path(__file__).resolve().parent / "model_registry.yaml"

STATUS_ENUM = {
    "production_candidate", "benchmark", "research_candidate",
    "research_only_failed_gate", "negative_result", "baseline", "retired_claim", "orphaned",
}
# statuses that may NEVER be claimable, regardless of the claimable flag
NEVER_CLAIMABLE = {"research_only_failed_gate", "negative_result", "retired_claim",
                   "baseline", "orphaned"}
FAMILY_ENUM = {"tree", "neural", "foundation", "continual", "baseline", "detector"}
REQUIRED_FIELDS = ("model_id", "family", "architecture", "task", "status",
                   "claimable", "supports_calibration", "dependencies", "entrypoint")

# pip-name -> import-name for the dependency availability probe
_IMPORT_NAME = {
    "scikit-learn": "sklearn", "chronos-forecasting": "chronos",
    "pytorch-lightning": "pytorch_lightning",
}


# ---------------------------------------------------------------- registry I/O + validation
def load_registry(path: Path | None = None) -> dict:
    import yaml

    path = Path(path) if path else REGISTRY_PATH
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "models" not in data:
        raise ValueError(f"{path} must be a mapping with a top-level 'models:' list")
    return data


def validate(reg: dict, repo_root: Path | None = None) -> list[str]:
    """Return a list of human-readable validation errors (empty = valid)."""
    repo_root = repo_root or _REPO_ROOT
    errors: list[str] = []
    seen: set[str] = set()
    for m in reg.get("models", []):
        mid = m.get("model_id", "<missing id>")
        for f in REQUIRED_FIELDS:
            if f not in m:
                errors.append(f"{mid}: missing required field '{f}'")
        if mid in seen:
            errors.append(f"{mid}: duplicate model_id")
        seen.add(mid)
        if m.get("status") not in STATUS_ENUM:
            errors.append(f"{mid}: status '{m.get('status')}' not in {sorted(STATUS_ENUM)}")
        if m.get("family") not in FAMILY_ENUM:
            errors.append(f"{mid}: family '{m.get('family')}' not in {sorted(FAMILY_ENUM)}")
        if not isinstance(m.get("claimable"), bool):
            errors.append(f"{mid}: 'claimable' must be a bool")
        # fail-closed claim rules
        if m.get("claimable") is True:
            if m.get("status") in NEVER_CLAIMABLE:
                errors.append(f"{mid}: status '{m['status']}' may not be claimable")
            ev = (m.get("evidence") or "").strip()
            if not ev:
                errors.append(f"{mid}: claimable=true requires a non-empty 'evidence' path")
            elif not (repo_root / ev).exists():
                errors.append(f"{mid}: evidence path does not exist: {ev}")
    return errors


def is_claimable(m: dict, repo_root: Path | None = None) -> bool:
    """The fail-closed truth: declared claimable, allowed status, evidence on disk."""
    repo_root = repo_root or _REPO_ROOT
    if m.get("claimable") is not True or m.get("status") in NEVER_CLAIMABLE:
        return False
    ev = (m.get("evidence") or "").strip()
    return bool(ev) and (repo_root / ev).exists()


def deps_available(m: dict) -> bool:
    for dep in m.get("dependencies", []) or []:
        name = _IMPORT_NAME.get(dep, dep.replace("-", "_"))
        if importlib.util.find_spec(name) is None:
            return False
    return True


# ---------------------------------------------------------------- report rendering
def render_report(reg: dict, repo_root: Path | None = None) -> str:
    repo_root = repo_root or _REPO_ROOT
    models = reg.get("models", [])
    errs = validate(reg, repo_root)

    lines = [
        "# Model Suite — what exists, what has deps, what is claimable",
        "",
        "Generated from `research/model_lab/model_registry.yaml` by `research/model_lab/model_suite.py`.",
        "Fail-closed: a model is **claimable** only if the registry marks it so AND its evidence file exists.",
        "",
        f"- registry validation: {'PASS' if not errs else 'FAIL — ' + str(len(errs)) + ' error(s)'}",
        f"- models registered: {len(models)}",
        f"- claimable now: {sum(1 for m in models if is_claimable(m, repo_root))}",
        "",
        "| model | family | task | neural? | deps available? | beats baseline? | calibrated? | claimable? | status | blocker |",
        "|---|---|---|:-:|:-:|---|:-:|:-:|---|---|",
    ]
    for m in models:
        fam = m.get("family", "?")
        neural = "yes" if fam in {"neural", "foundation", "continual"} else "no"
        deps = "yes" if deps_available(m) else "no"
        cal = "yes" if m.get("supports_calibration") else "no"
        claim = "**YES**" if is_claimable(m, repo_root) else "no"
        lines.append(
            f"| `{m.get('model_id')}` | {fam} | {m.get('task')} | {neural} | {deps} | "
            f"{m.get('beats_baseline', '?')} | {cal} | {claim} | {m.get('status')} | "
            f"{m.get('blockers', '') or ''} |")

    lines += [
        "",
        "## Legend",
        "- **neural?** — `neural`/`foundation`/`continual` families are neural; `tree`/`baseline` are not. "
        "The MoE/EWC/LatentTSF stack is `moe_ewc_latenttsf` (family `continual`).",
        "- **deps available?** — whether the model's declared dependencies import in this environment. "
        "This is not an execution smoke test for the registry entrypoint.",
        "- **beats baseline?** — against the honest baseline (best-naive for forecasting; "
        "AB411 / Virtual-Beach MLR / station-memory for bacteria), stated straight.",
        "- **claimable?** — safe to put in a pitch/paper. Only `production_candidate` (and, "
        "case-by-case, a promoted `benchmark`) with on-disk evidence qualifies.",
        "",
        "## The fundable headline",
        "`bacteria_hgbt_isotonic` (+ `bacteria_hgbt_spatial`) remains THE fundable, statewide "
        "headline: a calibrated enterococcus nowcast that beats deployed practice and, with the "
        "learned spatial surface, generalizes to beaches it never trained on. A second, distinct "
        "claimable now exists on the marine-HAB frontier: `da_forecast_hgbt` (domoic-acid), which "
        "beats best-naive AND the operational NOAA C-HARM nowcast 8/8 stations -- but on a small "
        "52-event test set and pending a prospective pilot, so it is a frontier result, not the "
        "statewide headline. Everything on the M1-forecasting / neural side remains a benchmark, "
        "research-only, or a documented negative.",
    ]
    if errs:
        lines += ["", "## Validation errors", *[f"- {e}" for e in errs]]
    return "\n".join(lines)


# ---------------------------------------------------------------- bacteria suite (live run)
def _bacteria_candidates():
    """model_id -> (factory | None, status). None factory => unavailable_dependency."""
    cands: dict[str, tuple] = {}

    def _hgbt():
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=600, learning_rate=0.05, l2_regularization=1.0,
            class_weight="balanced", early_stopping=True, validation_fraction=0.1,
            random_state=42)
    cands["bacteria_hgbt_isotonic"] = (_hgbt, "ok")

    if importlib.util.find_spec("xgboost") is not None:
        def _xgb():
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.9,
                colsample_bytree=0.9, reg_lambda=1.0, eval_metric="logloss",
                tree_method="hist", random_state=42)
        cands["bacteria_xgboost"] = (_xgb, "ok")
    else:
        cands["bacteria_xgboost"] = (None, "unavailable_dependency")

    for mid, mod in [("bacteria_lightgbm", "lightgbm"), ("bacteria_catboost", "catboost")]:
        if importlib.util.find_spec(mod) is not None:
            cands[mid] = (_make_optional_factory(mod), "ok")
        else:
            cands[mid] = (None, "unavailable_dependency")
    return cands


def _make_optional_factory(mod: str):
    def _factory():
        if mod == "lightgbm":
            from lightgbm import LGBMClassifier
            return LGBMClassifier(n_estimators=400, learning_rate=0.05, class_weight="balanced",
                                  random_state=42)
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=400, learning_rate=0.05, depth=6, verbose=False,
                                  random_seed=42)
    return _factory


def run_bacteria_suite(obs_path: Path | None = None, rain_dir: str | None = "bacteria_results/rainfall",
                       reveal_lag_days: int = 2, label: str = "enterococcus",
                       stratum: str = "EXCLUDE_SAN_DIEGO") -> dict:
    """Score the incumbent + registered tabular candidates on the honest stratum via the existing
    leakage-safe benchmark. Returns standardized per-model metrics + the operational verdict."""
    from research.bacteria import operational_benchmark as ob

    obs = Path(obs_path) if obs_path else ob.default_obs_path()
    if not obs.exists():
        return {"error": f"bacteria data not found at {obs}; cannot run the live suite"}
    rd = rain_dir if (rain_dir and (_REPO_ROOT / rain_dir).exists()) else None

    results = {}
    for mid, (factory, state) in _bacteria_candidates().items():
        if factory is None:
            results[mid] = {"status": state}
            continue
        try:
            res = ob.run(obs, clf=factory(), rain_dir=rd, reveal_lag_days=reveal_lag_days, label=label)
        except Exception as e:  # a candidate that errors is reported, never crashes the suite
            results[mid] = {"status": "errored", "error": f"{type(e).__name__}: {e}"}
            continue
        s = res["strata"].get(stratum, {})
        models = s.get("models", {})
        cal = models.get("model_hgbt_calibrated", {})
        raw = models.get("model_hgbt", {})
        verdict = s.get("operational_verdict", {}) or {}
        results[mid] = {
            "status": "ok",
            "ap_raw": raw.get("ap"), "ap_calibrated": cal.get("ap"),
            "roc_auc": cal.get("roc_auc"), "brier": cal.get("brier"), "ece_calibrated": cal.get("ece"),
            "beats_best_operational": verdict.get("model_beats_operational_ranking"),
            "beats_vb_mlr": verdict.get("model_beats_vb_mlr"),
            "calibrated_deploy_ready": verdict.get("calibrated_deploy_ready"),
            # suite-level operational gate verdict; registry claimability remains separate.
            "passes_suite_gate": bool(verdict.get("model_beats_operational_ranking")
                                      and verdict.get("calibrated_deploy_ready")),
        }
    return {"stratum": stratum, "label": label, "reveal_lag_days": reveal_lag_days,
            "rain_used": rd is not None, "models": results}


def suite_to_markdown(suite: dict) -> str:
    if "error" in suite:
        return f"## Bacteria suite\n\n_{suite['error']}_"
    lines = [
        "## Bacteria suite (live run)",
        f"- stratum: {suite['stratum']} | label: {suite['label']} | lag: {suite['reveal_lag_days']}d "
        f"| rain: {suite['rain_used']}",
        "",
        "| model | AP (cal) | ROC-AUC | ECE | beats deployed practice | deploy-ready | passes suite gate |",
        "|---|--:|--:|--:|:-:|:-:|:-:|",
    ]
    for mid, r in suite["models"].items():
        if r.get("status") != "ok":
            lines.append(f"| `{mid}` | — | — | — | — | — | {r.get('status')} |")
            continue
        lines.append(
            f"| `{mid}` | {r['ap_calibrated']} | {r['roc_auc']} | {r['ece_calibrated']} | "
            f"{r['beats_best_operational']} | {r['calibrated_deploy_ready']} | {r['passes_suite_gate']} |")
    lines += [
        "",
        "_Two tree families landing within noise of each other is the point: the bacteria ceiling "
        "is the DATA, not the model class. lightgbm/catboost show `unavailable_dependency` until "
        "installed — by design, no heavy deps are added speculatively._",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- model-hardening audit
# These map the registry's internal `status` to the two external vocabularies the model-hardening
# work uses: a public-facing classification (proof report) and a recommended hardening action
# (inventory audit). Derived in one place so the registry stays the single source of truth.
CLASSIFICATION = {
    "production_candidate": "PRODUCTION_CANDIDATE",
    "benchmark": "BENCHMARK_WORKING",
    "research_candidate": "RESEARCH_WORKING_NOT_PROMOTED",
    "research_only_failed_gate": "RESEARCH_ONLY_FAILED_GATE",
    "negative_result": "NEGATIVE_RESULT",
    "baseline": "BASELINE",
    "orphaned": "ORPHANED",
    "retired_claim": "RETIRED_CLAIM",
}
RECOMMENDED_ACTION = {
    "production_candidate": "KEEP_AND_HARDEN",
    "benchmark": "WRAP_AS_BENCHMARK",
    "research_candidate": "DEMOTE_TO_RESEARCH",
    "research_only_failed_gate": "DEMOTE_TO_RESEARCH",
    "negative_result": "RETIRE_CLAIM",
    "baseline": "KEEP_AND_HARDEN",
    "orphaned": "MARK_ORPHANED",
    "retired_claim": "RETIRE_CLAIM",
}


def source_exists(m: dict, repo_root: Path | None = None) -> bool:
    repo_root = repo_root or _REPO_ROOT
    ep = (m.get("entrypoint") or "").strip()
    return bool(ep) and (repo_root / ep).exists()


def evidence_exists(m: dict, repo_root: Path | None = None) -> bool:
    repo_root = repo_root or _REPO_ROOT
    ev = (m.get("evidence") or "").strip()
    return bool(ev) and (repo_root / ev).exists()


def has_smoke_seam(m: dict) -> bool:
    return bool((m.get("smoke_command") or "").strip())


def runnable(m: dict, repo_root: Path | None = None) -> bool:
    """A model 'runs now' only if its source is present, its declared deps import, AND it has a
    cheap, output-isolated smoke seam. Orphaned / zero-shot-rerun-only models report False by
    design — having importable deps is NOT the same as being runnable."""
    return source_exists(m, repo_root) and deps_available(m) and has_smoke_seam(m)


def recommended_action(m: dict, repo_root: Path | None = None) -> str:
    """Status-derived hardening action, escalating to FIX_NOW when a model that is supposed to be
    servable has lost its source file."""
    repo_root = repo_root or _REPO_ROOT
    status = m.get("status")
    if status != "orphaned" and not source_exists(m, repo_root):
        return "FIX_NOW"
    return RECOMMENDED_ACTION.get(status, "MARK_ORPHANED")


def build_inventory(reg: dict, repo_root: Path | None = None) -> list[dict]:
    """One audit row per model — fully derived from the registry + live filesystem/dep checks."""
    repo_root = repo_root or _REPO_ROOT
    rows = []
    for m in reg.get("models", []):
        fam = m.get("family", "?")
        rows.append({
            "model_id": m.get("model_id"),
            "entrypoint": m.get("entrypoint"),
            "task": m.get("task"),
            "family": fam,
            "architecture": m.get("architecture"),
            "neural": fam in {"neural", "foundation", "continual"},
            "dependencies": list(m.get("dependencies") or []),
            "status": m.get("status"),
            "classification": CLASSIFICATION.get(m.get("status"), "ORPHANED"),
            "source_exists": source_exists(m, repo_root),
            "deps_available": deps_available(m),
            "evidence": (m.get("evidence") or "").strip() or None,
            "evidence_exists": evidence_exists(m, repo_root),
            "runnable": runnable(m, repo_root),
            "smoke_command": (m.get("smoke_command") or "").strip() or None,
            "supports_calibration": bool(m.get("supports_calibration")),
            "beats_baseline": m.get("beats_baseline"),
            "baseline": m.get("baseline"),
            "claimable": is_claimable(m, repo_root),
            "output_dir_policy": m.get("output_dir_policy"),
            "blocker": m.get("blockers") or "",
            "recommended_action": recommended_action(m, repo_root),
        })
    return rows


def build_status_matrix(reg: dict, repo_root: Path | None = None) -> dict:
    """Compact machine-readable status matrix used by the proof report."""
    repo_root = repo_root or _REPO_ROOT
    inv = build_inventory(reg, repo_root)
    counts: dict[str, int] = {}
    for r in inv:
        counts[r["classification"]] = counts.get(r["classification"], 0) + 1
    return {
        "registry_valid": not validate(reg, repo_root),
        "total_models": len(inv),
        "claimable": sum(1 for r in inv if r["claimable"]),
        "runnable": sum(1 for r in inv if r["runnable"]),
        "by_classification": counts,
        "models": {r["model_id"]: {
            "classification": r["classification"],
            "status": r["status"],
            "deps_available": r["deps_available"],
            "source_exists": r["source_exists"],
            "runnable": r["runnable"],
            "evidence_exists": r["evidence_exists"],
            "claimable": r["claimable"],
            "calibrated": r["supports_calibration"],
            "recommended_action": r["recommended_action"],
        } for r in inv},
    }


def render_inventory_md(reg: dict, repo_root: Path | None = None) -> str:
    repo_root = repo_root or _REPO_ROOT
    inv = build_inventory(reg, repo_root)
    errs = validate(reg, repo_root)
    lines = [
        "# Model Inventory Audit — Monterey Bay AI Lab",
        "",
        "Machine-generated by `research/model_lab/model_suite.py --audit` from the registry "
        "(`research/model_lab/model_registry.yaml`) plus live filesystem and dependency checks. "
        "Regenerate; do not hand-edit.",
        "",
        f"- registry validation: {'PASS' if not errs else 'FAIL (' + str(len(errs)) + ')'}",
        f"- models inventoried: {len(inv)}",
        f"- runnable now (source + deps + smoke seam): {sum(1 for r in inv if r['runnable'])}",
        f"- claimable now (fail-closed): {sum(1 for r in inv if r['claimable'])}",
        "",
        "## Summary table",
        "",
        "| model_id | family | task | status | source? | deps? | runs now? | evidence? | claimable? | action |",
        "|---|---|---|---|:-:|:-:|:-:|:-:|:-:|---|",
    ]
    yn = lambda b: "yes" if b else "no"
    for r in inv:
        lines.append(
            f"| `{r['model_id']}` | {r['family']} | {r['task']} | {r['status']} | "
            f"{yn(r['source_exists'])} | {yn(r['deps_available'])} | {yn(r['runnable'])} | "
            f"{yn(r['evidence_exists'])} | {'**YES**' if r['claimable'] else 'no'} | "
            f"{r['recommended_action']} |")
    lines += ["", "## Per-model detail", ""]
    for r in inv:
        lines += [
            f"### `{r['model_id']}`",
            f"- **type / family:** {r['family']} ({'neural' if r['neural'] else 'non-neural'})",
            f"- **architecture:** {r['architecture']}",
            f"- **task:** {r['task']}",
            f"- **entrypoint:** `{r['entrypoint']}` (source present: {yn(r['source_exists'])})",
            f"- **dependencies:** {', '.join(r['dependencies']) or 'none'} (available: {yn(r['deps_available'])})",
            f"- **status / classification:** {r['status']} -> {r['classification']}",
            f"- **runs now:** {yn(r['runnable'])}"
            + (f" — `{r['smoke_command']}`" if r['smoke_command'] else " — no cheap smoke seam (heavy / orphaned / zero-shot rerun)"),
            f"- **calibration supported:** {yn(r['supports_calibration'])}",
            f"- **baseline it must beat:** {r['baseline']}",
            f"- **beats baseline:** {r['beats_baseline']}",
            f"- **evidence:** {r['evidence'] or '(none)'} (present: {yn(r['evidence_exists'])})",
            f"- **claimable (fail-closed):** {yn(r['claimable'])}",
            f"- **output-dir policy:** {r['output_dir_policy']}",
            f"- **current blocker:** {r['blocker'] or 'n/a'}",
            f"- **recommended action:** {r['recommended_action']}",
            "",
        ]
    if errs:
        lines += ["## Validation errors", *[f"- {e}" for e in errs], ""]
    return "\n".join(lines)


def write_audit(repo_root: Path | None = None) -> dict:
    """Write the machine-derived hardening deliverables; returns the paths written."""
    repo_root = repo_root or _REPO_ROOT
    reg = load_registry()
    out_dir = repo_root / "reports" / "model_hardening"
    out_dir.mkdir(parents=True, exist_ok=True)
    inv = build_inventory(reg, repo_root)
    matrix = build_status_matrix(reg, repo_root)
    paths = {
        "inventory_md": out_dir / "MODEL_INVENTORY_AUDIT.md",
        "inventory_json": out_dir / "model_inventory_audit.json",
        "status_matrix_json": out_dir / "model_suite_status_matrix.json",
    }
    paths["inventory_md"].write_text(render_inventory_md(reg, repo_root) + "\n", encoding="utf-8")
    paths["inventory_json"].write_text(json.dumps(inv, indent=2), encoding="utf-8")
    paths["status_matrix_json"].write_text(json.dumps(matrix, indent=2), encoding="utf-8")
    return {k: str(v) for k, v in paths.items()}


# ---------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", action="store_true", help="regenerate MODEL_SUITE.md from the registry")
    ap.add_argument("--suite", choices=["bacteria"], help="run a live model suite")
    ap.add_argument("--audit", action="store_true",
                    help="write the model-hardening audit + status matrix to reports/model_hardening/")
    ap.add_argument("--validate", action="store_true", help="validate the registry and exit non-zero on error")
    ap.add_argument("--out", default=str(_REPO_ROOT / "MODEL_SUITE.md"))
    args = ap.parse_args(argv)

    reg = load_registry()
    errs = validate(reg)
    if args.validate:
        for e in errs:
            print(f"[registry] {e}")
        print("registry: " + ("PASS" if not errs else f"FAIL ({len(errs)} errors)"))
        return 1 if errs else 0

    if args.audit:
        written = write_audit()
        for k, v in written.items():
            print(f"wrote {k}: {v}")
        return 1 if errs else 0

    report_md = render_report(reg)
    suite_md = ""
    if args.suite == "bacteria":
        suite = run_bacteria_suite()
        suite_md = suite_to_markdown(suite)
        out_json = _REPO_ROOT / "reports" / "operational_benchmark" / "model_suite_bacteria.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(suite, indent=2), encoding="utf-8")
        try:
            print(suite_md)
        except UnicodeEncodeError:
            print("(wrote model_suite_bacteria.json; console cp1252 cannot print some glyphs)")

    if args.report or args.suite:
        full = report_md + ("\n\n" + suite_md if suite_md else "")
        Path(args.out).write_text(full + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    if not (args.report or args.suite):
        try:
            print(report_md)
        except UnicodeEncodeError:
            print("(registry valid; rerun with --report to write MODEL_SUITE.md)")
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
