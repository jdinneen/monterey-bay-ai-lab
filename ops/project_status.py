#!/usr/bin/env python
"""Project control tower: one read-only status snapshot for Monterey Bay AI Lab.

Usage:
    python ops/project_status.py [--root PATH] [--run-tests] [--stale-days N]

Writes exactly two files:
    reports/project_status/PROJECT_STATUS.md
    reports/project_status/project_status.json

Read-only by design: it inspects git, the model registry, gate/health report
artifacts, data/fetch outputs, and running python processes. It never mutates
model outputs, data, git state, or processes, and never writes outside
reports/project_status/. It reads existing artifacts instead of re-running
expensive gates/tests; rerun commands are included in the report.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - declared project dependency
    yaml = None

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - optional; degrade to file stats
    pa = None
    pq = None

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_REL = Path("reports") / "project_status"
SCHEMA_VERSION = 1

DEFAULT_STALE_DAYS = 14
VERY_DIRTY_THRESHOLD = 20  # changed+untracked files before worktree is "high risk"

UNKNOWN = "UNKNOWN"

# ---------------------------------------------------------------- declarative sources

DATA_ASSETS = [
    # (asset_id, relative path, kind)  kind: parquet | parquet_dir | csv | json
    ("m1_history", "mbal_history/opendap/m1_history.parquet", "parquet"),
    ("m2_history", "mbal_history/opendap/m2_history.parquet", "parquet"),
    ("noaa_drivers_daily", "mbal_history/noaa/noaa_drivers_daily.parquet", "parquet"),
    ("noaa_ndbc46042", "mbal_history/noaa/noaa_ndbc46042.parquet", "parquet"),
    ("noaa_coops", "mbal_history/noaa/noaa_coops.parquet", "parquet"),
    ("noaa_upwelling", "mbal_history/noaa/noaa_upwelling.parquet", "parquet"),
    ("curated_history", "mbal_pipeline/curated_history", "parquet_dir"),
    ("lakehouse_gold_runs", "lakehouse/gold/forecast_runs", "run_dir"),
    ("lakehouse_gold_metrics", "lakehouse/gold/forecast_metrics/metrics.parquet", "parquet"),
    ("statewide_training_frame", "bacteria_results/statewide/statewide_training_frame.parquet", "parquet"),
    ("statewide_observations", "bacteria_results/statewide/statewide_beach_observations.parquet", "parquet"),
    ("statewide_advisories", "bacteria_results/statewide/statewide_advisories.parquet", "parquet"),
    ("rainfall_grid", "bacteria_results/rainfall/rainfall_grid.parquet", "parquet"),
    ("cdip_waves", "bacteria_results/cdip_waves/cdip_waves.parquet", "parquet"),
    ("discharge_gauge", "bacteria_results/discharge/discharge_gauge.parquet", "parquet"),
    ("tide_stages", "bacteria_results/tide_stages/tide_stages.parquet", "parquet"),
]

FETCHERS = [
    # (fetcher_id, impl path, output paths, proof/manifest paths)
    ("opendap_m1_m2", "mbal_history/opendap/mbal_opendap_fetch.py",
     ["mbal_history/opendap/m1_history.parquet", "mbal_history/opendap/m2_history.parquet"],
     ["mbal_history/opendap/manifest.csv", "mbal_history/opendap/COVERAGE.md"]),
    ("noaa_drivers", "mbal_history/noaa/noaa_drivers_fetch.py",
     ["mbal_history/noaa/noaa_drivers_daily.parquet", "mbal_history/noaa/noaa_ndbc46042.parquet",
      "mbal_history/noaa/noaa_coops.parquet", "mbal_history/noaa/noaa_upwelling.parquet"],
     ["mbal_history/noaa/NOAA_FETCH_AUDIT.md"]),
    ("mur_sst", "mbal_history/noaa/fetch_mur_sst_resumable.py",
     ["mbal_history/noaa/mur_sst_cache"],
     ["mbal_history/noaa/manifest_final.csv"]),
    ("erddap_ndbc", "mbal_history/erddap/mbal_erddap_ndbc_fetch.py",
     [], ["mbal_history/erddap/ERDDAP_FETCH_AUDIT.md"]),
    ("chlorophyll", "ops/fetch_chlorophyll.py",
     ["mbal_history/noaa/chlorophyll_cache"], []),
    ("bacteria_rainfall", "research/bacteria/fetch_rainfall.py",
     ["bacteria_results/rainfall/rainfall_grid.parquet"], ["bacteria_results/rainfall/_fetch.log"]),
    ("bacteria_cdip_waves", "research/bacteria/fetch_cdip_waves.py",
     ["bacteria_results/cdip_waves/cdip_waves.parquet"], []),
    ("bacteria_discharge", "research/bacteria/fetch_discharge.py",
     ["bacteria_results/discharge/discharge_gauge.parquet"], []),
    ("bacteria_tide_stages", "research/bacteria/fetch_tide_stages.py",
     ["bacteria_results/tide_stages/tide_stages.parquet"], []),
    ("statewide_beachwatch", "research/bacteria/fetch_statewide_beachwatch.py",
     ["bacteria_results/statewide/statewide_training_frame.parquet",
      "bacteria_results/statewide/statewide_beach_observations.parquet"],
     ["bacteria_results/statewide/metrics.json"]),
]

GATE_ARTIFACTS = [
    ("release_gate", "release_gate/reports/release_gate_report.json",
     "python release_gate/mbal_release_gate.py"),
    ("sr_manager_gate", "release_gate/reports/sr_manager_gate_report.json",
     "python release_gate/mbal_sr_manager_gate.py"),
    ("promotion_matrix", "release_gate/reports/promotion_summary.json",
     "python release_gate/mbal_promotion_matrix.py"),
    ("data_health", "reports/data_health/data_health_summary.json",
     "python ops/data_health_agent.py"),
]

REGISTRY_REL = "research/model_lab/model_registry.yaml"
MODEL_SUITE_MD_REL = "MODEL_SUITE.md"

# statuses that must never appear as public claims
NON_CLAIMABLE_STATUSES = {
    "research_only_failed_gate", "negative_result", "baseline",
    "retired_claim", "orphaned", "research_candidate", "benchmark",
}

UNTRACKED_IMPORTANT_SUFFIXES = {".py", ".md", ".json", ".yaml", ".yml", ".toml", ".html"}
UNTRACKED_IGNORE_PARTS = {"__pycache__", ".pytest_cache", "node_modules", ".venv", "logs"}


# ---------------------------------------------------------------- small helpers

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


_PUNCT_MAP = {
    "—": "-", "–": "-", "‒": "-", "‑": "-",  # dashes
    "‘": "'", "’": "'", "“": '"', "”": '"',  # smart quotes
    "…": "...", "→": "->", "≥": ">=", "≤": "<=",
    "±": "+/-", "×": "x", "•": "-", " ": " ",
}


def _ascii(text: str) -> str:
    """Force ASCII so the report is safe on cp1252 Windows consoles/editors.

    Common typographic punctuation is mapped to readable ASCII first so it does
    not degrade into '?' replacement characters.
    """
    for src, dst in _PUNCT_MAP.items():
        text = text.replace(src, dst)
    return text.encode("ascii", "replace").decode("ascii")


def file_info(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"exists": path.exists()}
    if not out["exists"]:
        return out
    try:
        stat = path.stat()
        out["size_mb"] = round(stat.st_size / 1e6, 3)
        mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        out["modified_utc"] = _iso(mtime)
        out["age_days"] = round((_now_utc() - mtime).total_seconds() / 86400, 1)
    except OSError:
        pass
    return out


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None


def _run(cmd: list[str], cwd: Path, timeout: int = 30) -> str | None:
    """Run a read-only external command; None on any failure."""
    try:
        proc = subprocess.run(
            cmd, cwd=str(cwd), capture_output=True, text=True,
            timeout=timeout, encoding="utf-8", errors="replace",
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def parquet_summary(path: Path) -> dict[str, Any]:
    """Row count + temporal min/max from parquet metadata only (no full read)."""
    out = file_info(path)
    if not out["exists"]:
        return out
    if pq is None:
        out["rows"] = UNKNOWN
        return out
    try:
        pf = pq.ParquetFile(str(path))
        out["rows"] = pf.metadata.num_rows
        schema = pf.schema_arrow
        names = [f.name for f in schema]
        tcol = next((f.name for f in schema if pa.types.is_temporal(f.type)), None)
        if tcol is not None:
            idx = names.index(tcol)
            lo, hi = None, None
            for rg in range(pf.metadata.num_row_groups):
                stats = pf.metadata.row_group(rg).column(idx).statistics
                if stats is None or not stats.has_min_max:
                    continue
                lo = stats.min if lo is None else min(lo, stats.min)
                hi = stats.max if hi is None else max(hi, stats.max)
            out["time_column"] = tcol
            out["date_min"] = str(lo) if lo is not None else UNKNOWN
            out["date_max"] = str(hi) if hi is not None else UNKNOWN
    except Exception as exc:  # malformed parquet must not kill the report
        out["rows"] = UNKNOWN
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def asset_ready(summary: dict[str, Any], stale_days: float) -> str:
    if not summary.get("exists"):
        return "missing"
    if summary.get("error"):
        return "unreadable"
    rows = summary.get("rows")
    if rows == 0:
        return "empty"
    if summary.get("age_days", 0) > stale_days:
        return "stale"
    if rows == UNKNOWN or rows is None:
        return UNKNOWN
    return "ready"


# ---------------------------------------------------------------- collectors

def collect_git(root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"available": False}
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], root)
    if branch is None:
        out["risk"] = UNKNOWN
        return out
    out["available"] = True
    out["branch"] = branch.strip()
    head = _run(["git", "log", "-1", "--format=%h %cI %s"], root)
    out["last_commit"] = head.strip() if head else UNKNOWN
    porcelain = _run(["git", "status", "--porcelain"], root) or ""
    modified, deleted, untracked = [], [], []
    for line in porcelain.splitlines():
        if len(line) < 4:
            continue
        code, rel = line[:2], line[3:].strip()
        if code == "??":
            untracked.append(rel)
        elif "D" in code:
            deleted.append(rel)
        else:
            modified.append(rel)
    out["modified_count"] = len(modified)
    out["deleted_count"] = len(deleted)
    out["untracked_count"] = len(untracked)
    out["untracked"] = untracked
    total = len(modified) + len(deleted) + len(untracked)
    out["risk"] = ("clean" if total == 0
                   else "moderate" if total < VERY_DIRTY_THRESHOLD
                   else "high")
    worktrees = _run(["git", "worktree", "list", "--porcelain"], root) or ""
    out["worktree_count"] = sum(1 for ln in worktrees.splitlines() if ln.startswith("worktree "))
    return out


def important_untracked(git: dict[str, Any]) -> list[str]:
    files = []
    for rel in git.get("untracked", []):
        p = Path(rel)
        if any(part in UNTRACKED_IGNORE_PARTS for part in p.parts):
            continue
        if rel.endswith("/"):  # untracked directory: report it, contents unknown
            files.append(rel)
            continue
        if p.suffix.lower() in UNTRACKED_IMPORTANT_SUFFIXES:
            files.append(rel)
    return sorted(files)[:60]


def collect_data_assets(root: Path, stale_days: float) -> list[dict[str, Any]]:
    assets = []
    for asset_id, rel, kind in DATA_ASSETS:
        path = root / rel
        entry: dict[str, Any] = {"asset_id": asset_id, "path": rel, "kind": kind}
        if kind == "parquet":
            entry.update(parquet_summary(path))
            entry["status"] = asset_ready(entry, stale_days)
        elif kind == "parquet_dir":
            entry["exists"] = path.is_dir()
            if entry["exists"]:
                files = list(path.rglob("*.parquet"))
                years = sorted({p.name.split("=", 1)[1] for p in path.rglob("year=*") if p.is_dir()})
                entry["parquet_files"] = len(files)
                entry["partitions"] = sorted({p.name for p in path.iterdir() if p.is_dir()})
                if years:
                    entry["date_min"], entry["date_max"] = years[0], years[-1]
                entry["status"] = "ready" if files else "empty"
            else:
                entry["status"] = "missing"
        elif kind == "run_dir":
            entry["exists"] = path.is_dir()
            if entry["exists"]:
                runs = [p for p in path.iterdir() if p.is_dir() and p.name.startswith("run_id=")]
                entry["run_count"] = len(runs)
                entry["status"] = "ready" if runs else "empty"
            else:
                entry["status"] = "missing"
        assets.append(entry)
    return assets


def collect_fetchers(root: Path) -> list[dict[str, Any]]:
    fetchers = []
    for fetcher_id, impl, outputs, proofs in FETCHERS:
        impl_exists = (root / impl).exists()
        out_states = {rel: (root / rel).exists() for rel in outputs}
        proof_states = {rel: (root / rel).exists() for rel in proofs}
        outputs_ok = all(out_states.values()) if outputs else False
        proofs_ok = all(proof_states.values()) and bool(proofs)
        if not impl_exists:
            status = "missing_impl"
        elif outputs_ok and proofs_ok:
            status = "ready"
        elif outputs_ok:
            status = "outputs_present_no_manifest"
        else:
            status = "impl_only"
        fetchers.append({
            "fetcher_id": fetcher_id,
            "impl": impl,
            "impl_exists": impl_exists,
            "outputs": out_states,
            "proofs": proof_states,
            "status": status,
        })
    return fetchers


def collect_models(root: Path) -> dict[str, Any]:
    reg_path = root / REGISTRY_REL
    out: dict[str, Any] = {"registry_path": REGISTRY_REL, "models": []}
    if yaml is None:
        out["error"] = "PyYAML not available"
        return out
    if not reg_path.exists():
        out["error"] = "model registry not found"
        return out
    try:
        data = yaml.safe_load(reg_path.read_text(encoding="utf-8"))
    except Exception as exc:
        out["error"] = f"registry unreadable: {type(exc).__name__}"
        return out
    for m in (data or {}).get("models", []):
        evidence = (m.get("evidence") or "").strip()
        evidence_exists = bool(evidence) and (root / evidence).exists()
        claimable = bool(m.get("claimable"))
        out["models"].append({
            "model_id": m.get("model_id", UNKNOWN),
            "family": m.get("family", UNKNOWN),
            "task": m.get("task", UNKNOWN),
            "status": m.get("status", UNKNOWN),
            "claimable": claimable,
            "evidence": evidence,
            "evidence_exists": evidence_exists,
            "claimable_verified": claimable and evidence_exists,
            "beats_baseline": m.get("beats_baseline", UNKNOWN),
        })
    suite_md = file_info(root / MODEL_SUITE_MD_REL)
    reg_info = file_info(reg_path)
    out["model_suite_md"] = suite_md
    if suite_md.get("exists") and reg_info.get("exists"):
        out["model_suite_md_stale"] = (
            suite_md.get("modified_utc", "") < reg_info.get("modified_utc", "")
        )
    else:
        out["model_suite_md_stale"] = UNKNOWN
    return out


def collect_gates(root: Path, stale_days: float) -> list[dict[str, Any]]:
    gates = []
    for gate_id, rel, rerun in GATE_ARTIFACTS:
        path = root / rel
        entry: dict[str, Any] = {"gate_id": gate_id, "artifact": rel, "rerun_command": rerun}
        entry.update(file_info(path))
        data = read_json(path) if entry["exists"] else None
        if isinstance(data, dict):
            entry["overall_status"] = data.get("overall_status", UNKNOWN)
            entry["generated_at_utc"] = data.get("generated_at_utc", UNKNOWN)
            checks = data.get("checks")
            if isinstance(checks, list):
                entry["check_statuses"] = {
                    str(c.get("name", "?")): str(c.get("status", "?"))
                    for c in checks if isinstance(c, dict)
                }
            if gate_id == "promotion_matrix":
                entry["overall_status"] = "PASS" if data.get("promoted_row_count") else "WARN"
                entry["promoted_row_count"] = data.get("promoted_row_count", UNKNOWN)
                entry["unique_promoted_model_cells"] = data.get(
                    "unique_promoted_model_cell_count", UNKNOWN)
        elif entry["exists"]:
            entry["overall_status"] = "unreadable"
        else:
            entry["overall_status"] = "missing"
        if entry.get("age_days", 0) > stale_days:
            entry["stale"] = True
        gates.append(entry)
    return gates


def collect_tests(root: Path, run_tests: bool) -> dict[str, Any]:
    out: dict[str, Any] = {
        "entrypoint": "ops/run_tests.py",
        "entrypoint_exists": (root / "ops" / "run_tests.py").exists(),
        "command": "python ops/run_tests.py",
        "last_result": UNKNOWN,
        "note": "no persisted test artifact; not run by default (use --run-tests)",
    }
    if run_tests and out["entrypoint_exists"]:
        try:
            proc = subprocess.run(
                [sys.executable, "ops/run_tests.py"], cwd=str(root),
                capture_output=True, text=True, timeout=1800,
                encoding="utf-8", errors="replace",
            )
            out["last_result"] = "PASS" if proc.returncode == 0 else "FAIL"
            out["note"] = "ran via --run-tests in this snapshot"
            tail = (proc.stdout or "").strip().splitlines()[-3:]
            out["output_tail"] = tail
        except (OSError, subprocess.TimeoutExpired) as exc:
            out["last_result"] = "ERROR"
            out["note"] = f"test run failed to execute: {type(exc).__name__}"
    return out


def collect_processes(root: Path) -> list[dict[str, Any]]:
    """List running python processes (read-only) via CIM; [] if unavailable."""
    if os.name != "nt":
        return []
    script = (
        "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
        "Select-Object ProcessId,CreationDate,CommandLine | ConvertTo-Json -Compress"
    )
    raw = _run(["powershell", "-NoProfile", "-Command", script], root, timeout=30)
    if not raw or not raw.strip():
        return []
    try:
        data = json.loads(raw)
    except ValueError:
        return []
    if isinstance(data, dict):
        data = [data]
    procs = []
    for item in data:
        cmdline = str(item.get("CommandLine") or "")
        if "project_status.py" in cmdline or not cmdline:
            continue
        low = cmdline.lower()
        if "pytest" in low or "run_tests" in low:
            kind = "test"
        elif "fetch" in low:
            kind = "fetch"
        elif any(k in low for k in ("train", "forecast", "neural", "moe", "mbal_deep")):
            kind = "model"
        else:
            kind = "other"
        procs.append({
            "pid": item.get("ProcessId"),
            "started": str(item.get("CreationDate") or UNKNOWN),
            "kind": kind,
            "command": cmdline[:240],
        })
    return sorted(procs, key=lambda p: str(p.get("pid")))


def collect_unsafe_artifacts(root: Path, stale_days: float,
                             models: dict[str, Any],
                             gates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unsafe: list[dict[str, Any]] = []

    def add(kind: str, path: str, reason: str) -> None:
        unsafe.append({"kind": kind, "path": path, "reason": reason})

    # smoke-only and quarantined outputs must not be confused with production
    for base_rel in ("nn_results", "reports", "sota_continual_learning"):
        base = root / base_rel
        if not base.is_dir():
            continue
        for entry in sorted(base.iterdir()):
            name = entry.name.lower()
            if "smoke" in name:
                add("smoke_output", f"{base_rel}/{entry.name}",
                    "smoke artifact; not production evidence")
            if name == "_quarantine" and entry.is_dir():
                count = sum(1 for _ in entry.iterdir())
                add("quarantine", f"{base_rel}/{entry.name}",
                    f"{count} quarantined entries; excluded from claims")

    # registry models that must never be claimed
    for m in models.get("models", []):
        if m["status"] in {"orphaned", "research_only_failed_gate", "retired_claim"}:
            add("non_claimable_model", m["model_id"],
                f"registry status {m['status']}; must not appear in public claims")
        if m["claimable"] and not m["evidence_exists"]:
            add("broken_claim", m["model_id"],
                "registry says claimable but evidence file is missing")

    # stale gate artifacts
    for g in gates:
        if g.get("stale"):
            add("stale_gate", g["artifact"],
                f"gate artifact older than {stale_days} days; rerun: {g['rerun_command']}")
        elif g.get("overall_status") in ("missing", "unreadable"):
            add("missing_gate", g["artifact"], f"gate artifact {g['overall_status']}")

    # stale forecast result snapshots
    for rel in ("mbal_forecast_v2_results/model_results.json",):
        info = file_info(root / rel)
        if info.get("exists") and info.get("age_days", 0) > stale_days:
            add("stale_output", rel, f"last modified {info['modified_utc']}")
    return unsafe


def collect_public_claims(root: Path, models: dict[str, Any],
                          gates: list[dict[str, Any]]) -> dict[str, Any]:
    claimable, blocked = [], []
    for m in models.get("models", []):
        if m["claimable_verified"] and m["status"] == "production_candidate":
            claimable.append({
                "model_id": m["model_id"],
                "claim_basis": m["beats_baseline"],
                "evidence": m["evidence"],
            })
        elif m["status"] in NON_CLAIMABLE_STATUSES or not m["claimable_verified"]:
            blocked.append({"model_id": m["model_id"], "status": m["status"]})

    promo = next((g for g in gates if g["gate_id"] == "promotion_matrix"), {})
    m1_claim = UNKNOWN
    cells = promo.get("unique_promoted_model_cells")
    if isinstance(cells, int):
        m1_claim = (f"M1/M2 forecasting is promotable only per target/horizon cell "
                    f"({cells} promoted model-cells); no global forecasting claim.")

    # statewide metric, reported with explicit provenance so it is not mistaken
    # for the registry's LOBO-verified figure (different, more conservative split)
    statewide_metric: dict[str, Any] = {"value": UNKNOWN}
    metrics_rel = "bacteria_results/statewide/metrics.json"
    metrics = read_json(root / metrics_rel)
    if isinstance(metrics, dict):
        node = metrics.get("statewide_with_rainfall")
        if isinstance(node, dict):
            for key in ("roc_auc", "auc"):
                if isinstance(node.get(key), (int, float)):
                    statewide_metric = {
                        "metric": key,
                        "value": round(float(node[key]), 3),
                        "split": "statewide_with_rainfall test split",
                        "source": metrics_rel,
                        "caveat": ("operational/test split; not the LOBO-verified "
                                   "registry figure - use registry evidence for claims"),
                    }
                    break
    return {
        "claimable_models": claimable,
        "not_claimable": blocked,
        "statewide_metric": statewide_metric,
        "forecasting_stance": m1_claim,
        "fetcher_stance": "fetchers/data sources are claimable only with manifest or audit proof",
    }


def derive_next_actions(git: dict[str, Any], models: dict[str, Any],
                        gates: list[dict[str, Any]], tests: dict[str, Any],
                        unsafe: list[dict[str, Any]],
                        fetchers: list[dict[str, Any]]) -> list[str]:
    actions: list[str] = []
    if git.get("risk") == "high":
        total = (git.get("modified_count", 0) + git.get("deleted_count", 0)
                 + git.get("untracked_count", 0))
        actions.append(
            f"Reduce worktree risk: {total} modified/deleted/untracked files on "
            f"branch {git.get('branch', '?')}; commit or branch the in-flight work.")
    warn_gates = [g for g in gates
                  if str(g.get("overall_status")) not in ("PASS", UNKNOWN)
                  or any(s not in ("PASS",) for s in g.get("check_statuses", {}).values())]
    for g in warn_gates[:2]:
        bad = [n for n, s in g.get("check_statuses", {}).items() if s != "PASS"]
        detail = f" (checks: {', '.join(bad)})" if bad else ""
        actions.append(f"Investigate {g['gate_id']} status "
                       f"{g.get('overall_status')}{detail}; rerun: {g['rerun_command']}")
    broken = [u for u in unsafe if u["kind"] == "broken_claim"]
    for u in broken[:1]:
        actions.append(f"Fix registry claim for {u['path']}: evidence file missing.")
    if tests.get("last_result") == UNKNOWN:
        actions.append("Refresh test evidence: python ops/run_tests.py "
                       "(no persisted result; status is UNKNOWN).")
    stale = [u for u in unsafe if u["kind"] in ("stale_gate", "stale_output")]
    for u in stale[:1]:
        actions.append(f"Refresh stale artifact {u['path']}: {u['reason']}")
    if models.get("model_suite_md_stale") is True:
        actions.append("Re-render MODEL_SUITE.md from the registry: "
                       "python research/model_lab/model_suite.py")
    not_ready = [f["fetcher_id"] for f in fetchers
                 if f["status"] in ("impl_only", "outputs_present_no_manifest")]
    if not_ready:
        actions.append("Add manifest/audit proof for fetchers before claiming them ready: "
                       + ", ".join(not_ready[:5]))
    if not actions:
        actions.append("No blocking issues detected; proceed with planned work.")
    return actions[:5]


def overall_status(git: dict[str, Any], gates: list[dict[str, Any]],
                   models: dict[str, Any]) -> str:
    gate_states = [str(g.get("overall_status")) for g in gates]
    if any(s in ("FAIL",) for s in gate_states):
        return "RED"
    if git.get("risk") == "high" or any(s not in ("PASS", UNKNOWN) for s in gate_states):
        return "YELLOW"
    return "GREEN"


# ---------------------------------------------------------------- assembly

def build_status(root: Path, stale_days: float = DEFAULT_STALE_DAYS,
                 run_tests: bool = False, include_processes: bool = True) -> dict[str, Any]:
    git = collect_git(root)
    models = collect_models(root)
    gates = collect_gates(root, stale_days)
    fetchers = collect_fetchers(root)
    tests = collect_tests(root, run_tests)
    unsafe = collect_unsafe_artifacts(root, stale_days, models, gates)
    status: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _iso(_now_utc()),
        "root": str(root),
        "stale_days_threshold": stale_days,
        "git_status_summary": {k: v for k, v in git.items() if k != "untracked"},
        "untracked_important_files": important_untracked(git),
        "data_assets": collect_data_assets(root, stale_days),
        "fetchers": fetchers,
        "models": models,
        "gates": gates,
        "tests": tests,
        "active_processes": collect_processes(root) if include_processes else [],
        "unsafe_artifacts": unsafe,
        "public_claims": collect_public_claims(root, models, gates),
    }
    status["next_actions"] = derive_next_actions(
        status["git_status_summary"], models, gates, tests, unsafe, fetchers)
    status["overall_status"] = overall_status(status["git_status_summary"], gates, models)
    return status


# ---------------------------------------------------------------- markdown rendering

def _md_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return lines


def render_markdown(s: dict[str, Any]) -> str:
    git = s["git_status_summary"]
    lines: list[str] = []
    add = lines.append
    add("# PROJECT STATUS - Monterey Bay AI Lab")
    add("")
    add(f"Generated: {s['generated_at']}  |  Read-only snapshot  |  "
        f"Rerun: `python ops/project_status.py`")
    add("")

    add("## 1. Overall Status")
    add("")
    add(f"**{s['overall_status']}**")
    claims = s["public_claims"]["claimable_models"]
    add(f"- Claimable models (registry + evidence verified): "
        f"{len(claims)} ({', '.join(c['model_id'] for c in claims) or 'none'})")
    gate_bits = [f"{g['gate_id']}={g.get('overall_status')}" for g in s["gates"]]
    add(f"- Gates: {', '.join(gate_bits) or UNKNOWN}")
    add(f"- Worktree risk: {git.get('risk', UNKNOWN)}; tests: {s['tests']['last_result']}")
    add("")

    add("## 2. Worktree Risk")
    add("")
    if git.get("available"):
        add(f"- Branch: `{git.get('branch')}`; last commit: {git.get('last_commit')}")
        add(f"- Modified: {git.get('modified_count')}, deleted: {git.get('deleted_count')}, "
            f"untracked: {git.get('untracked_count')}; risk: **{git.get('risk')}**")
        add(f"- Worktrees: {git.get('worktree_count', UNKNOWN)}")
    else:
        add(f"- git state: {UNKNOWN} (git unavailable)")
    untracked = s["untracked_important_files"]
    if untracked:
        add(f"- Important untracked files ({len(untracked)} shown, code/docs/config only):")
        for rel in untracked[:25]:
            add(f"  - `{rel}`")
        if len(untracked) > 25:
            add(f"  - ... and {len(untracked) - 25} more (see JSON)")
    add("")

    add("## 3. Data Assets")
    add("")
    rows = []
    for a in s["data_assets"]:
        rows.append([
            a["asset_id"], a["status"],
            a.get("rows", a.get("parquet_files", a.get("run_count", "-"))),
            f"{a.get('date_min', '-')} to {a.get('date_max', '-')}"
            if a.get("date_min") else "-",
            a.get("size_mb", "-"), a.get("age_days", "-"),
        ])
    lines += _md_table(["asset", "status", "rows/files", "date range", "size MB", "age d"], rows)
    add("")

    add("## 4. Fetcher Status")
    add("")
    rows = [[f["fetcher_id"], f["status"], f["impl"],
             f"{sum(f['outputs'].values())}/{len(f['outputs'])}" if f["outputs"] else "-",
             f"{sum(f['proofs'].values())}/{len(f['proofs'])}" if f["proofs"] else "none"]
            for f in s["fetchers"]]
    lines += _md_table(["fetcher", "status", "impl", "outputs", "proofs"], rows)
    add("")
    add("'ready' requires implementation + outputs + manifest/audit proof.")
    add("")

    add("## 5. Model Suite")
    add("")
    models = s["models"]
    if models.get("error"):
        add(f"- Registry: {models['error']}")
    rows = [[m["model_id"], m["family"], m["status"],
             "YES" if m["claimable_verified"] else ("BROKEN" if m["claimable"] else "no"),
             m["evidence"] or "-"]
            for m in models.get("models", [])]
    lines += _md_table(["model", "family", "status", "claimable", "evidence"], rows)
    if models.get("model_suite_md_stale") is True:
        add("")
        add("- WARNING: MODEL_SUITE.md is older than the registry; re-render it.")
    add("")

    add("## 6. Gates and Tests")
    add("")
    for g in s["gates"]:
        checks = g.get("check_statuses", {})
        bad = [f"{n}={st}" for n, st in checks.items() if st != "PASS"]
        detail = f"; non-PASS checks: {', '.join(bad)}" if bad else ""
        stale = " (STALE)" if g.get("stale") else ""
        add(f"- **{g['gate_id']}**: {g.get('overall_status')}{stale} "
            f"(artifact: `{g['artifact']}`, generated: {g.get('generated_at_utc', UNKNOWN)})"
            f"{detail}. Rerun: `{g['rerun_command']}`")
    t = s["tests"]
    add(f"- **tests**: {t['last_result']} - {t['note']}. Rerun: `{t['command']}`")
    add("")

    add("## 7. Active Processes")
    add("")
    procs = s["active_processes"]
    if procs:
        rows = [[p["pid"], p["kind"], p["started"], p["command"][:110]] for p in procs]
        lines += _md_table(["pid", "kind", "started", "command"], rows)
    else:
        add("- No long-running python jobs detected (or process inspection unavailable).")
    add("")

    add("## 8. Unsafe or Stale Artifacts")
    add("")
    if s["unsafe_artifacts"]:
        for u in s["unsafe_artifacts"]:
            add(f"- [{u['kind']}] `{u['path']}` - {u['reason']}")
    else:
        add("- None detected.")
    add("")

    add("## 9. Public Claims")
    add("")
    pc = s["public_claims"]
    if pc["claimable_models"]:
        add("Safe to claim now:")
        for c in pc["claimable_models"]:
            add(f"- **{c['model_id']}** - {c['claim_basis']} (evidence: `{c['evidence']}`)")
    else:
        add("- Nothing is currently claimable with verified evidence.")
    sm = pc.get("statewide_metric", {})
    if sm.get("value") != UNKNOWN:
        add(f"- Statewide bacteria {sm['metric']} = {sm['value']} ({sm['split']}, "
            f"source `{sm['source']}`). Caveat: {sm['caveat']}.")
    add(f"- {pc['forecasting_stance']}")
    add(f"- {pc['fetcher_stance']}")
    add(f"- NOT claimable ({len(pc['not_claimable'])} models): "
        + ", ".join(m["model_id"] for m in pc["not_claimable"]))
    add("")

    add("## 10. Top 5 Next Actions")
    add("")
    for i, action in enumerate(s["next_actions"], 1):
        add(f"{i}. {action}")
    add("")
    return _ascii("\n".join(lines))


# ---------------------------------------------------------------- html dashboard

# status token -> semantic color class (matches the repo's existing viz palette)
_GOOD = {"ready", "PASS", "GREEN", "clean"}
_WARN = {"WARN", "YELLOW", "moderate", "stale", "outputs_present_no_manifest",
         "impl_only", "empty", "candidate_split_mismatch"}
_BAD = {"FAIL", "RED", "high", "missing", "unreadable", "broken", "missing_impl",
        "BROKEN", "ERROR"}


def _tone(value: Any) -> str:
    v = str(value)
    if v in _GOOD:
        return "good"
    if v in _WARN:
        return "warn"
    if v in _BAD:
        return "bad"
    if v == UNKNOWN:
        return "dim"
    return "neutral"


def _esc(value: Any) -> str:
    return (str(value).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _pill(value: Any, label: str | None = None) -> str:
    return f'<span class="pill {_tone(value)}">{_esc(label if label is not None else value)}</span>'


def _html_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


_HTML_CSS = """
:root {
  --bg-dark: #07090E;
  --bg-panel: rgba(18, 23, 30, 0.65);
  --border-color: rgba(255, 255, 255, 0.06);
  --text-main: #F1F5F9;
  --text-muted: #94A3B8;
  --accent-blue: #3B82F6;
  --accent-purple: #8B5CF6;
  --accent-cyan: #06B6D4;
  --status-good: #10B981;
  --status-good-bg: rgba(16, 185, 129, 0.12);
  --status-warn: #F59E0B;
  --status-warn-bg: rgba(245, 158, 11, 0.12);
  --status-bad: #EF4444;
  --status-bad-bg: rgba(239, 68, 68, 0.12);
  --status-neutral: #64748B;
  --status-neutral-bg: rgba(100, 116, 139, 0.15);
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background-color: var(--bg-dark);
  background-image: 
    radial-gradient(circle at 10% 0%, rgba(139, 92, 246, 0.15) 0%, transparent 40%),
    radial-gradient(circle at 90% 100%, rgba(6, 182, 212, 0.1) 0%, transparent 40%);
  background-attachment: fixed;
  color: var(--text-main);
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
  padding-bottom: 80px;
}

code {
  font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace;
  background: rgba(0,0,0,0.3);
  padding: 0.2em 0.4em;
  border-radius: 4px;
  font-size: 0.85em;
  color: var(--accent-cyan);
  border: 1px solid rgba(255,255,255,0.05);
}

.wrap { max-width: 1440px; margin: 0 auto; padding: 0 32px; }

header {
  padding: 40px 32px 30px;
  border-bottom: 1px solid var(--border-color);
  margin-bottom: 40px;
  background: linear-gradient(180deg, rgba(7, 9, 14, 0.9) 0%, rgba(7, 9, 14, 0) 100%);
  position: sticky;
  top: 0;
  z-index: 100;
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
}

.header-inner { max-width: 1376px; margin: 0 auto; }

h1 {
  font-size: 2.4rem;
  font-weight: 800;
  letter-spacing: -0.5px;
  margin-bottom: 8px;
  color: #fff;
  display: flex;
  align-items: center;
  gap: 16px;
}

h1::before {
  content: '';
  display: inline-block;
  width: 14px;
  height: 14px;
  border-radius: 50%;
  background: var(--status-good);
  box-shadow: 0 0 16px var(--status-good);
}

.sub { color: var(--text-muted); font-size: 0.85rem; font-weight: 500; display: flex; gap: 16px; align-items: center; }

.cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 20px;
  margin-bottom: 40px;
}

.stat {
  background: var(--bg-panel);
  border: 1px solid var(--border-color);
  border-radius: 16px;
  padding: 24px;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  backdrop-filter: blur(10px);
  position: relative;
  overflow: hidden;
}
.stat::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
}
.stat:hover {
  transform: translateY(-4px);
  box-shadow: 0 20px 40px -10px rgba(0,0,0,0.5);
  border-color: rgba(255,255,255,0.15);
}

.stat .k { color: var(--text-muted); font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px; }
.stat .v { font-size: 2rem; font-weight: 700; color: var(--text-main); display: flex; align-items: baseline; gap: 8px; }

.dashboard-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(640px, 1fr));
  gap: 24px;
  align-items: start;
}

.full-width { grid-column: 1 / -1; }

section {
  background: var(--bg-panel);
  border: 1px solid var(--border-color);
  border-radius: 16px;
  padding: 28px;
  backdrop-filter: blur(10px);
  transition: border-color 0.3s ease;
  height: 100%;
}
section:hover {
  border-color: rgba(255,255,255,0.12);
}

section h2 {
  font-size: 1.15rem;
  font-weight: 600;
  margin-bottom: 24px;
  display: flex;
  align-items: center;
  gap: 12px;
  color: var(--text-main);
  border-bottom: 1px solid rgba(255,255,255,0.04);
  padding-bottom: 16px;
}

section h2 .num {
  background: rgba(139, 92, 246, 0.15);
  color: var(--accent-purple);
  width: 28px;
  height: 28px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 8px;
  font-size: 0.85rem;
  font-weight: 700;
}

table { width: 100%; border-collapse: separate; border-spacing: 0; font-size: 0.85rem; }
th, td { text-align: left; padding: 14px 12px; border-bottom: 1px solid var(--border-color); }
th { color: var(--text-muted); font-weight: 600; text-transform: uppercase; font-size: 0.7rem; letter-spacing: 0.5px; background: rgba(0,0,0,0.2); }
th:first-child { border-top-left-radius: 8px; padding-left: 16px; }
th:last-child { border-top-right-radius: 8px; padding-right: 16px; }
td:first-child { padding-left: 16px; }
td:last-child { padding-right: 16px; }
tbody tr { transition: background-color 0.2s ease; }
tbody tr:hover { background-color: rgba(255,255,255,0.03); }
tbody tr:last-child td { border-bottom: none; }

.pill {
  display: inline-flex; align-items: center; justify-content: center;
  padding: 4px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600;
  letter-spacing: 0.3px; white-space: nowrap;
}
.pill.good { background: var(--status-good-bg); color: var(--status-good); border: 1px solid rgba(16, 185, 129, 0.3); }
.pill.warn { background: var(--status-warn-bg); color: var(--status-warn); border: 1px solid rgba(245, 158, 11, 0.3); }
.pill.bad { background: var(--status-bad-bg); color: var(--status-bad); border: 1px solid rgba(239, 68, 68, 0.3); }
.pill.dim { background: var(--status-neutral-bg); color: var(--text-muted); border: 1px solid rgba(255, 255, 255, 0.1); }
.pill.neutral { background: rgba(59, 130, 246, 0.15); color: var(--accent-blue); border: 1px solid rgba(59, 130, 246, 0.3); }

.bigbadge { font-size: 1.2rem; padding: 6px 16px; border-radius: 12px; letter-spacing: 1px; text-transform: uppercase; }

ul { list-style: none; }
ul li { position: relative; padding-left: 16px; margin-bottom: 8px; color: var(--text-main); font-size: 0.9rem; }
ul li::before { content: ''; position: absolute; left: 0; top: 8px; width: 6px; height: 6px; border-radius: 50%; background: var(--accent-blue); }

ol.actions { counter-reset: a; list-style: none; }
ol.actions li {
  counter-increment: a;
  background: rgba(0,0,0,0.2);
  border: 1px solid var(--border-color);
  border-left: 3px solid var(--accent-blue);
  border-radius: 8px;
  padding: 16px 20px 16px 48px;
  margin-bottom: 12px;
  position: relative;
  transition: transform 0.2s ease, background 0.2s ease;
  line-height: 1.5;
}
ol.actions li:hover { transform: translateX(4px); background: rgba(0,0,0,0.4); }
ol.actions li::before {
  content: counter(a);
  position: absolute; left: 16px; top: 16px;
  color: var(--accent-blue); font-weight: 700; font-size: 0.95rem;
}

.claim {
  border-left: 3px solid var(--status-good);
  padding: 16px 20px;
  background: linear-gradient(90deg, var(--status-good-bg) 0%, transparent 100%);
  border-radius: 0 12px 12px 0;
  margin-bottom: 12px;
}
.claim b { color: var(--status-good); font-size: 1.05rem; display: block; margin-bottom: 4px;}

.muted { color: var(--text-muted); font-size: 0.85rem; }
.flex { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.mt-4 { margin-top: 16px; }
.mb-2 { margin-bottom: 8px; }

/* Scrollbar */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: var(--bg-dark); }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }
"""


def render_html(s: dict[str, Any]) -> str:
    git = s["git_status_summary"]
    pc = s["public_claims"]
    H: list[str] = []
    add = H.append

    add("<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>")
    add("<meta name='viewport' content='width=device-width,initial-scale=1'>")
    add("<title>Monterey Bay AI Lab - Command Center</title>")
    add(f"<style>{_HTML_CSS}</style></head><body>")

    # ---- header + stat cards
    add("<header><div class='header-inner'>")
    add("<h1>Monterey Bay AI Lab</h1>")
    add(f"<div class='sub'><span>Generated {_esc(s['generated_at'])}</span> &middot; <span>Read-only snapshot</span> "
        f"&middot; <code>python ops/project_status.py</code></div>")
    add("</div></header><div class='wrap'>")

    overall = s["overall_status"]
    add("<div class='cards'>")
    add(f"<div class='stat'><div class='k'>System Status</div><div class='v'>"
        f"<span class='bigbadge pill {_tone(overall)}'>{_esc(overall)}</span></div></div>")
    add(f"<div class='stat'><div class='k'>Worktree Risk</div><div class='v'>"
        f"{_pill(git.get('risk', UNKNOWN))}</div></div>")
    add(f"<div class='stat'><div class='k'>Claimable Models</div>"
        f"<div class='v'>{len(pc['claimable_models'])}</div></div>")
    n_ready = sum(1 for a in s["data_assets"] if a.get("status") == "ready")
    add(f"<div class='stat'><div class='k'>Data Assets Ready</div>"
        f"<div class='v'>{n_ready} <span style='font-size: 1rem; color: var(--text-muted); margin-left: 4px;'>/ {len(s['data_assets'])}</span></div></div>")
    gates_pass = sum(1 for g in s["gates"] if g.get("overall_status") == "PASS")
    add(f"<div class='stat'><div class='k'>Gates Passing</div>"
        f"<div class='v'>{gates_pass} <span style='font-size: 1rem; color: var(--text-muted); margin-left: 4px;'>/ {len(s['gates'])}</span></div></div>")
    add(f"<div class='stat'><div class='k'>Test Suite</div><div class='v'>"
        f"{_pill(s['tests']['last_result'])}</div></div>")
    add("</div>")

    add("<div class='dashboard-grid'>")

    def sec(num: int, title: str, inner: str, classes: str = "") -> None:
        cls_str = f" class='{classes}'" if classes else ""
        add(f"<section{cls_str}><h2><span class='num'>{num}</span> {_esc(title)}</h2>{inner}</section>")

    # ---- 10. next actions (Promoted to top)
    items = "".join(f"<li>{_esc(a)}</li>" for a in s["next_actions"])
    sec(1, "Top Actions", f"<ol class='actions'>{items}</ol>", classes="full-width")

    # ---- 2. worktree
    if git.get("available"):
        wt = (f"<div class='flex mb-2'>{_pill(git.get('risk'))}"
              f"<span>branch <code>{_esc(git.get('branch'))}</code></span></div>"
              f"<div class='muted mb-2'>{_esc(git.get('last_commit'))}</div>"
              f"<p class='mb-2'>Modified <b>{git.get('modified_count')}</b> &middot; "
              f"Deleted <b>{git.get('deleted_count')}</b> &middot; "
              f"Untracked <b>{git.get('untracked_count')}</b> &middot; "
              f"Worktrees {git.get('worktree_count', UNKNOWN)}</p>")
        uf = s["untracked_important_files"]
        if uf:
            items_li = "".join(f"<li><code>{_esc(f)}</code></li>" for f in uf[:20])
            extra = f"<li class='muted'>... {len(uf) - 20} more</li>" if len(uf) > 20 else ""
            wt += (f"<div class='muted mt-4 mb-2'>Important untracked files:</div><ul>{items_li}{extra}</ul>")
    else:
        wt = f"<p>{_pill(UNKNOWN)} git unavailable</p>"
    sec(2, "Worktree & Git", wt)

    # ---- 9. public claims
    inner = ""
    if pc["claimable_models"]:
        for c in pc["claimable_models"]:
            inner += (f"<div class='claim'><b>{_esc(c['model_id'])}</b> "
                      f"{_esc(c['claim_basis'])}<br>"
                      f"<span class='muted'>Evidence: <code>{_esc(c['evidence'])}</code></span></div>")
    else:
        inner += "<p class='muted'>Nothing currently claimable with verified evidence.</p>"
    sm = pc.get("statewide_metric", {})
    if sm.get("value") != UNKNOWN:
        inner += (f"<p class='mt-4'>Statewide bacteria <b>{_esc(sm['metric'])} = {_esc(sm['value'])}</b> "
                  f"({_esc(sm['split'])}, source <code>{_esc(sm['source'])}</code>).<br>"
                  f"<span class='muted'>Caveat: {_esc(sm['caveat'])}.</span></p>")
    inner += f"<p class='muted mt-4'>{_esc(pc['forecasting_stance'])}</p>"
    inner += f"<p class='muted'>{_esc(pc['fetcher_stance'])}</p>"
    not_claim = " ".join(_pill("dim", m["model_id"]) for m in pc["not_claimable"])
    inner += (f"<div class='muted mt-4 mb-2'>NOT claimable ({len(pc['not_claimable'])}):</div>"
              f"<div class='flex'>{not_claim}</div>")
    sec(3, "Public Claims", inner)

    # ---- 5. models
    rows = []
    for m in s["models"].get("models", []):
        claim = ("YES" if m["claimable_verified"]
                 else ("BROKEN" if m["claimable"] else "no"))
        rows.append([f"<code>{_esc(m['model_id'])}</code>", _esc(m["family"]),
                     _pill(m["status"]),
                     _pill(claim, claim),
                     f"<code>{_esc(m['evidence'])}</code>" if m["evidence"] else "&mdash;"])
    inner = _html_table(["Model", "Family", "Status", "Claimable", "Evidence"], rows)
    if s["models"].get("model_suite_md_stale") is True:
        inner += "<div class='muted mt-4'>WARNING: MODEL_SUITE.md is older than the registry; re-render it.</div>"
    sec(4, "Model Suite", inner, classes="full-width")

    # ---- 3. data assets
    rows = []
    for a in s["data_assets"]:
        rng = (f"{a.get('date_min')} &rarr; {a.get('date_max')}"
               if a.get("date_min") else "&mdash;")
        rows.append([
            f"<code>{_esc(a['asset_id'])}</code>", _pill(a.get("status")),
            _esc(a.get("rows", a.get("parquet_files", a.get("run_count", "-")))),
            rng, _esc(a.get("size_mb", "-")), _esc(a.get("age_days", "-")),
        ])
    sec(5, "Data Assets",
        _html_table(["Asset", "Status", "Rows/Files", "Date Range", "MB", "Age (d)"], rows), classes="full-width")

    # ---- 4. fetchers
    rows = []
    for f in s["fetchers"]:
        out = f"{sum(f['outputs'].values())}/{len(f['outputs'])}" if f["outputs"] else "&mdash;"
        prf = f"{sum(f['proofs'].values())}/{len(f['proofs'])}" if f["proofs"] else "none"
        rows.append([f"<code>{_esc(f['fetcher_id'])}</code>", _pill(f["status"]),
                     f"<code>{_esc(f['impl'])}</code>", out, prf])
    sec(6, "Fetcher Status",
        _html_table(["Fetcher", "Status", "Implementation", "Outputs", "Proofs"], rows)
        + "<div class='muted mt-4'>'ready' = implementation + outputs + manifest/audit proof.</div>", classes="full-width")

    # ---- 6. gates + tests
    rows = []
    for g in s["gates"]:
        bad = [f"{n}={st}" for n, st in g.get("check_statuses", {}).items() if st != "PASS"]
        detail = ("<br><span class='muted'>non-PASS: " + _esc(", ".join(bad)) + "</span>") if bad else ""
        stale = " (STALE)" if g.get("stale") else ""
        rows.append([f"<code>{_esc(g['gate_id'])}</code>",
                     _pill(g.get("overall_status")) + _esc(stale) + detail,
                     _esc(g.get("generated_at_utc", UNKNOWN)),
                     f"<code>{_esc(g['rerun_command'])}</code>"])
    t = s["tests"]
    rows.append(["<code>tests</code>", _pill(t["last_result"]) + f"<br><span class='muted'>{_esc(t['note'])}</span>",
                 "&mdash;", f"<code>{_esc(t['command'])}</code>"])
    sec(7, "Gates and Tests",
        _html_table(["Gate", "Status", "Generated", "Rerun"], rows), classes="full-width")

    # ---- 7. processes
    procs = s["active_processes"]
    if procs:
        rows = [[_esc(p["pid"]), _pill(p["kind"], p["kind"]), _esc(p["started"]),
                 f"<code>{_esc(p['command'][:120])}</code>"] for p in procs]
        inner = _html_table(["PID", "Kind", "Started", "Command"], rows)
    else:
        inner = "<p class='muted'>No long-running python jobs detected (or inspection unavailable).</p>"
    sec(8, "Active Processes", inner)

    # ---- 8. unsafe
    if s["unsafe_artifacts"]:
        items = "".join(f"<li><span class='k'>[{_esc(u['kind'])}]</span> "
                        f"<code>{_esc(u['path'])}</code><br><span class='muted'>{_esc(u['reason'])}</span></li>"
                        for u in s["unsafe_artifacts"])
        inner = f"<ul class='unsafe'>{items}</ul>"
    else:
        inner = "<p class='muted'>None detected.</p>"
    sec(9, "Unsafe or Stale Artifacts", inner)

    add("</div>") # close dashboard-grid
    add("</div></body></html>")
    return "".join(H)



# ---------------------------------------------------------------- entrypoint

def write_reports(status: dict[str, Any], root: Path) -> tuple[Path, Path, Path]:
    out_dir = root / OUT_REL
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "PROJECT_STATUS.md"
    json_path = out_dir / "project_status.json"
    html_path = out_dir / "PROJECT_STATUS.html"
    md_path.write_text(render_markdown(status), encoding="utf-8")
    json_path.write_text(
        json.dumps(status, indent=2, ensure_ascii=True, default=str) + "\n",
        encoding="utf-8")
    html_path.write_text(render_html(status), encoding="utf-8")
    return md_path, json_path, html_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only project status snapshot.")
    parser.add_argument("--root", type=Path, default=REPO_ROOT,
                        help="repo root (default: repository containing this script)")
    parser.add_argument("--run-tests", action="store_true",
                        help="also run ops/run_tests.py (slow; off by default)")
    parser.add_argument("--stale-days", type=float, default=DEFAULT_STALE_DAYS,
                        help=f"age in days before an artifact is stale (default {DEFAULT_STALE_DAYS})")
    parser.add_argument("--no-processes", action="store_true",
                        help="skip process inspection")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    status = build_status(root, stale_days=args.stale_days,
                          run_tests=args.run_tests,
                          include_processes=not args.no_processes)
    md_path, json_path, html_path = write_reports(status, root)

    print(_ascii(f"overall: {status['overall_status']}"))
    print(_ascii(f"worktree risk: {status['git_status_summary'].get('risk', UNKNOWN)}"))
    claims = status["public_claims"]["claimable_models"]
    print(_ascii("claimable: " + (", ".join(c["model_id"] for c in claims) or "none")))
    for action in status["next_actions"]:
        print(_ascii(f"next: {action}"))
    print(_ascii(f"report: {md_path}"))
    print(_ascii(f"json:   {json_path}"))
    print(_ascii(f"html:   {html_path}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
