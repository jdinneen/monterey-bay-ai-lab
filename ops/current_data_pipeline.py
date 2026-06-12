#!/usr/bin/env python3
"""Current-data lakehouse and model-evaluation pipeline.

This is the lead orchestration layer for "use all current data" work:

1. Refresh the data-fetch status matrix.
2. Materialize a lakehouse source inventory for every registered/validated source.
3. Run target-compatible model/evaluation jobs against the current local data snapshot.
4. Write dashboard-visible pipeline artifacts.

It does not copy giant source parquet files into a second location. The lakehouse
inventory is the contract: source rows, validation state, curated path, coverage,
and checksum all point to the authoritative local curated/trusted table.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FETCH_STATUS = ROOT / "reports" / "data_fetch" / "fetch_status_matrix.json"
PIPELINE_DIR = ROOT / "reports" / "current_data_pipeline"
PIPELINE_JSON = PIPELINE_DIR / "current_data_pipeline.json"
PIPELINE_MD = PIPELINE_DIR / "current_data_pipeline.md"
LAKEHOUSE_SOURCE_DIR = ROOT / "lakehouse" / "silver" / "source_inventory"
LAKEHOUSE_SOURCE_PARQUET = LAKEHOUSE_SOURCE_DIR / "source_inventory.parquet"
LAKEHOUSE_SOURCE_JSON = LAKEHOUSE_SOURCE_DIR / "source_inventory.json"
DRIVER_MANIFEST = ROOT / "lakehouse" / "silver" / "external_drivers" / "drivers_manifest.json"


@dataclass
class Job:
    key: str
    command: list[str]
    output_paths: list[Path]
    target: str


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run_cmd(command: list[str], *, timeout: int | None = None) -> dict[str, Any]:
    t0 = time.time()
    proc = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    return {
        "command": command,
        "returncode": proc.returncode,
        "seconds": round(time.time() - t0, 2),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def refresh_fetch_status() -> dict[str, Any]:
    result = run_cmd([sys.executable, "ops/data_fetch.py", "report"], timeout=180)
    if not FETCH_STATUS.exists():
        raise RuntimeError(f"fetch status matrix missing after report: {FETCH_STATUS}")
    status = json.loads(FETCH_STATUS.read_text(encoding="utf-8"))
    status["_refresh_command"] = result
    return status


def materialize_source_inventory(status: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for src in status.get("sources", []):
        curated_raw = src.get("curated_path") or ""
        curated = Path(curated_raw) if curated_raw else None
        if curated is not None and not curated.is_absolute():
            curated = ROOT / curated
        curated_exists = bool(curated and curated.exists())
        row = {
            "source": src.get("source"),
            "title": src.get("title"),
            "priority": src.get("priority"),
            "status": src.get("status"),
            "rows": int(src.get("rows") or 0),
            "columns": src.get("columns"),
            "date_min": src.get("date_min"),
            "date_max": src.get("date_max"),
            "curated_path": str(curated.relative_to(ROOT) if curated_exists else curated_raw),
            "curated_exists": curated_exists,
            "sha256": _sha256(curated) if curated is not None else None,
            "duplicate_key_count": src.get("duplicate_key_count"),
        }
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["priority", "source"], na_position="last")
    LAKEHOUSE_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(LAKEHOUSE_SOURCE_PARQUET, index=False)
    LAKEHOUSE_SOURCE_JSON.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")

    ready = df[df["status"].eq("READY_FOR_MODELING")].copy()
    not_ready = df[~df["status"].eq("READY_FOR_MODELING")].copy()
    summary = {
        "sources": int(len(df)),
        "ready_sources": int(len(ready)),
        "not_ready_sources": int(len(not_ready)),
        "ready_rows": int(pd.to_numeric(ready["rows"], errors="coerce").fillna(0).sum()),
        "not_ready": not_ready[["source", "status", "rows"]].to_dict("records"),
        "parquet": str(LAKEHOUSE_SOURCE_PARQUET.relative_to(ROOT)),
        "json": str(LAKEHOUSE_SOURCE_JSON.relative_to(ROOT)),
    }
    return summary


def write_driver_manifest(status: dict[str, Any], inventory: dict[str, Any]) -> dict[str, Any]:
    ready_sources = [
        {
            "source": s.get("source"),
            "rows": int(s.get("rows") or 0),
            "date_min": s.get("date_min"),
            "date_max": s.get("date_max"),
            "curated_path": s.get("curated_path"),
        }
        for s in status.get("sources", [])
        if s.get("status") == "READY_FOR_MODELING"
    ]
    manifest = {
        "schema_version": 2,
        "driver_family": "current_data_inventory",
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "source_inventory": inventory,
        "ready_sources": ready_sources,
        "policy": {
            "data_tables_are_referenced_not_duplicated": True,
            "model_runs_must_use_dashboard_pipeline_manifest": str(PIPELINE_JSON.relative_to(ROOT)),
            "note": "This manifest catalogs all current validated sources; model-specific join code decides target compatibility.",
        },
    }
    DRIVER_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    DRIVER_MANIFEST.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return {"path": str(DRIVER_MANIFEST.relative_to(ROOT)), "ready_sources": len(ready_sources)}


def model_jobs() -> list[Job]:
    return [
        Job(
            key="bacteria_operational_enterococcus_all_current_hooks",
            target="statewide beach bacteria enterococcus exceedance",
            command=[
                sys.executable,
                "research/bacteria/operational_benchmark.py",
                "--label",
                "enterococcus",
                "--reveal-lag-days",
                "2",
                "--rain-dir",
                "bacteria_results/rainfall",
                "--discharge-dir",
                "bacteria_results/discharge",
                "--cdip-dir",
                "bacteria_results/cdip_waves",
                "--tide-stages-dir",
                "bacteria_results/tide_stages",
                "--ocean-dir",
                "data/external_curated",
                "--events-jsonl",
                "reports/operational_benchmark/latest_events.jsonl",
            ],
            output_paths=[
                ROOT / "reports" / "operational_benchmark" / "operational_benchmark.json",
                ROOT / "reports" / "operational_benchmark" / "operational_benchmark.md",
            ],
        ),
        Job(
            key="hab_da_forecast",
            target="marine HAB domoic-acid forecast",
            command=[sys.executable, "research/hab/da_forecast.py"],
            output_paths=[ROOT / "reports" / "hab" / "da_forecast.json"],
        ),
        Job(
            key="hab_charm_comparison",
            target="marine HAB C-HARM benchmark comparison",
            command=[sys.executable, "research/hab/charm_comparison.py"],
            output_paths=[ROOT / "reports" / "hab" / "charm_comparison.json"],
        ),
        Job(
            key="hab_sota_sweep_all_normalized_signals",
            target="marine HAB compatible model sweep",
            command=[sys.executable, "research/hab/hab_sota_sweep.py"],
            output_paths=[ROOT / "reports" / "hab" / "hab_sota_sweep.json"],
        ),
        Job(
            key="bacteria_physical_spatial_covariates",
            target="bacteria physical/spatial ablation",
            command=[
                sys.executable,
                "research/bacteria/physical_spatial_covariates.py",
                "--label",
                "enterococcus",
                "--reveal-lag-days",
                "2",
                "--n-perm",
                "99",
            ],
            output_paths=[ROOT / "reports" / "operational_benchmark" / "physical_spatial_covariates.json"],
        ),
    ]


def run_model_jobs(*, skip_models: bool, timeout: int) -> list[dict[str, Any]]:
    rows = []
    for job in model_jobs():
        if skip_models:
            rows.append({"key": job.key, "target": job.target, "status": "skipped", "reason": "--skip-models"})
            continue
        result = run_cmd(job.command, timeout=timeout)
        outputs = []
        for path in job.output_paths:
            outputs.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "exists": path.exists(),
                    "sha256": _sha256(path),
                    "mtime": path.stat().st_mtime if path.exists() else None,
                }
            )
        status = "passed" if result["returncode"] == 0 and all(o["exists"] for o in outputs) else "failed"
        rows.append({
            "key": job.key,
            "target": job.target,
            "status": status,
            "command": result["command"],
            "seconds": result["seconds"],
            "returncode": result["returncode"],
            "outputs": outputs,
            "stdout_tail": result["stdout_tail"],
            "stderr_tail": result["stderr_tail"],
        })
    rows.append({
        "key": "neural_forecast_lakehouse_models",
        "target": "M1/M2 time-series forecasting",
        "status": "tracked_existing_lakehouse_gold",
        "reason": "Neural model families are regression forecasters with existing lakehouse gold metrics; they are not target-compatible classifiers for the bacteria/HAB benchmark jobs.",
        "metrics_path": "lakehouse/gold/forecast_metrics/metrics.parquet",
    })
    return rows


def write_pipeline_report(status: dict[str, Any], inventory: dict[str, Any], driver_manifest: dict[str, Any], jobs: list[dict[str, Any]]) -> dict[str, Any]:
    model_failures = [j for j in jobs if j.get("status") == "failed"]
    fetch_counts = status.get("counts", {})
    not_ready = inventory.get("not_ready", [])
    report = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "fetch_counts": fetch_counts,
        "source_inventory": inventory,
        "driver_manifest": driver_manifest,
        "model_jobs": jobs,
        "status": "PASS" if not model_failures else "FAIL",
        "open_data_blockers": not_ready,
        "model_failures": [{"key": j["key"], "returncode": j.get("returncode")} for j in model_failures],
    }
    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
    PIPELINE_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    write_markdown(report)
    return report


def write_markdown(report: dict[str, Any]) -> None:
    lines = [
        "# Current Data Pipeline",
        "",
        f"- status: **{report['status']}**",
        f"- generated_at: `{report['generated_at']}`",
        f"- ready sources: {report['source_inventory']['ready_sources']}/{report['source_inventory']['sources']}",
        f"- ready rows: {report['source_inventory']['ready_rows']:,}",
        f"- source inventory: `{report['source_inventory']['parquet']}`",
        f"- driver manifest: `{report['driver_manifest']['path']}`",
        "",
        "## Model Jobs",
        "",
        "| job | target | status | seconds |",
        "|---|---|---|--:|",
    ]
    for job in report["model_jobs"]:
        lines.append(f"| {job['key']} | {job['target']} | {job['status']} | {job.get('seconds', '')} |")
    lines += ["", "## Open Data Blockers", ""]
    blockers = report.get("open_data_blockers") or []
    if not blockers:
        lines.append("No non-ready registered sources.")
    else:
        lines.extend(["| source | status | rows |", "|---|---|--:|"])
        for b in blockers:
            lines.append(f"| {b.get('source')} | {b.get('status')} | {b.get('rows')} |")
    PIPELINE_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--model-timeout", type=int, default=1800)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    status = refresh_fetch_status()
    inventory = materialize_source_inventory(status)
    driver_manifest = write_driver_manifest(status, inventory)
    jobs = run_model_jobs(skip_models=args.skip_models, timeout=args.model_timeout)
    report = write_pipeline_report(status, inventory, driver_manifest, jobs)
    print(json.dumps({
        "status": report["status"],
        "ready_sources": inventory["ready_sources"],
        "sources": inventory["sources"],
        "model_failures": report["model_failures"],
        "open_data_blockers": report["open_data_blockers"],
        "report": str(PIPELINE_JSON.relative_to(ROOT)),
    }, indent=2, default=str))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
