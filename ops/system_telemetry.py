#!/usr/bin/env python3
"""Durable workstation telemetry recorder for Monterey Bay AI Lab.

Writes daily JSONL snapshots that can be analyzed later for GPU/CPU/IO/model
optimization. Retention is forever by default; set --retention-days to prune.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import psutil
except ImportError:  # pragma: no cover - local workstation has psutil
    psutil = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_ROOT = ROOT / "ops" / "telemetry"
STACK_SERVER_PATH = ROOT / "ops" / "stack_viz" / "server.py"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().replace(microsecond=0).isoformat()


def load_stack_server() -> Any:
    spec = importlib.util.spec_from_file_location("mbal_stack_viz_server", STACK_SERVER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load stack server from {STACK_SERVER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def hidden_subprocess_kwargs() -> dict[str, int]:
    if sys.platform != "win32":
        return {}
    return {"creationflags": subprocess.CREATE_NO_WINDOW}


def run_command(cmd: list[str], timeout: float = 8.0) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            **hidden_subprocess_kwargs(),
        )
    except Exception:
        return None


def nvidia_smi_path() -> str | None:
    exe = shutil.which("nvidia-smi")
    if exe:
        return exe
    win_smi = Path(os.environ.get("WINDIR", r"C:\Windows")) / "System32" / "nvidia-smi.exe"
    return str(win_smi) if win_smi.exists() else None


def parse_csv_line(line: str) -> list[str]:
    return [part.strip() for part in line.split(",")]


def query_gpu_extended() -> dict[str, Any]:
    smi = nvidia_smi_path()
    if not smi:
        return {"available": False}
    fields = [
        "pstate",
        "clocks.current.graphics",
        "clocks.current.memory",
        "clocks.max.graphics",
        "clocks.max.memory",
        "fan.speed",
        "pcie.link.gen.current",
        "pcie.link.width.current",
        "utilization.encoder",
        "utilization.decoder",
    ]
    proc = run_command([smi, f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"], timeout=8)
    if not proc or proc.returncode != 0 or not proc.stdout.strip():
        return {"available": False, "error": (proc.stderr.strip() if proc else "nvidia-smi failed")}
    values = parse_csv_line(proc.stdout.strip().splitlines()[0])
    return {"available": True, **dict(zip(fields, values))}


def query_compute_apps() -> list[dict[str, Any]]:
    smi = nvidia_smi_path()
    if not smi:
        return []
    fields = "pid,process_name,used_memory"
    proc = run_command([smi, f"--query-compute-apps={fields}", "--format=csv,noheader,nounits"], timeout=8)
    if not proc or proc.returncode != 0 or not proc.stdout.strip():
        return []
    rows = []
    for line in proc.stdout.strip().splitlines():
        parts = parse_csv_line(line)
        if len(parts) >= 3:
            rows.append({"pid": parts[0], "process_name": parts[1], "used_memory_mib": parts[2]})
    return rows


def disk_partitions() -> list[dict[str, Any]]:
    if not psutil:
        return []
    rows = []
    for part in psutil.disk_partitions(all=False):
        try:
            usage = psutil.disk_usage(part.mountpoint)
        except Exception:
            continue
        rows.append({
            "device": part.device,
            "mountpoint": part.mountpoint,
            "fstype": part.fstype,
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
            "used_pct": usage.percent,
        })
    return rows


def io_counter_dict(counter: Any) -> dict[str, int]:
    if counter is None:
        return {}
    fields = getattr(counter, "_fields", ())
    return {field: int(getattr(counter, field)) for field in fields}


def collect_host_metrics() -> dict[str, Any]:
    if not psutil:
        return {"psutil_available": False}
    cpu_percent = psutil.cpu_percent(interval=0.2)
    per_cpu = psutil.cpu_percent(interval=None, percpu=True)
    virtual = psutil.virtual_memory()
    swap = psutil.swap_memory()
    disk_io = psutil.disk_io_counters(perdisk=True)
    net_io = psutil.net_io_counters(pernic=True)
    freq = psutil.cpu_freq()

    return {
        "psutil_available": True,
        "boot_time": datetime.fromtimestamp(psutil.boot_time(), timezone.utc).replace(microsecond=0).isoformat(),
        "cpu": {
            "logical_count": psutil.cpu_count(logical=True),
            "physical_count": psutil.cpu_count(logical=False),
            "percent": cpu_percent,
            "per_cpu_percent": per_cpu,
            "freq_mhz": {
                "current": getattr(freq, "current", None),
                "min": getattr(freq, "min", None),
                "max": getattr(freq, "max", None),
            } if freq else {},
        },
        "memory": {
            "total_bytes": virtual.total,
            "available_bytes": virtual.available,
            "used_bytes": virtual.used,
            "used_pct": virtual.percent,
        },
        "swap": {
            "total_bytes": swap.total,
            "used_bytes": swap.used,
            "free_bytes": swap.free,
            "used_pct": swap.percent,
            "sin_bytes": swap.sin,
            "sout_bytes": swap.sout,
        },
        "disks": {
            "partitions": disk_partitions(),
            "io": {name: io_counter_dict(counter) for name, counter in (disk_io or {}).items()},
        },
        "network": {
            "io": {name: io_counter_dict(counter) for name, counter in (net_io or {}).items()},
        },
    }


def process_rollup(processes: list[dict[str, Any]]) -> dict[str, Any]:
    stages: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0, "ram_mb": 0.0, "vram_mib": 0.0})
    kinds: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0, "ram_mb": 0.0, "vram_mib": 0.0})
    for proc in processes:
        stage = str(proc.get("stage") or "unknown")
        kind = str(proc.get("kind") or "unknown")
        ram = float(proc.get("ram_mb") or 0)
        vram = float(proc.get("vram_mib") or 0)
        stages[stage]["count"] += 1
        stages[stage]["ram_mb"] += ram
        stages[stage]["vram_mib"] += vram
        kinds[kind]["count"] += 1
        kinds[kind]["ram_mb"] += ram
        kinds[kind]["vram_mib"] += vram
    return {
        "by_stage": {k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in stages.items()},
        "by_kind": {k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in kinds.items()},
    }


def trim_text(value: Any, limit: int = 260) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def compact_process(proc: dict[str, Any]) -> dict[str, Any]:
    return {
        "pid": proc.get("pid"),
        "name": proc.get("name"),
        "kind": proc.get("kind"),
        "stage": proc.get("stage"),
        "model": proc.get("model"),
        "ram_mb": proc.get("ram_mb"),
        "cpu_s": proc.get("cpu_s"),
        "vram_mib": proc.get("vram_mib"),
        "command": trim_text(proc.get("command")),
    }


def compact_lakehouse(lakehouse: dict[str, Any]) -> dict[str, Any]:
    compact = {key: value for key, value in lakehouse.items() if key != "promotion_summary"}
    promo = lakehouse.get("promotion_summary") or {}
    if isinstance(promo, dict):
        keep = (
            "rows",
            "promoted_row_count",
            "unique_promoted_model_cell_count",
            "unique_promoted_target_horizon_count",
            "status_counts",
        )
        compact_promo = {key: promo.get(key) for key in keep if key in promo}
        if compact_promo:
            compact["promotion_summary"] = compact_promo
    return compact


def compact_idle_queue(idle_queue: dict[str, Any]) -> dict[str, Any]:
    jobs = []
    source_jobs = idle_queue.get("jobs", []) if isinstance(idle_queue, dict) else []
    for job in source_jobs:
        jobs.append({
            "id": job.get("id"),
            "label": job.get("label"),
            "status": job.get("status"),
            "reason": trim_text(job.get("reason"), 180),
            "log": job.get("log"),
        })
    return {
        "configured": idle_queue.get("configured"),
        "updated_at": idle_queue.get("updated_at"),
        "mode": idle_queue.get("mode"),
        "queued": idle_queue.get("queued"),
        "running": idle_queue.get("running"),
        "last_idle_check": idle_queue.get("last_idle_check"),
        "jobs": jobs,
    }


def compact_stack_state(state: dict[str, Any]) -> dict[str, Any]:
    stack = state.get("stack") or {}
    return {
        "timestamp": state.get("timestamp"),
        "project_root": state.get("project_root"),
        "gpu": state.get("gpu"),
        "admission": state.get("admission"),
        "processes": [compact_process(proc) for proc in state.get("processes", [])[:60]],
        "metrics": state.get("metrics"),
        "lakehouse": compact_lakehouse(state.get("lakehouse") or {}),
        "idle_queue": compact_idle_queue(state.get("idle_queue") or {}),
        "stack": {
            "active_training_count": stack.get("active_training_count"),
            "active_llm_count": stack.get("active_llm_count"),
            "vram_warn": stack.get("vram_warn"),
            "nodes": stack.get("nodes"),
        },
    }


def make_snapshot(stack_cache: Any) -> dict[str, Any]:
    state = stack_cache.get()
    compact_state = compact_stack_state(state)
    return {
        "schema_version": 1,
        "captured_at": utc_now_iso(),
        "project_root": str(ROOT),
        "stack_state": compact_state,
        "host": collect_host_metrics(),
        "gpu_extended": query_gpu_extended(),
        "compute_apps": query_compute_apps(),
        "rollups": {
            "processes": process_rollup(compact_state.get("processes", [])),
        },
    }


def write_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, separators=(",", ":"), ensure_ascii=False, default=str))
        f.write("\n")


def log_path(log_root: Path, captured_at: datetime) -> Path:
    day = captured_at.strftime("%Y-%m-%d")
    return log_root / day / "system_telemetry.jsonl"


def write_latest(log_root: Path, payload: dict[str, Any]) -> None:
    latest = log_root / "latest_system_telemetry.json"
    tmp = latest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(latest)


def apply_retention(log_root: Path, retention_days: int) -> None:
    if retention_days <= 0 or not log_root.exists():
        return
    cutoff = time.time() - retention_days * 86400
    for child in log_root.iterdir():
        if not child.is_dir():
            continue
        try:
            if child.stat().st_mtime < cutoff:
                shutil.rmtree(child)
        except Exception:
            continue


def print_status(log_root: Path) -> None:
    latest = log_root / "latest_system_telemetry.json"
    print(f"log_root={log_root}")
    if latest.exists():
        data = json.loads(latest.read_text(encoding="utf-8"))
        gpu = data.get("stack_state", {}).get("gpu", {})
        host = data.get("host", {})
        mem = host.get("memory", {})
        print(f"latest={data.get('captured_at')}")
        print(f"gpu={gpu.get('gpu_util_pct')}% vram={gpu.get('used_mib')}/{gpu.get('total_mib')} MiB")
        print(f"cpu={host.get('cpu', {}).get('percent')}% ram={mem.get('used_pct')}%")
    else:
        print("latest=missing")
    days = sorted([p.name for p in log_root.iterdir() if p.is_dir()]) if log_root.exists() else []
    print(f"days={', '.join(days[-7:]) if days else '-'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record durable workstation telemetry for optimization.")
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--interval-seconds", type=float, default=10.0)
    parser.add_argument("--once", action="store_true", help="capture one snapshot and exit")
    parser.add_argument("--daemon", action="store_true", help="keep recording until stopped")
    parser.add_argument("--status", action="store_true", help="print latest telemetry status and exit")
    parser.add_argument("--retention-days", type=int, default=0, help="0 keeps logs forever")
    args = parser.parse_args(argv)

    if args.status:
        print_status(args.log_root)
        return 0

    stack_server = load_stack_server()
    stack_cache = stack_server.StateCache()

    while True:
        captured = utc_now()
        snapshot = make_snapshot(stack_cache)
        write_jsonl(log_path(args.log_root, captured), snapshot)
        write_latest(args.log_root, snapshot)
        apply_retention(args.log_root, args.retention_days)
        if args.once or not args.daemon:
            return 0
        time.sleep(max(1.0, args.interval_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
