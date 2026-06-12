#!/usr/bin/env python3
"""Real-time stack visualization server for Monterey Bay AI Lab runs.

This is intentionally dependency-light: stdlib HTTP server + local probes. It
observes running processes, GPU/VRAM, logs, and lakehouse artifacts, then serves
a browser dashboard from ./static.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent / "static"
SERVER_LOG_DIR = PROJECT_ROOT / "ops" / "stack_viz" / "logs"
CACHE_TTL_S = 1.0

LOG_CANDIDATES = [
    PROJECT_ROOT / "sota_continual_learning" / "production_run.log",
    PROJECT_ROOT / "sota_continual_learning" / "retrain_fixed.log",
    PROJECT_ROOT / "sota_continual_learning" / "training_log.txt",
    PROJECT_ROOT / "ops" / "_gcs_mirror_run.log",
]

ARTIFACT_ROOTS = [
    PROJECT_ROOT / "lakehouse" / "gold" / "forecast_predictions",
    PROJECT_ROOT / "lakehouse" / "gold" / "forecast_metrics",
    PROJECT_ROOT / "lakehouse" / "gold" / "forecast_runs",
    PROJECT_ROOT / "lakehouse" / "gold" / "promotion_matrix",
    PROJECT_ROOT / "lakehouse" / "silver" / "forecast_splits",
    PROJECT_ROOT / "sota_continual_learning" / "output_production" / "checkpoints",
    PROJECT_ROOT / "nn_results",
]


@dataclass
class GpuSnapshot:
    name: str = "No NVIDIA GPU"
    driver: str = ""
    used_mib: int = 0
    total_mib: int = 0
    free_mib: int = 0
    gpu_util_pct: int = 0
    mem_util_pct: int = 0
    temp_c: int = 0
    power_w: float = 0.0
    power_limit_w: float = 0.0

    @property
    def used_pct(self) -> float:
        return round((self.used_mib / self.total_mib) * 100, 1) if self.total_mib else 0.0


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def local_day() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def write_server_log(message: str) -> None:
    try:
        SERVER_LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = SERVER_LOG_DIR / f"server-{local_day()}.log"
        with path.open("a", encoding="utf-8", errors="replace") as f:
            f.write(message.rstrip() + "\n")
    except Exception:
        pass


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
    except Exception:
        return str(path)


def hidden_subprocess_kwargs() -> dict:
    if sys.platform != "win32":
        return {}
    return {"creationflags": subprocess.CREATE_NO_WINDOW}


def run_command(cmd: list[str], timeout: float = 6.0) -> subprocess.CompletedProcess[str] | None:
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


def parse_int(value: str, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return default


def parse_float(value: str, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except Exception:
        return default


def query_gpu() -> GpuSnapshot:
    smi = shutil.which("nvidia-smi")
    if not smi:
        win_smi = Path(os.environ.get("WINDIR", r"C:\Windows")) / "System32" / "nvidia-smi.exe"
        smi = str(win_smi) if win_smi.exists() else None
    if not smi:
        return GpuSnapshot()
    fields = (
        "name,driver_version,memory.used,memory.total,memory.free,"
        "utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit"
    )
    proc = run_command([smi, f"--query-gpu={fields}", "--format=csv,noheader,nounits"], timeout=6)
    if not proc or proc.returncode != 0 or not proc.stdout.strip():
        return GpuSnapshot(name="GPU probe unavailable")
    parts = [p.strip() for p in proc.stdout.strip().splitlines()[0].split(",")]
    if len(parts) < 10:
        return GpuSnapshot(name="GPU parse unavailable")
    return GpuSnapshot(
        name=parts[0],
        driver=parts[1],
        used_mib=parse_int(parts[2]),
        total_mib=parse_int(parts[3]),
        free_mib=parse_int(parts[4]),
        gpu_util_pct=parse_int(parts[5]),
        mem_util_pct=parse_int(parts[6]),
        temp_c=parse_int(parts[7]),
        power_w=parse_float(parts[8]),
        power_limit_w=parse_float(parts[9]),
    )


def powershell_json(script: str, timeout: float = 10.0) -> object:
    if sys.platform != "win32":
        return []
    ps = shutil.which("powershell.exe") or shutil.which("pwsh.exe")
    if not ps:
        return []
    proc = run_command([ps, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script], timeout)
    if not proc or proc.returncode != 0 or not proc.stdout.strip():
        return []
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return []


def query_processes() -> list[dict]:
    script = r"""
$samples = Get-Counter '\GPU Process Memory(*)\Dedicated Usage' -ErrorAction SilentlyContinue |
  Select-Object -ExpandProperty CounterSamples
$mem = @{}
foreach ($s in $samples) {
  if ($s.CookedValue -le 0) { continue }
  if ($s.InstanceName -match 'pid_(\d+)_') {
    $pidText = $matches[1]
    if (-not $mem.ContainsKey($pidText)) { $mem[$pidText] = 0.0 }
    $mem[$pidText] += [double]$s.CookedValue
  }
}
$patterns = 'mbal|run_production|smoke_test_production|monte_carlo|qwen|ollama|llama-server|gcs_mirror|migrate_gold|normalize_|bacteria_ingestion|codex|gemini|claude'
$rows = @()
Get-CimInstance Win32_Process | Where-Object {
  $cmdText = if ($_.CommandLine) { $_.CommandLine } else { '' }
  $_.ProcessId -ne $PID -and
  $cmdText -notmatch 'Get-Counter.*GPU Process Memory' -and
  (
    ($_.Name -match 'python|node|ollama|llama-server|codex|claude|gemini|qwen') -or
    ($cmdText -match $patterns)
  )
} | ForEach-Object {
  $procId = [int]$_.ProcessId
  $p = Get-Process -Id $procId -ErrorAction SilentlyContinue
  $pidText = [string]$procId
  $rows += [pscustomobject]@{
    pid = $procId
    name = $_.Name
    command_line = if ($_.CommandLine) { ($_.CommandLine -replace '\s+', ' ') } else { '' }
    ram_mb = if ($p) { [math]::Round($p.WorkingSet64 / 1MB, 1) } else { 0 }
    cpu_s = if ($p -and $p.CPU) { [math]::Round($p.CPU, 1) } else { 0 }
    vram_mib = if ($mem.ContainsKey($pidText)) { [math]::Round($mem[$pidText] / 1MB, 1) } else { 0 }
  }
}
$rows | Sort-Object vram_mib, ram_mb -Descending | Select-Object -First 80 | ConvertTo-Json -Compress
"""
    data = powershell_json(script)
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return []
    return [classify_process(row) for row in data]


def classify_process(row: dict) -> dict:
    cmd = str(row.get("command_line") or "")
    name = str(row.get("name") or "")
    text = f"{name} {cmd}".lower()
    kind = "support"
    stage = "desktop"
    model = ""
    priority = 5
    if "run_production.py" in text or "smoke_test_production.py" in text:
        kind, stage, model, priority = "continual", "training", "continual_learner", 1
    elif "mbal_neural_forecast.py" in text:
        kind, stage, priority = "neuralforecast", "training", 1
        m = re.search(r"--model\s+([^\s]+)", cmd)
        model = m.group(1) if m else "neuralforecast"
    elif "mbal_train.py" in text:
        kind, stage, model, priority = "orchestrator", "training", "sweep", 1
    elif "monte_carlo" in text:
        kind, stage, model, priority = "monte_carlo", "uncertainty", "monte_carlo", 2
    elif "ollama" in text or "llama-server" in text:
        kind, stage, model, priority = "llm", "model_server", "qwen/ollama", 2
    elif "qwen" in text:
        kind, stage, model, priority = "agent", "agent", "qwen", 2
    elif "gcs_mirror" in text or "migrate_gold" in text or "normalize_" in text:
        kind, stage, model, priority = "data_ops", "lakehouse", "data_ops", 3
    elif "bacteria_ingestion" in text:
        kind, stage, model, priority = "ingest", "bronze", "bacteria_ingestion", 3
    elif "codex" in text or "claude" in text or "gemini" in text:
        kind, stage, model, priority = "assistant", "agent", name.replace(".exe", ""), 4

    row = {
        "pid": int(row.get("pid") or 0),
        "name": name,
        "kind": kind,
        "stage": stage,
        "model": model,
        "priority": priority,
        "ram_mb": parse_float(str(row.get("ram_mb", 0))),
        "cpu_s": parse_float(str(row.get("cpu_s", 0))),
        "vram_mib": parse_float(str(row.get("vram_mib", 0))),
        "command": cmd,
    }
    return row


def tail_file(path: Path, max_bytes: int = 128_000) -> str:
    if not path.exists() or not path.is_file():
        return ""
    size = path.stat().st_size
    with path.open("rb") as fh:
        if size > max_bytes:
            fh.seek(size - max_bytes)
        raw = fh.read()
    return raw.decode("utf-8", errors="replace")


def latest_run_logs(limit: int = 8) -> list[Path]:
    root = PROJECT_ROOT / "nn_results"
    if not root.exists():
        return []
    logs = list(root.glob("*/run.log"))
    logs.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    return logs[:limit]


def event_stage(path: Path, line: str) -> str:
    text = f"{path} {line}".lower()
    if "cross_validation" in text or "rmse" in text or "leaderboard" in text:
        return "evaluation"
    if "checkpoint" in text or "step " in text or "loss" in text:
        return "training"
    if "lakehouse" in text or "run_id=" in text or "split_id=" in text:
        return "gold"
    if "gcs" in text or "mirror" in text or "migrate" in text:
        return "export"
    if "vram" in text or "gpu" in text:
        return "gpu"
    return "system"


def event_severity(line: str) -> str:
    upper = line.upper()
    if "ERROR" in upper or "FAILED" in upper or "DENIED" in upper or "OOM" in upper:
        return "error"
    if "WARNING" in upper or "GUARD" in upper or "REFUSING" in upper:
        return "warn"
    if "DONE" in upper or "COMPLETE" in upper or "SAVED" in upper:
        return "done"
    return "info"


def parse_logs() -> tuple[list[dict], dict]:
    files = [p for p in LOG_CANDIDATES if p.exists()] + latest_run_logs()
    events: list[dict] = []
    metrics: dict = {
        "latest_step": None,
        "latest_loss": None,
        "latest_speed": None,
        "latest_vram_line": None,
        "latest_rmse": None,
        "latest_run_id": None,
        "training_complete": False,
    }
    for path in files:
        text = tail_file(path)
        if not text:
            continue
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in lines[-160:]:
            step = re.search(r"Step\s+(\d+)/(\d+).*?Loss:\s*([0-9.eE+-]+).*?Speed:\s*([0-9.eE+-]+)", line)
            if step:
                metrics["latest_step"] = {"step": int(step.group(1)), "total": int(step.group(2))}
                metrics["latest_loss"] = parse_float(step.group(3))
                metrics["latest_speed"] = parse_float(step.group(4))
            vram = re.search(r"VRAM:\s*([0-9.]+)/([0-9.]+)\s*GB\s*\(([0-9.]+)%\)", line)
            if vram:
                metrics["latest_vram_line"] = {
                    "used_gb": parse_float(vram.group(1)),
                    "total_gb": parse_float(vram.group(2)),
                    "pct": parse_float(vram.group(3)),
                }
            run_id = re.search(r"run_id=([A-Za-z0-9_]+)", line)
            if run_id:
                metrics["latest_run_id"] = run_id.group(1)
            if "mean RMSE by horizon:" in line:
                metrics["latest_rmse"] = line.split("mean RMSE by horizon:", 1)[-1].strip()
            if "Training complete" in line or line.startswith("DONE "):
                metrics["training_complete"] = True
            if any(token in line for token in (
                "Step ", "VRAM:", "cross_validation", "DONE ", "Training complete",
                "Checkpoint", "lakehouse run_id", "GPU_ADMISSION", "ERROR", "WARNING",
                "mean RMSE by horizon", "Starting", "Loaded",
            )):
                events.append({
                    "source": rel(path),
                    "stage": event_stage(path, line),
                    "severity": event_severity(line),
                    "text": line[-420:],
                    "mtime": path.stat().st_mtime,
                })
    events.sort(key=lambda e: (e["mtime"], e["source"]), reverse=True)
    return events[:80], metrics


def scan_artifacts(limit: int = 40) -> list[dict]:
    out: list[dict] = []
    for root in ARTIFACT_ROOTS:
        if not root.exists():
            continue
        try:
            files = [p for p in root.rglob("*") if p.is_file()]
        except Exception:
            continue
        for path in files:
            try:
                st = path.stat()
            except OSError:
                continue
            if st.st_size == 0:
                continue
            suffix = path.suffix.lower().lstrip(".") or "file"
            kind = "metric" if "metric" in str(path).lower() else "prediction" if "prediction" in str(path).lower() else suffix
            out.append({
                "path": rel(path),
                "name": path.name,
                "kind": kind,
                "size_bytes": st.st_size,
                "size_label": size_label(st.st_size),
                "mtime": st.st_mtime,
                "age_s": max(0, time.time() - st.st_mtime),
            })
    out.sort(key=lambda row: row["mtime"], reverse=True)
    return out[:limit]


def size_label(size: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{size} B"


def lakehouse_summary() -> dict:
    roots = {
        "silver_splits": PROJECT_ROOT / "lakehouse" / "silver" / "forecast_splits",
        "gold_runs": PROJECT_ROOT / "lakehouse" / "gold" / "forecast_runs",
        "gold_metrics": PROJECT_ROOT / "lakehouse" / "gold" / "forecast_metrics",
        "gold_predictions": PROJECT_ROOT / "lakehouse" / "gold" / "forecast_predictions",
        "promotion_matrix": PROJECT_ROOT / "lakehouse" / "gold" / "promotion_matrix",
    }
    summary = {}
    for key, root in roots.items():
        count = 0
        total = 0
        latest = 0.0
        if root.exists():
            for path in root.rglob("*"):
                if path.is_file():
                    try:
                        st = path.stat()
                    except OSError:
                        continue
                    count += 1
                    total += st.st_size
                    latest = max(latest, st.st_mtime)
        summary[key] = {
            "files": count,
            "bytes": total,
            "size_label": size_label(total),
            "latest_age_s": max(0, time.time() - latest) if latest else None,
        }
    promo = PROJECT_ROOT / "lakehouse" / "gold" / "promotion_matrix" / "promotion_summary.json"
    if promo.exists():
        try:
            summary["promotion_summary"] = json.loads(promo.read_text(encoding="utf-8"))
        except Exception:
            summary["promotion_summary"] = {}
    return summary


def idle_queue_summary() -> dict:
    queue_path = PROJECT_ROOT / "ops" / "gpu_idle_queue.json"
    state_path = PROJECT_ROOT / "ops" / "gpu_idle_state.json"
    if not queue_path.exists():
        return {"configured": False, "jobs": []}
    try:
        queue = json.loads(queue_path.read_text(encoding="utf-8"))
    except Exception:
        queue = {"jobs": []}
    try:
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    except Exception:
        state = {}
    states = state.get("jobs", {})
    jobs = []
    for raw in queue.get("jobs", []):
        jid = raw.get("id", raw.get("label", "?"))
        js = states.get(jid, {})
        jobs.append({
            "id": jid,
            "label": raw.get("label", jid),
            "priority": raw.get("priority", 100),
            "enabled": raw.get("enabled", True),
            "status": js.get("status", "queued" if raw.get("enabled", True) else "disabled"),
            "reason": js.get("reason") or state.get("last_idle_check", {}).get("reason", ""),
            "log": js.get("log", ""),
        })
    jobs.sort(key=lambda row: (row["priority"], row["id"]))
    queued = sum(1 for row in jobs if row["enabled"] and row["status"] in {"queued", "would_run"})
    running = sum(1 for row in jobs if row["status"] == "running")
    return {
        "configured": True,
        "updated_at": state.get("updated_at"),
        "mode": state.get("mode"),
        "last_idle_check": state.get("last_idle_check"),
        "queued": queued,
        "running": running,
        "jobs": jobs,
    }


def node_status(active: bool, warn: bool = False, done: bool = False) -> str:
    if warn:
        return "warn"
    if active:
        return "active"
    if done:
        return "done"
    return "idle"


def build_stack(gpu: GpuSnapshot, processes: list[dict], metrics: dict, lakehouse: dict, artifacts: list[dict]) -> dict:
    active_training = [p for p in processes if p["stage"] in {"training", "uncertainty"} and p["vram_mib"] > 256]
    active_data = [p for p in processes if p["stage"] in {"bronze", "lakehouse"}]
    active_llm = [p for p in processes if p["stage"] in {"model_server", "agent"} and p["vram_mib"] > 512]
    recent_artifact = artifacts[0] if artifacts else None
    recent_gold = bool(recent_artifact and recent_artifact["age_s"] < 600 and "lakehouse/gold" in recent_artifact["path"])
    vram_warn = bool(gpu.total_mib and (gpu.free_mib < 4096 or gpu.used_pct >= 85))

    promotion = lakehouse.get("promotion_summary") or {}
    promoted = promotion.get("promote") or promotion.get("promoted") or promotion.get("promote_count")
    matrix_rows = promotion.get("rows") or promotion.get("matrix_rows")

    nodes = [
        {
            "id": "sources",
            "label": "Sources",
            "sub": "MBAL, NOAA, CenCOOS, local history",
            "status": node_status(bool(active_data)),
            "meter": 72,
            "metrics": [
                {"label": "raw lakehouse", "value": lakehouse.get("silver_splits", {}).get("size_label", "-")},
                {"label": "active ingest", "value": str(len(active_data))},
            ],
        },
        {
            "id": "bronze",
            "label": "Bronze",
            "sub": "source registry, raw snapshots, mirror logs",
            "status": node_status(bool(active_data)),
            "meter": 56 if active_data else 34,
            "metrics": [
                {"label": "data ops", "value": str(len(active_data))},
                {"label": "latest export", "value": age_label(PROJECT_ROOT / "ops" / "_gcs_mirror_run.log")},
            ],
        },
        {
            "id": "silver",
            "label": "Silver",
            "sub": "QC, masks, splits, leakage controls",
            "status": node_status(recent_gold),
            "meter": 62 if recent_gold else 48,
            "metrics": [
                {"label": "split files", "value": str(lakehouse.get("silver_splits", {}).get("files", 0))},
                {"label": "split store", "value": lakehouse.get("silver_splits", {}).get("size_label", "-")},
            ],
        },
        {
            "id": "training",
            "label": "Training",
            "sub": "continual learner, MoE, NeuralForecast",
            "status": node_status(bool(active_training), warn=vram_warn),
            "meter": min(100, gpu.gpu_util_pct),
            "metrics": [
                {"label": "gpu util", "value": f"{gpu.gpu_util_pct}%"},
                {"label": "models", "value": str(len(active_training))},
            ],
        },
        {
            "id": "evaluation",
            "label": "Evaluation",
            "sub": "walk-forward CV, RMSE, uncertainty",
            "status": node_status(bool(metrics.get("latest_rmse") or metrics.get("latest_step")), done=bool(metrics.get("training_complete"))),
            "meter": step_pct(metrics),
            "metrics": [
                {"label": "step", "value": step_label(metrics)},
                {"label": "loss", "value": metric_label(metrics.get("latest_loss"))},
            ],
        },
        {
            "id": "gold",
            "label": "Gold",
            "sub": "predictions, metrics, run manifests",
            "status": node_status(recent_gold, done=bool(lakehouse.get("gold_runs", {}).get("files"))),
            "meter": 78 if recent_gold else 52,
            "metrics": [
                {"label": "gold files", "value": str(lakehouse.get("gold_runs", {}).get("files", 0))},
                {"label": "run id", "value": short_id(metrics.get("latest_run_id"))},
            ],
        },
        {
            "id": "claims",
            "label": "Claim Gate",
            "sub": "promotion matrix, claim boundaries",
            "status": node_status(bool(promotion), done=bool(promotion)),
            "meter": 70 if promotion else 30,
            "metrics": [
                {"label": "matrix rows", "value": str(matrix_rows or "-")},
                {"label": "promoted", "value": str(promoted or "-")},
            ],
        },
    ]

    flow_base = 0.25 + (gpu.gpu_util_pct / 100) * 0.65
    if not active_training and not recent_gold:
        flow_base = 0.18
    flows = []
    for left, right in zip(nodes, nodes[1:]):
        intensity = flow_base
        if right["id"] == "training" and active_training:
            intensity = max(intensity, 0.9)
        if right["id"] in {"gold", "claims"} and recent_gold:
            intensity = max(intensity, 0.78)
        flows.append({"from": left["id"], "to": right["id"], "intensity": round(intensity, 2)})

    return {
        "nodes": nodes,
        "flows": flows,
        "active_training_count": len(active_training),
        "active_llm_count": len(active_llm),
        "vram_warn": vram_warn,
    }


def step_pct(metrics: dict) -> int:
    step = metrics.get("latest_step")
    if not step:
        return 0
    total = max(1, int(step.get("total") or 1))
    return min(100, int((int(step.get("step") or 0) / total) * 100))


def step_label(metrics: dict) -> str:
    step = metrics.get("latest_step")
    if not step:
        return "-"
    return f"{step.get('step')}/{step.get('total')}"


def metric_label(value: object) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.4g}"
    except Exception:
        return str(value)


def short_id(value: object) -> str:
    text = str(value or "")
    return text[:12] if text else "-"


def age_label(path: Path) -> str:
    if not path.exists():
        return "-"
    age = max(0, time.time() - path.stat().st_mtime)
    if age < 60:
        return f"{int(age)}s"
    if age < 3600:
        return f"{int(age // 60)}m"
    return f"{age / 3600:.1f}h"


class StateCache:
    def __init__(self) -> None:
        self.created = 0.0
        self.payload: dict | None = None
        self.gpu_history: list[dict] = []

    def get(self) -> dict:
        now = time.time()
        if self.payload is not None and now - self.created < CACHE_TTL_S:
            return self.payload
        gpu = query_gpu()
        self.gpu_history.append({
            "t": now,
            "gpu_util_pct": gpu.gpu_util_pct,
            "mem_util_pct": gpu.mem_util_pct,
            "used_pct": gpu.used_pct,
        })
        self.gpu_history = [row for row in self.gpu_history if now - row["t"] <= 60]
        processes = query_processes()
        events, metrics = parse_logs()
        artifacts = scan_artifacts()
        lakehouse = lakehouse_summary()
        idle_queue = idle_queue_summary()
        stack = build_stack(gpu, processes, metrics, lakehouse, artifacts)
        gpu_rollup = rolling_gpu(self.gpu_history, now)
        payload = {
            "timestamp": now_iso(),
            "project_root": str(PROJECT_ROOT),
            "gpu": {**asdict(gpu), "used_pct": gpu.used_pct, **gpu_rollup},
            "processes": sorted(processes, key=lambda p: (p["priority"], -p["vram_mib"], -p["ram_mb"]))[:60],
            "events": events,
            "metrics": metrics,
            "artifacts": artifacts,
            "lakehouse": lakehouse,
            "idle_queue": idle_queue,
            "stack": stack,
            "admission": {
                "max_used_pct": float(os.environ.get("MBAL_GPU_GUARD_MAX_USED_PCT", "0.85")),
                "reserve_mib": int(float(os.environ.get("MBAL_GPU_GUARD_RESERVE_MIB", "4096"))),
                "status": "tight" if stack["vram_warn"] else "open",
            },
        }
        self.payload = payload
        self.created = now
        return payload


def rolling_gpu(history: list[dict], now: float) -> dict:
    recent_30 = [row for row in history if now - row["t"] <= 30]
    recent_60 = [row for row in history if now - row["t"] <= 60]

    def avg(rows: list[dict], key: str) -> float:
        return round(sum(float(row.get(key) or 0) for row in rows) / len(rows), 1) if rows else 0.0

    def maxv(rows: list[dict], key: str) -> float:
        return round(max((float(row.get(key) or 0) for row in rows), default=0.0), 1)

    return {
        "gpu_util_avg_30s": avg(recent_30, "gpu_util_pct"),
        "gpu_util_max_30s": maxv(recent_30, "gpu_util_pct"),
        "mem_util_avg_30s": avg(recent_30, "mem_util_pct"),
        "samples_30s": len(recent_30),
        "gpu_util_avg_60s": avg(recent_60, "gpu_util_pct"),
        "gpu_util_max_60s": maxv(recent_60, "gpu_util_pct"),
    }


CACHE = StateCache()


class StackVizHandler(BaseHTTPRequestHandler):
    server_version = "MBALStackViz/0.1"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/state":
            self.send_json(CACHE.get())
            return
        if parsed.path == "/api/state.js":
            params = parse_qs(parsed.query)
            callback = params.get("callback", ["StackVizState"])[0]
            callback = re.sub(r"[^A-Za-z0-9_.$]", "", callback) or "StackVizState"
            self.send_js(f"{callback}({json.dumps(CACHE.get(), separators=(',', ':'), ensure_ascii=False)});")
            return
        if parsed.path in {"/", "/index.html"}:
            self.send_index()
            return
        path = unquote(parsed.path.lstrip("/"))
        static_path = (STATIC_DIR / path).resolve()
        if not str(static_path).startswith(str(STATIC_DIR.resolve())):
            self.send_error(403)
            return
        if static_path.exists() and static_path.is_file():
            self.send_static(static_path)
            return
        self.send_error(404)

    def log_message(self, fmt: str, *args: object) -> None:
        message = "[%s] %s" % (self.log_date_time_string(), fmt % args)
        write_server_log(message)
        stream = getattr(sys, "stderr", None)
        if stream:
            try:
                stream.write(message + "\n")
            except Exception:
                pass

    def send_json(self, payload: dict) -> None:
        raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def send_index(self) -> None:
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        state = json.dumps(CACHE.get(), separators=(",", ":"), ensure_ascii=False)
        injection = f"<script>window.__STACK_STATE__={state};</script>"
        html = html.replace("<!--STACK_STATE-->", injection)
        raw = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def send_static(self, path: Path) -> None:
        raw = path.read_bytes()
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def send_js(self, text: str) -> None:
        raw = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/javascript; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the live Monterey Bay AI Lab stack visualizer.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    server = ThreadingHTTPServer((args.host, args.port), StackVizHandler)
    print(f"Stack visualization running at http://{args.host}:{args.port}")
    print(f"Project root: {PROJECT_ROOT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping stack visualization.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
