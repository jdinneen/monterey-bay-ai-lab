#!/usr/bin/env python3
"""Idle-aware GPU work runner for Monterey Bay AI Lab.

Runs queued model/evaluation jobs only when the RTX 5090 is actually available.
This fills quiet GPU time without stacking jobs into VRAM contention.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from gpu_admission import check_gpu_admission, query_gpu_state
except ImportError:
    from ops.gpu_admission import check_gpu_admission, query_gpu_state


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUEUE = ROOT / "ops" / "gpu_idle_queue.json"
DEFAULT_STATE = ROOT / "ops" / "gpu_idle_state.json"
DEFAULT_LOG_DIR = ROOT / "ops" / "gpu_idle_logs"
DEFAULT_LOCK = ROOT / "ops" / "gpu_idle_runner.lock"
BLOCKER_PATTERNS = (
    "run_production.py",
    "smoke_test_production.py",
    "mbal_neural_forecast.py",
    "mbal_train.py",
    "monte_carlo_production.py",
    "llama-server.exe",
)


@dataclass
class IdleSettings:
    poll_seconds: float = 30.0
    stable_samples: int = 3
    idle_gpu_pct: int = 20
    max_vram_used_pct: float = 0.70
    reserve_mib: int = 4096
    max_job_seconds: int = 6 * 3600


@dataclass
class QueueJob:
    id: str
    label: str
    command: list[str]
    priority: int = 100
    enabled: bool = True
    cwd: str = "."
    request_mib: int = 8192
    max_used_pct: float | None = None
    reserve_mib: int | None = None
    max_seconds: int | None = None
    skip_if_exists: str | None = None
    env: dict[str, str] = field(default_factory=dict)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def hidden_subprocess_kwargs() -> dict:
    if sys.platform != "win32":
        return {}
    return {"creationflags": subprocess.CREATE_NO_WINDOW}


def job_popen_kwargs() -> dict:
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            proc = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
                **hidden_subprocess_kwargs(),
            )
            return str(pid) in proc.stdout
        except Exception:
            return True
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def acquire_runner_lock(path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                first_line = path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
                pid = int(first_line.strip())
            except Exception:
                pid = 0
            if process_exists(pid):
                return False
            try:
                path.unlink()
            except FileNotFoundError:
                continue
            except OSError:
                return False
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as lock:
            lock.write(f"{os.getpid()}\n{utc_now()}\n")
        return True


def release_runner_lock(path: Path) -> None:
    try:
        first_line = path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
        if int(first_line.strip()) != os.getpid():
            return
        path.unlink()
    except Exception:
        return


def terminate_process_tree(proc: subprocess.Popen, timeout: float = 30.0) -> None:
    if proc.poll() is not None:
        return
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                **hidden_subprocess_kwargs(),
            )
        except Exception:
            proc.kill()
    else:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except Exception:
            proc.kill()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        if sys.platform == "win32":
            proc.kill()
        else:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                proc.kill()
        proc.wait(timeout=5)


def load_queue(path: Path) -> tuple[IdleSettings, list[QueueJob]]:
    data = read_json(path, {})
    settings = IdleSettings(**{**IdleSettings().__dict__, **data.get("settings", {})})
    jobs = []
    for raw in data.get("jobs", []):
        merged = {
            "priority": 100,
            "enabled": True,
            "cwd": ".",
            "request_mib": 8192,
            "env": {},
            **raw,
        }
        jobs.append(QueueJob(**merged))
    jobs.sort(key=lambda job: (job.priority, job.id))
    return settings, jobs


def normalize_token(token: str) -> str:
    if token == "$PYTHON":
        return sys.executable
    return token.replace("$ROOT", str(ROOT))


def normalize_command(command: list[str]) -> list[str]:
    return [normalize_token(str(part)) for part in command]


def normalize_path(path: str | None) -> Path | None:
    if not path:
        return None
    value = normalize_token(path)
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def state_jobs(state: dict) -> dict:
    return state.setdefault("jobs", {})


def job_state(state: dict, job_id: str) -> dict:
    return state_jobs(state).setdefault(job_id, {"status": "queued", "attempts": 0})


def is_done_or_skipped(state: dict, job: QueueJob) -> bool:
    js = job_state(state, job.id)
    if js.get("status") in {"done", "skipped"}:
        return True
    skip_path = normalize_path(job.skip_if_exists)
    if skip_path and skip_path.exists():
        js.update({
            "status": "skipped",
            "reason": f"artifact exists: {skip_path}",
            "ended_at": utc_now(),
        })
        return True
    return False


def powershell_process_rows() -> list[dict]:
    if sys.platform != "win32":
        return []
    ps = shutil.which("powershell.exe") or shutil.which("pwsh.exe")
    if not ps:
        return []
    script = r"""
$patterns = 'run_production\.py|smoke_test_production\.py|mbal_neural_forecast\.py|mbal_train\.py|monte_carlo_production\.py|llama-server\.exe'
Get-CimInstance Win32_Process | Where-Object {
  ($_.Name -match 'python|llama-server') -and ($_.CommandLine -match $patterns)
} | Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Compress
"""
    try:
        proc = subprocess.run(
            [ps, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            **hidden_subprocess_kwargs(),
        )
    except Exception:
        return []
    raw = proc.stdout.strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        data = [data]
    return data if isinstance(data, list) else []


def active_blockers() -> list[dict]:
    rows = powershell_process_rows()
    me = os.getpid()
    blockers = []
    for row in rows:
        pid = int(row.get("ProcessId") or 0)
        cmd = str(row.get("CommandLine") or "")
        if pid == me:
            continue
        if "gpu_idle_runner.py" in cmd:
            continue
        blockers.append({"pid": pid, "name": row.get("Name"), "command": cmd})
    return blockers


def current_idle_reason(job: QueueJob, settings: IdleSettings) -> tuple[bool, str]:
    state = query_gpu_state()
    if state is None:
        return False, "nvidia-smi unavailable; refusing GPU idle launch"
    blockers = active_blockers()
    if blockers:
        ids = ", ".join(str(b["pid"]) for b in blockers[:6])
        return False, f"active GPU/model process blockers: {ids}"
    if state.util_pct is not None and state.util_pct > settings.idle_gpu_pct:
        return False, f"GPU compute {state.util_pct}% > idle ceiling {settings.idle_gpu_pct}%"
    if state.used_pct > settings.max_vram_used_pct:
        return False, f"VRAM {state.used_pct * 100:.1f}% > idle ceiling {settings.max_vram_used_pct * 100:.1f}%"
    denial = check_gpu_admission(
        label=job.label,
        request_mib=job.request_mib,
        reserve_mib=job.reserve_mib if job.reserve_mib is not None else settings.reserve_mib,
        max_used_pct=job.max_used_pct if job.max_used_pct is not None else settings.max_vram_used_pct,
    )
    if denial:
        first_reason = next((line.strip(" -") for line in denial.splitlines() if line.strip().startswith("-")), "admission denied")
        return False, first_reason
    return True, f"idle: {state.free_mib} MiB free, GPU {state.util_pct}%"


def wait_until_idle(job: QueueJob, settings: IdleSettings, state: dict, state_path: Path, dry_run: bool) -> bool:
    stable = 0
    while True:
        ok, reason = current_idle_reason(job, settings)
        state.update({
            "updated_at": utc_now(),
            "runner_pid": os.getpid(),
            "mode": "dry_run" if dry_run else "active",
            "current_candidate": job.id,
            "idle_hits": stable,
            "last_idle_check": {"ok": ok, "reason": reason},
        })
        write_json(state_path, state)
        if ok:
            stable += 1
            if stable >= settings.stable_samples:
                return True
        else:
            stable = 0
        time.sleep(settings.poll_seconds)


def run_job(job: QueueJob, settings: IdleSettings, state: dict, state_path: Path, dry_run: bool) -> int:
    DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = DEFAULT_LOG_DIR / f"{stamp}_{job.id}.log"
    cmd = normalize_command(job.command)
    cwd_path = normalize_path(job.cwd) or ROOT
    env = os.environ.copy()
    env.update({str(k): normalize_token(str(v)) for k, v in job.env.items()})
    max_seconds = job.max_seconds or settings.max_job_seconds

    js = job_state(state, job.id)
    js.update({
        "status": "would_run" if dry_run else "running",
        "attempts": int(js.get("attempts") or 0) + (0 if dry_run else 1),
        "started_at": utc_now(),
        "command": cmd,
        "cwd": str(cwd_path),
        "log": str(log_path),
    })
    write_json(state_path, state)

    if dry_run:
        return 0

    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write(f"=== {utc_now()} launching {job.id}: {job.label} ===\n")
        log.write(f"cwd={cwd_path}\n")
        log.write("cmd=" + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd_path),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            **job_popen_kwargs(),
        )
        js["pid"] = proc.pid
        write_json(state_path, state)
        try:
            exit_code = proc.wait(timeout=max_seconds)
        except subprocess.TimeoutExpired:
            terminate_process_tree(proc)
            exit_code = 124
            log.write(f"\n=== killed after max_seconds={max_seconds} ===\n")

    js.update({
        "status": "done" if exit_code == 0 else "failed",
        "exit_code": exit_code,
        "ended_at": utc_now(),
    })
    write_json(state_path, state)
    return exit_code


def select_next_job(jobs: list[QueueJob], state: dict) -> QueueJob | None:
    for job in jobs:
        if not job.enabled:
            continue
        if is_done_or_skipped(state, job):
            continue
        js = job_state(state, job.id)
        if js.get("status") == "failed":
            continue
        return job
    return None


def print_status(queue_path: Path, state_path: Path) -> None:
    settings, jobs = load_queue(queue_path)
    state = read_json(state_path, {})
    gpu = query_gpu_state()
    print(f"queue={queue_path}")
    print(f"state={state_path}")
    if gpu:
        print(f"gpu={gpu.name} util={gpu.util_pct}% vram={gpu.used_mib}/{gpu.total_mib} MiB free={gpu.free_mib} MiB")
    print(f"idle settings: util<={settings.idle_gpu_pct}% vram<={settings.max_vram_used_pct * 100:.1f}% stable={settings.stable_samples}")
    for job in jobs:
        js = job_state(state, job.id)
        print(f"{job.id:28} {js.get('status', 'queued'):10} priority={job.priority} label={job.label}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run queued GPU jobs when the workstation is idle.")
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--once", action="store_true", help="launch at most one queued job, then exit")
    parser.add_argument("--daemon", action="store_true", help="keep launching queued jobs as idle time appears")
    parser.add_argument("--dry-run", action="store_true", help="write what would run without launching")
    parser.add_argument("--status", action="store_true", help="print queue status and exit")
    parser.add_argument("--reset-failed", action="store_true", help="turn failed jobs back to queued")
    args = parser.parse_args(argv)

    if args.status:
        print_status(args.queue, args.state)
        return 0

    settings, jobs = load_queue(args.queue)
    state = read_json(args.state, {"jobs": {}})
    if not acquire_runner_lock(args.lock):
        print(f"another gpu_idle_runner is already active; lock={args.lock}", file=sys.stderr)
        return 2

    try:
        if args.reset_failed:
            for js in state_jobs(state).values():
                if js.get("status") == "failed":
                    js["status"] = "queued"
            write_json(args.state, state)

        while True:
            job = select_next_job(jobs, state)
            if not job:
                state.update({"updated_at": utc_now(), "runner_pid": os.getpid(), "status": "empty"})
                write_json(args.state, state)
                return 0
            wait_until_idle(job, settings, state, args.state, args.dry_run)
            exit_code = run_job(job, settings, state, args.state, args.dry_run)
            if args.once or args.dry_run:
                return exit_code
            if not args.daemon:
                return exit_code
    finally:
        release_runner_lock(args.lock)


if __name__ == "__main__":
    raise SystemExit(main())
