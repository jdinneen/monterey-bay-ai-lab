#!/usr/bin/env python3
"""
MBARI Training Orchestrator (Hardened Version)
Standard: High-Availability Production Standards
Robustly manages GPU training jobs with signal safety, resource guarding, and observability.
"""

import os
import sys
import json
import time
import subprocess
import argparse
import signal
import threading
import queue
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

try:
    import filelock
except ImportError:
    print("Error: 'filelock' package required. Run: pip install filelock")
    sys.exit(1)

# --- Configuration & Policy ---
PROJECT_ROOT = Path(os.environ.get("MBARI_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
OPS_DIR = PROJECT_ROOT / "ops"
RESULTS_DIR = PROJECT_ROOT / "nn_results"
ORCH_LOG_JSON = RESULTS_DIR / "orchestrator_events.jsonl"
GPU_LOCK_FILE = OPS_DIR / "mbari_gpu.lock"

STABLE_ENV = {
    "MBARI_BATCH_SIZE": "12",
    "MBARI_WINDOWS_BATCH": "64",
    "MBARI_INFER_WINDOWS_BATCH": "128",
    "MBARI_VALID_BATCH": "64",
    "MBARI_ACCEL": "gpu",
    "PYTHONUNBUFFERED": "1"  # Crucial for real-time log monitoring
}

REQUIRED_ARTIFACTS = ["summary.json", "leaderboard.csv", "cv_predictions.parquet"]

# --- Global State for Signal Handling ---
running_process: Optional[subprocess.Popen] = None
shutdown_event = threading.Event()

def log_event(event_type: str, **kwargs):
    """Structured JSON logging for observability."""
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        **kwargs
    }
    print(f"[{event['timestamp']}] {event_type}: {json.dumps(kwargs)}")
    
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(ORCH_LOG_JSON, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")

def get_vram_info() -> Dict[str, int]:
    """Returns MiB of free and total VRAM via nvidia-smi."""
    try:
        res = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,nounits,noheader"],
            encoding='utf-8'
        )
        free, total = map(int, res.strip().split(','))
        return {"free": free, "total": total}
    except Exception as e:
        log_event("VRAM_CHECK_FAILED", error=str(e))
        return {"free": 0, "total": 0}

def signal_handler(signum, frame):
    """Graceful shutdown: ensure child processes are killed."""
    signame = signal.Signals(signum).name
    log_event("ORCHESTRATOR_SHUTDOWN_SIGNAL", signal=signame)
    shutdown_event.set()
    if running_process:
        log_event("TERMINATING_CHILD_PROCESS", pid=running_process.pid)
        running_process.terminate()
        try:
            running_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            log_event("KILLING_CHILD_PROCESS", pid=running_process.pid)
            running_process.kill()
    sys.exit(128 + signum)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def check_resume(outdir: Path, force: bool) -> bool:
    """Verifies all required artifacts exist and are non-empty."""
    if force:
        return False
    
    if not outdir.exists():
        return False
        
    for art in REQUIRED_ARTIFACTS:
        p = outdir / art
        if not p.exists() or p.stat().st_size == 0:
            return False
            
    # Optional: could check summary.json content here
    return True

def enqueue_output(out_stream, out_queue):
    """Helper for non-blocking I/O."""
    for line in iter(out_stream.readline, ''):
        out_queue.put(line)
    out_stream.close()

def run_job(job: Dict[str, Any], force: bool, heartbeat_mins: int) -> bool:
    global running_process
    label = job.get("label", "unlabeled")
    model = job.get("model")
    outdir_name = job.get("outdir")
    extra_args = job.get("args", [])
    
    outdir = RESULTS_DIR / outdir_name
    
    if check_resume(outdir, force):
        log_event("JOB_SKIPPED", label=label, reason="already_complete")
        return True

    log_event("JOB_START_REQUEST", label=label, model=model)
    
    env = os.environ.copy()
    env.update(STABLE_ENV)
    
    script_path = PROJECT_ROOT / "mbari_neural_forecast.py"
    cmd = [sys.executable, str(script_path), "--model", model, "--outdir", str(outdir)] + extra_args
    
    raw_log_path = RESULTS_DIR / f"{outdir_name}_raw.log"
    outdir.mkdir(parents=True, exist_ok=True)

    # Locking with timeout to avoid permanent hangs
    lock = filelock.FileLock(GPU_LOCK_FILE, timeout=5)
    try:
        with lock:
            vram = get_vram_info()
            log_event("GPU_LOCK_ACQUIRED", label=label, vram_free_mib=vram["free"])
            
            if vram["free"] < 2000: # Lowered threshold but still checking
                log_event("LOW_VRAM_WARNING", label=label, free=vram["free"])

            start_time = time.time()
            running_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                bufsize=1
            )
            
            # Non-blocking log reading via thread
            q = queue.Queue()
            t = threading.Thread(target=enqueue_output, args=(running_process.stdout, q))
            t.daemon = True
            t.start()
            
            last_heartbeat = time.time()
            with open(raw_log_path, "w", encoding="utf-8") as raw_f:
                while True:
                    if shutdown_event.is_set():
                        return False
                        
                    # Process lines from the queue
                    try:
                        line = q.get_nowait()
                        raw_f.write(line)
                        raw_f.flush()
                        
                        l_upper = line.upper()
                        # Keywords that indicate actual forward progress
                        if any(k in l_upper for k in ['DONE', 'EPOCH', 'WINDOW', 'STEP', 'CROSS_VALIDATION', 'SERIES']):
                            last_heartbeat = time.time()
                    except queue.Empty:
                        if running_process.poll() is not None:
                            break
                        time.sleep(1) # Wait for output
                    
                    # Heartbeat check
                    elapsed_mins = (time.time() - last_heartbeat) / 60
                    if elapsed_mins > heartbeat_mins:
                        log_event("STALL_DETECTED", label=label, stall_duration_mins=elapsed_mins, action="kill")
                        running_process.kill()
                        return False
                    
                    # Periodic VRAM logging every 2 minutes
                    if int(time.time() - start_time) % 120 == 0:
                        vram_periodic = get_vram_info()
                        log_event("HEARTBEAT_VRAM", label=label, free=vram_periodic["free"])

            exit_code = running_process.wait()
            duration_mins = round((time.time() - start_time) / 60, 2)
            running_process = None
            
            if exit_code == 0:
                log_event("JOB_FINISHED_SUCCESS", label=label, duration_mins=duration_mins)
                return True
            else:
                log_event("JOB_FINISHED_FAILURE", label=label, exit_code=exit_code, duration_mins=duration_mins)
                return False

    except filelock.Timeout:
        log_event("GPU_LOCK_TIMEOUT", label=label, timeout_sec=5)
        return False
    except Exception as e:
        log_event("UNEXPECTED_ERROR", label=label, error=str(e))
        return False

def main():
    parser = argparse.ArgumentParser(description="MBARI Hardened Training Orchestrator")
    parser.add_argument("--jobs", help="Path to JSON file defining jobs")
    parser.add_argument("--force", action="store_true", help="Force rerun even if results exist")
    parser.add_argument("--heartbeat", type=int, default=15, help="Minutes of silence before killing a job")
    parser.add_argument("--model", help="Single model run: model name")
    parser.add_argument("--outdir", help="Single model run: output directory name")
    parser.add_argument("--args", nargs=argparse.REMAINDER, help="Additional arguments for single model run")
    
    args = parser.parse_args()
    
    # Setup job list
    jobs = []
    if args.jobs:
        jobs = json.loads(Path(args.jobs).read_text(encoding="utf-8"))
    elif args.model and args.outdir:
        jobs = [{
            "label": args.model,
            "model": args.model,
            "outdir": args.outdir,
            "args": args.args or []
        }]
    else:
        # Defaults if no JSON or model provided
        drv = str(PROJECT_ROOT / "nn_cache/drivers_hourly.parquet")
        man = str(PROJECT_ROOT / "nn_cache/drivers_manifest.json")
        drvArgs = ["--drivers-parquet", drv, "--drivers-manifest", man]
        
        models = ["nhits", "tft", "patchtst", "itransformer", "tsmixerx", "dlinear"]
        for m in models:
            jobs.append({"label": m, "model": m, "outdir": m, "args": []})
            jobs.append({"label": f"{m}_drv", "model": m, "outdir": f"{m}_drv", "args": drvArgs})

    log_event("ORCHESTRATOR_START", total_jobs=len(jobs))
    
    successes = 0
    for job in jobs:
        if shutdown_event.is_set():
            break
        if run_job(job, args.force, args.heartbeat):
            successes += 1
            
    log_event("ORCHESTRATOR_END", successful_jobs=successes, total_jobs=len(jobs))

if __name__ == "__main__":
    main()
