#!/usr/bin/env python3
"""
MBARI neural-forecast training ORCHESTRATOR (single source of truth).

This wraps the stable trainer ``mbari_neural_forecast.py`` (one model per invocation,
writes ``<outdir>/summary.json`` on success) and codifies every run-flow lesson learned
the hard way on the RTX 5090 (sm_120, 32 GB, torch 2.11+cu128) workstation:

  1. SINGLE-GPU MUTEX        -> one GPU training subprocess at a time (lockfile w/ live-pid
                                check + stale-lock steal). Kills the concurrent-GPU
                                ``cudaErrorUnknown`` (cudaErrorUnknown) root cause.
  2. POWER-STATE GUARD       -> verify AC sleep/hibernate idle == Never; optionally enforce;
                                hold an OS wake-lock (SetThreadExecutionState) for the run so
                                a misconfigured policy cannot S3-sleep mid-train.
  3. GPU PREFLIGHT + PROFILE -> parse nvidia-smi; wait while another process holds VRAM; pick
                                a memory-safe batch PROFILE (safe/medium/large) instead of the
                                trainer's heavy defaults.
  4. OOM/CUDA/STALL BACKOFF  -> per-job subprocess w/ log tail; CUDA/OOM error -> retry at the
                                next-smaller profile -> finally fall back to CPU; no-output for
                                a stall-timeout -> kill + retry smaller. Capped retries.
  5. IDEMPOTENT RESUME       -> skip jobs whose summary.json already matches the current
                                cache_version (unless --force); restart continues where it left.
  6. DETACHED + OBSERVABLE   -> runs independent of the interactive terminal; per-job logs +
                                a single nn_results/run_manifest.json; ``--status`` progress table.

DO NOT import this from / modify mbari_neural_forecast.py -- the trainer is treated as a
stable black box invoked over subprocess. All timeouts are injectable so the logic is unit
testable on CPU/mock without ever touching the GPU.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional

# --------------------------------------------------------------------------------------
# Paths & constants (match mbari_neural_forecast.py)
# --------------------------------------------------------------------------------------
ROOT = Path(os.environ.get("MBARI_PROJECT_ROOT", Path(__file__).resolve().parent)).resolve()
TRAINER = ROOT / "mbari_neural_forecast.py"
CACHE_DIR = ROOT / "nn_cache"
RESULTS_DIR = ROOT / "nn_results"
LOCK_PATH = CACHE_DIR / ".gpu.lock"
MANIFEST_PATH = RESULTS_DIR / "run_manifest.json"
CACHE_VERSION = "v3_past_only_fill_origin_observed_missingness"  # keep in sync with the trainer

PYTHON = os.environ.get("MBARI_PYTHON", sys.executable)

# --------------------------------------------------------------------------------------
# 3. Memory-aware batch profiles. Ordered smallest -> largest. "safe" is the proven-stable
#    profile (the trainer's heavy defaults thrashed the 32 GB card to 31.8 GB and stalled).
# --------------------------------------------------------------------------------------
BATCH_PROFILES: dict[str, dict[str, int]] = {
    "safe":   {"MBARI_BATCH_SIZE": 12, "MBARI_WINDOWS_BATCH": 64,  "MBARI_INFER_WINDOWS_BATCH": 128, "MBARI_VALID_BATCH": 64},
    "medium": {"MBARI_BATCH_SIZE": 16, "MBARI_WINDOWS_BATCH": 128, "MBARI_INFER_WINDOWS_BATCH": 256, "MBARI_VALID_BATCH": 128},
    "large":  {"MBARI_BATCH_SIZE": 24, "MBARI_WINDOWS_BATCH": 256, "MBARI_INFER_WINDOWS_BATCH": 512, "MBARI_VALID_BATCH": 256},
}
# Backoff order: the largest -> smaller. A job starting at "large" can retry "medium" then "safe".
PROFILE_ORDER = ["large", "medium", "safe"]


def smaller_profile(name: str) -> Optional[str]:
    """Return the next-smaller profile name, or None if already smallest ('safe')."""
    if name not in PROFILE_ORDER:
        # Unknown profile: treat as already at the floor.
        return None
    i = PROFILE_ORDER.index(name)
    return PROFILE_ORDER[i + 1] if i + 1 < len(PROFILE_ORDER) else None


# Signatures in trainer output that mean "GPU/CUDA blew up -> retry smaller / fall back to CPU".
CUDA_ERROR_PATTERNS = [
    "CUDA error",
    "CUDA out of memory",
    "cudaErrorUnknown",
    "AcceleratorError",
    "torch.cuda.OutOfMemoryError",
    "OutOfMemoryError",
    "CUBLAS_STATUS",
    "device-side assert",
    "an illegal memory access",
]

CAPACITY_ERROR_PATTERNS = [
    "DefaultCPUAllocator: not enough memory",
    "not enough memory: you tried to allocate",
    "paging file is too small",
    "Insufficient system resources exist to complete the requested service",
    "std::bad_alloc",
]

# Full-scale driver exogenous data creates very large training windows. Today we
# saw this risk across multiple model families, including production-shaped
# driver/quantile runs, so every driver-enabled job is bounded by default.
DEFAULT_DRIVER_GUARD_TAIL_WEEKS = 104.0


def is_cuda_error(text: str) -> bool:
    """True if the captured subprocess output contains a CUDA/OOM failure signature."""
    return any(p.lower() in text.lower() for p in CUDA_ERROR_PATTERNS)


def is_capacity_error(text: str) -> bool:
    """True when a failure is host RAM/pagefile capacity, not a retryable CUDA backoff."""
    return any(p.lower() in text.lower() for p in CAPACITY_ERROR_PATTERNS)


def is_driver_job(job: "Job") -> bool:
    return "--drivers-parquet" in job.extra_args or "--drivers-manifest" in job.extra_args


def resolve_cli_path(value: str, root: Path = ROOT) -> str:
    """Resolve local file arguments relative to the project root, preserving URI-like paths (s3://, gs://, etc.)."""
    if "://" in value:
        return value
    p = Path(value)
    return str(p if p.is_absolute() else root / p)


def resolve_known_path_args(args: list[str], root: Path = ROOT) -> list[str]:
    """Resolve path-valued trainer args so job JSON can be portable."""
    path_flags = {"--drivers-parquet", "--drivers-manifest"}
    out: list[str] = []
    i = 0
    while i < len(args):
        out.append(args[i])
        if args[i] in path_flags and i + 1 < len(args):
            out.append(resolve_cli_path(args[i + 1], root))
            i += 2
            continue
        i += 1
    return out


def driver_guard_reason(job: "Job") -> Optional[str]:
    """Return a safety reason if a driver-heavy job is disabled by default."""
    if not is_driver_job(job):
        return None
    return (
        f"driver-heavy {job.model} is guarded by default; previous full-scale runs "
        "attempted host allocations above 100 GiB. The orchestrator will use a "
        "bounded recent-history shape unless --allow-unsafe-driver-models is set."
    )


def with_driver_guard_shape(job: "Job", tail_weeks: float) -> "Job":
    """Return a job with a bounded recent-history shape for guarded driver models."""
    if tail_weeks <= 0 or driver_guard_reason(job) is None:
        return job
    args = list(job.extra_args)
    if "--tail-weeks" not in args:
        args += ["--tail-weeks", f"{tail_weeks:g}"]
    return Job(job.label, job.model, job.outdir, args, job.profile)


# ======================================================================================
# 2. Power-state guard
# ======================================================================================
# powercfg GUID aliases for AC idle sleep / hibernate timeouts.
_STANDBYIDLE = ("SCHEME_CURRENT", "SUB_SLEEP", "STANDBYIDLE")
_HIBERNATEIDLE = ("SCHEME_CURRENT", "SUB_SLEEP", "HIBERNATEIDLE")
_AC_INDEX_RE = re.compile(r"Current AC Power Setting Index:\s*(0x[0-9a-fA-F]+|\d+)")


def parse_ac_index(powercfg_output: str) -> Optional[int]:
    """Extract the 'Current AC Power Setting Index' value (seconds; 0 == Never) from a
    ``powercfg /query`` block. Returns None if not found. Pure: unit-testable on samples."""
    m = _AC_INDEX_RE.search(powercfg_output)
    if not m:
        return None
    raw = m.group(1)
    return int(raw, 16) if raw.lower().startswith("0x") else int(raw)


def _powercfg_query(setting: tuple[str, str, str]) -> str:
    try:
        r = subprocess.run(["powercfg", "/query", *setting],
                           capture_output=True, text=True, timeout=20)
        return r.stdout or ""
    except Exception as e:  # pragma: no cover - environment dependent
        return f"<powercfg query failed: {e}>"


def check_power_state(enforce: bool, logfn: Callable[[str], None]) -> dict:
    """Verify AC sleep & hibernate idle timeouts are 0 (Never). If not and ``enforce`` is set,
    apply 0 via ``powercfg /change``. Read-only otherwise (just warn loudly).
    Returns a dict describing what was found / changed."""
    report: dict = {}
    for label, setting, change_key in (
        ("standby", _STANDBYIDLE, "standby-timeout-ac"),
        ("hibernate", _HIBERNATEIDLE, "hibernate-timeout-ac"),
    ):
        out = _powercfg_query(setting)
        idx = parse_ac_index(out)
        report[label] = {"ac_seconds": idx}
        if idx == 0:
            logfn(f"[power] {label} AC idle timeout = Never (0) OK")
            continue
        logfn(f"[power] *** WARNING *** {label} AC idle timeout = {idx} s (NOT Never). "
              f"A misconfigured policy could sleep mid-run.")
        if enforce:
            try:
                subprocess.run(["powercfg", "/change", change_key, "0"],
                               capture_output=True, text=True, timeout=20, check=True)
                report[label]["enforced_to"] = 0
                logfn(f"[power] enforced {change_key} = 0 (Never)")
            except Exception as e:  # pragma: no cover
                report[label]["enforce_error"] = str(e)
                logfn(f"[power] failed to enforce {change_key}: {e}")
        else:
            logfn(f"[power] run with --enforce-power to set {change_key}=0 automatically")
    return report


# SetThreadExecutionState flags (winbase.h). ES_CONTINUOUS keeps the state until released;
# ES_SYSTEM_REQUIRED forces the system awake (no S3 sleep) for the calling thread's lifetime.
_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001
_ES_AWAYMODE_REQUIRED = 0x00000040


class WakeLock:
    """Hold an OS-level wake lock for the run's duration (context manager).

    On Windows calls SetThreadExecutionState(ES_CONTINUOUS|ES_SYSTEM_REQUIRED). On any other
    platform (or if ctypes is unavailable) it is a harmless no-op so the orchestrator and its
    tests run anywhere. Released on __exit__ so power policy returns to normal."""

    def __init__(self, logfn: Callable[[str], None] = print, enabled: bool = True):
        self._logfn = logfn
        self._enabled = enabled and sys.platform == "win32"
        self._held = False

    def __enter__(self) -> "WakeLock":
        if not self._enabled:
            return self
        try:
            res = ctypes.windll.kernel32.SetThreadExecutionState(
                _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED | _ES_AWAYMODE_REQUIRED)
            if res == 0:
                # AWAYMODE not always supported; retry without it.
                res = ctypes.windll.kernel32.SetThreadExecutionState(
                    _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED)
            self._held = res != 0
            self._logfn("[power] wake-lock acquired (system stays awake for the run)"
                        if self._held else "[power] wake-lock request returned 0 (continuing)")
        except Exception as e:  # pragma: no cover
            self._logfn(f"[power] wake-lock unavailable: {e}")
        return self

    def __exit__(self, *exc) -> None:
        if self._held:
            try:
                ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
                self._logfn("[power] wake-lock released")
            except Exception:  # pragma: no cover
                pass
        self._held = False


# ======================================================================================
# 1. Single-GPU mutex (lockfile)
# ======================================================================================
def pid_is_alive(pid: int) -> bool:
    """True if a process with this pid is currently running. Cross-platform best-effort."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                                 capture_output=True, text=True, timeout=15).stdout
            return str(pid) in out
        except Exception:  # pragma: no cover
            return True  # fail safe: assume alive rather than steal a live lock
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:  # pragma: no cover
        return False


class GpuLock:
    """File-based single-GPU mutex.

    The lockfile stores ``{"pid", "host", "start", "label"}``. Acquisition:
      - no file            -> acquire.
      - file w/ DEAD pid   -> steal (stale) + acquire.
      - file w/ LIVE pid   -> if wait: poll until released/stale, else refuse (RuntimeError).
    ``pid_alive`` is injectable so the steal/refuse logic is unit-testable without real procs.
    """

    def __init__(self, path: Path = LOCK_PATH, *, label: str = "",
                 pid_alive: Callable[[int], bool] = pid_is_alive,
                 logfn: Callable[[str], None] = print, sleep: Callable[[float], None] = time.sleep):
        self.path = Path(path)
        self.label = label
        self._pid_alive = pid_alive
        self._logfn = logfn
        self._sleep = sleep
        self._acquired = False

    def _read(self) -> Optional[dict]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "pid": os.getpid(),
            "host": os.environ.get("COMPUTERNAME", ""),
            "start": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "label": self.label,
        }, indent=2), encoding="utf-8")

    def try_acquire(self) -> bool:
        """Attempt a single non-blocking acquire. Steals a stale (dead-pid) lock.
        Returns True on success, False if held by a live foreign pid."""
        info = self._read()
        if info is None:
            self._write()
            self._acquired = True
            return True
        held_pid = int(info.get("pid", -1))
        if held_pid == os.getpid():  # re-entrant for our own pid
            self._acquired = True
            return True
        if not self._pid_alive(held_pid):
            self._logfn(f"[lock] stealing stale lock from dead pid={held_pid} "
                        f"(label={info.get('label')!r})")
            self._write()
            self._acquired = True
            return True
        self._logfn(f"[lock] held by LIVE pid={held_pid} (label={info.get('label')!r}, "
                    f"start={info.get('start')})")
        return False

    def acquire(self, *, wait: bool, poll_s: float = 5.0, timeout_s: Optional[float] = None) -> bool:
        """Acquire the lock. If ``wait`` is False, one attempt then refuse with RuntimeError.
        If ``wait`` is True, poll every ``poll_s`` until acquired or ``timeout_s`` elapses."""
        if self.try_acquire():
            return True
        if not wait:
            raise RuntimeError(
                f"GPU lock held by another live process (see {self.path}). "
                f"Refusing to launch a concurrent GPU job. Use --wait-lock to queue.")
        waited = 0.0
        while True:
            self._sleep(poll_s)
            waited += poll_s
            if self.try_acquire():
                return True
            if timeout_s is not None and waited >= timeout_s:
                raise TimeoutError(f"timed out after {waited:.0f}s waiting for GPU lock {self.path}")
            self._logfn(f"[lock] waiting for GPU lock ... {waited:.0f}s elapsed")

    def release(self) -> None:
        if not self._acquired:
            return
        info = self._read()
        if info and int(info.get("pid", -1)) == os.getpid():
            try:
                self.path.unlink()
            except FileNotFoundError:  # pragma: no cover
                pass
        self._acquired = False

    def __enter__(self) -> "GpuLock":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


# ======================================================================================
# 3. GPU preflight (nvidia-smi)
# ======================================================================================
@dataclass
class GpuState:
    mem_used_mib: int
    mem_total_mib: int
    util_pct: int

    @property
    def mem_used_frac(self) -> float:
        return self.mem_used_mib / self.mem_total_mib if self.mem_total_mib else 0.0


def parse_nvidia_smi(csv_text: str) -> Optional[GpuState]:
    """Parse the first data row of:
        nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
    Header line is skipped; units (' MiB', ' %') are tolerated. Returns None if unparseable.
    Pure -> unit-testable with sample strings (no GPU needed)."""
    for line in csv_text.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("memory.used"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        nums = []
        for p in parts[:3]:
            m = re.search(r"\d+", p)
            if not m:
                break
            nums.append(int(m.group()))
        if len(nums) == 3:
            return GpuState(mem_used_mib=nums[0], mem_total_mib=nums[1], util_pct=nums[2])
    return None


def query_gpu() -> Optional[GpuState]:
    """Run nvidia-smi (read-only) and parse it. Returns None if nvidia-smi is absent."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        r = subprocess.run(
            [exe, "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv"],
            capture_output=True, text=True, timeout=30)
        return parse_nvidia_smi(r.stdout)
    except Exception:  # pragma: no cover
        return None


def gpu_is_busy(state: Optional[GpuState], busy_frac: float = 0.25) -> bool:
    """Heuristic: another process is holding significant VRAM (don't launch into contention).
    None state (no nvidia-smi) is treated as not-busy so CPU-only environments proceed."""
    if state is None:
        return False
    return state.mem_used_frac >= busy_frac


def wait_for_gpu(*, busy_frac: float, poll_s: float, timeout_s: Optional[float],
                 query: Callable[[], Optional[GpuState]] = query_gpu,
                 logfn: Callable[[str], None] = print,
                 sleep: Callable[[float], None] = time.sleep) -> Optional[GpuState]:
    """Block until VRAM usage is below ``busy_frac`` (or timeout). ``query`` injectable for tests."""
    waited = 0.0
    while True:
        state = query()
        if not gpu_is_busy(state, busy_frac):
            return state
        logfn(f"[gpu] busy: {state.mem_used_mib}/{state.mem_total_mib} MiB "
              f"({state.mem_used_frac*100:.0f}%), util={state.util_pct}% -> waiting {poll_s}s")
        if timeout_s is not None and waited >= timeout_s:
            raise TimeoutError(f"GPU still busy after {waited:.0f}s")
        sleep(poll_s)
        waited += poll_s


# ======================================================================================
# Job model + 4/5/6: backoff state machine, resume, manifest
# ======================================================================================
@dataclass
class Job:
    """One declarative training job."""
    label: str
    model: str
    outdir: str                      # relative to nn_results, or absolute
    extra_args: list[str] = field(default_factory=list)
    profile: str = "safe"            # starting batch profile

    def resolved_outdir(self, results_dir: Path) -> Path:
        p = Path(self.outdir)
        return p if p.is_absolute() else results_dir / p


@dataclass
class Attempt:
    profile: str
    accel: str
    exit_code: Optional[int]
    minutes: float
    reason: str


@dataclass
class JobResult:
    label: str
    model: str
    status: str = "pending"          # pending|done|skipped|failed
    attempts: list = field(default_factory=list)
    minutes: float = 0.0
    final_profile: Optional[str] = None
    final_accel: Optional[str] = None


def job_done(outdir: Path, cache_version: str = CACHE_VERSION) -> bool:
    """5. Idempotent resume: a job is 'done' iff summary.json exists w/ the current cache_version."""
    sp = outdir / "summary.json"
    if not sp.exists():
        return False
    try:
        return json.loads(sp.read_text(encoding="utf-8")).get("cache_version") == cache_version
    except (json.JSONDecodeError, OSError):
        return False


@dataclass
class RunOutcome:
    """Result of a single subprocess run (one attempt)."""
    exit_code: int
    reason: str          # "ok" | "cuda_error" | "stall" | "nonzero_exit"
    minutes: float


def run_one_attempt(
    job: Job, outdir: Path, profile: str, accel: str, *,
    python: str = PYTHON, trainer: Path = TRAINER, results_dir: Path = RESULTS_DIR,
    stall_timeout_s: float = 480.0, poll_s: float = 2.0,
    extra_env: Optional[dict] = None,
    runner: Optional[Callable] = None,
    logfn: Callable[[str], None] = print,
) -> RunOutcome:
    """4. Run the trainer ONCE as a subprocess with the given batch profile + accelerator,
    tailing its log. Detects CUDA/OOM errors and stalls (no new output for stall_timeout_s).

    ``runner`` is injectable: it receives (cmd, env, raw_log_path, stall_timeout_s, poll_s,
    logfn) and must return a RunOutcome. Default = _subprocess_runner (real). Tests pass a fake."""
    env = os.environ.copy()
    env["MBARI_ACCEL"] = accel
    for k, v in BATCH_PROFILES[profile].items():
        env[k] = str(v)
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})

    cmd = [
        python,
        str(trainer),
        "--model",
        job.model,
        "--outdir",
        str(outdir),
        *resolve_known_path_args(job.extra_args),
    ]
    raw_log = results_dir / f"{job.label}_raw.log"
    logfn(f"[run] {job.label} model={job.model} profile={profile} accel={accel}")
    logfn(f"[run]   cmd: {' '.join(cmd)}")
    run = runner or _subprocess_runner
    return run(cmd, env, raw_log, stall_timeout_s, poll_s, logfn)


def _subprocess_runner(cmd, env, raw_log: Path, stall_timeout_s: float, poll_s: float,
                       logfn: Callable[[str], None]) -> RunOutcome:
    """Real subprocess runner: launch detached from the interactive terminal, stream output to
    a raw log, watch for stalls (no file growth for stall_timeout_s) and CUDA error signatures."""
    raw_log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tail_buf: list[str] = []
    with open(raw_log, "w", encoding="utf-8", errors="replace") as fh:
        # CREATE_NO_WINDOW keeps it independent of the interactive console on Windows.
        creationflags = 0x08000000 if sys.platform == "win32" else 0
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, env=env, creationflags=creationflags)
        last_output = time.time()
        lines: queue.Queue[Optional[str]] = queue.Queue()

        def read_stdout() -> None:
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    lines.put(line)
            finally:
                lines.put(None)

        reader = threading.Thread(target=read_stdout, name="mbari-train-log-reader", daemon=True)
        reader.start()
        stream_done = False
        try:
            while True:
                try:
                    line = lines.get(timeout=poll_s)
                except queue.Empty:
                    line = None

                if line is None:
                    if not lines.empty():
                        continue
                    if proc.poll() is not None:
                        break
                    if stream_done:
                        time.sleep(poll_s)
                    elif reader.is_alive():
                        pass
                    else:
                        stream_done = True
                    line = ""

                if line:
                    fh.write(line)
                    fh.flush()
                    tail_buf.append(line)
                    if len(tail_buf) > 200:
                        tail_buf.pop(0)
                    last_output = time.time()
                    continue

                if time.time() - last_output > stall_timeout_s:
                    logfn(f"[run] STALL: no output for {stall_timeout_s:.0f}s -> killing")
                    proc.kill()
                    proc.wait(timeout=30)
                    return RunOutcome(exit_code=-9, reason="stall", minutes=(time.time() - t0) / 60)
        finally:
            if proc.poll() is None:
                proc.kill()
            reader.join(timeout=2)
        code = proc.wait()
    minutes = (time.time() - t0) / 60
    tail = "".join(tail_buf)
    if is_capacity_error(tail):
        return RunOutcome(exit_code=code, reason="capacity_error", minutes=minutes)
    if is_cuda_error(tail):
        return RunOutcome(exit_code=code, reason="cuda_error", minutes=minutes)
    if code == 0:
        return RunOutcome(exit_code=0, reason="ok", minutes=minutes)
    return RunOutcome(exit_code=code, reason="nonzero_exit", minutes=minutes)


def run_job_with_backoff(
    job: Job, *, results_dir: Path = RESULTS_DIR, cache_version: str = CACHE_VERSION,
    force: bool = False, max_retries: int = 4, cpu_fallback: bool = True,
    stall_timeout_s: float = 480.0, start_accel: str = "gpu",
    run_attempt: Optional[Callable] = None,
    logfn: Callable[[str], None] = print,
) -> JobResult:
    """4/5. Run one job with the OOM/CUDA/stall auto-backoff state machine:

        start at job.profile (gpu) -> on cuda_error/stall: retry next-smaller profile (gpu)
        -> when no smaller gpu profile remains and cpu_fallback: retry once on CPU (safe profile)
        -> stop on success, or when retries exhausted.

    ``run_attempt`` is injectable (signature like run_one_attempt) so the whole state machine
    is unit-testable with a fake that returns scripted RunOutcomes -- no GPU, no real trainer."""
    outdir = job.resolved_outdir(results_dir)
    result = JobResult(label=job.label, model=job.model)

    if not force and job_done(outdir, cache_version):
        logfn(f">>> SKIP {job.label} (summary.json cache_version={cache_version} up-to-date)")
        result.status = "skipped"
        return result

    outdir.mkdir(parents=True, exist_ok=True)
    attempt_fn = run_attempt or run_one_attempt

    profile = job.profile
    accel = start_accel
    tried_cpu = (start_accel == "cpu")  # already on CPU -> no further CPU fallback move
    for n in range(max_retries):
        logfn(f">>> ATTEMPT {n+1}/{max_retries} {job.label} profile={profile} accel={accel}")
        outcome: RunOutcome = attempt_fn(
            job, outdir, profile, accel,
            results_dir=results_dir, stall_timeout_s=stall_timeout_s, logfn=logfn)
        result.attempts.append(asdict(Attempt(
            profile=profile, accel=accel, exit_code=outcome.exit_code,
            minutes=round(outcome.minutes, 2), reason=outcome.reason)))
        result.minutes += outcome.minutes

        # Success requires both a clean exit AND a valid summary.json (the trainer's done-marker).
        if outcome.reason == "ok" and job_done(outdir, cache_version):
            result.status = "done"
            result.final_profile, result.final_accel = profile, accel
            logfn(f">>> DONE {job.label} ({result.minutes:.1f} min total)")
            return result

        # Decide the backoff move.
        if outcome.reason in ("cuda_error", "stall"):
            nxt = smaller_profile(profile)
            if nxt is not None:
                logfn(f">>> BACKOFF {job.label}: {outcome.reason} -> smaller profile '{nxt}'")
                profile = nxt
                continue
            if cpu_fallback and not tried_cpu and not is_driver_job(job):
                logfn(f">>> FALLBACK {job.label}: {outcome.reason} at smallest GPU profile -> CPU")
                accel, profile, tried_cpu = "cpu", "safe", True
                continue
            if is_driver_job(job) and cpu_fallback and not tried_cpu:
                logfn(f">>> NO_CPU_FALLBACK {job.label}: driver-heavy job could exceed host RAM/pagefile")
            logfn(f">>> GIVEUP {job.label}: {outcome.reason}, no smaller profile / CPU exhausted")
            break
        elif outcome.reason == "capacity_error":
            logfn(f">>> CAPACITY_STOP {job.label}: host RAM/pagefile capacity failure; not retrying")
            break
        else:
            # ok-but-no-summary, or a plain nonzero exit that is NOT a CUDA error: real bug.
            logfn(f">>> FAIL {job.label}: reason={outcome.reason} exit={outcome.exit_code} "
                  f"(not a CUDA/stall error -> not retrying)")
            break

    result.status = "failed"
    result.final_profile, result.final_accel = profile, accel
    return result


# ======================================================================================
# Manifest + status
# ======================================================================================
def write_manifest(results: list[JobResult], path: Path = MANIFEST_PATH,
                   meta: Optional[dict] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cache_version": CACHE_VERSION,
        "meta": meta or {},
        "jobs": [asdict(r) for r in results],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_status(jobs: list[Job], results_dir: Path = RESULTS_DIR,
                 manifest_path: Path = MANIFEST_PATH, logfn: Callable[[str], None] = print) -> None:
    """6. Print a progress table combining on-disk summary.json state with the last manifest."""
    manifest = {}
    if manifest_path.exists():
        try:
            for j in json.loads(manifest_path.read_text(encoding="utf-8")).get("jobs", []):
                manifest[j["label"]] = j
        except (json.JSONDecodeError, OSError):
            pass
    logfn(f"{'LABEL':<22} {'MODEL':<14} {'DISK':<8} {'STATUS':<9} {'PROF':<7} {'ACCEL':<6} {'MIN':>7} ATT")
    logfn("-" * 92)
    for job in jobs:
        outdir = job.resolved_outdir(results_dir)
        disk = "done" if job_done(outdir) else "-"
        m = manifest.get(job.label, {})
        status = m.get("status")
        final_profile = m.get("final_profile")
        final_accel = m.get("final_accel")
        minutes = m.get("minutes", 0)
        attempts = len(m.get("attempts", []))
        if disk == "done" and not status:
            status = "done"
        logfn(f"{job.label:<22} {job.model:<14} {disk:<8} "
              f"{(status or 'pending'):<9} {str(final_profile or '-'):<7} "
              f"{str(final_accel or '-'):<6} {minutes:>7.1f} "
              f"{attempts}")


# ======================================================================================
# Job lists
# ======================================================================================
def default_jobs() -> list[Job]:
    """The Phase-2 job list (mirrors ops/run_nn_phase2.ps1) as declarative Job objects."""
    drv = ["--drivers-parquet", "nn_cache/drivers_hourly.parquet",
           "--drivers-manifest", "nn_cache/drivers_manifest.json"]
    return [
        Job("nhits", "nhits", "nhits"),
        Job("tft", "tft", "tft"),
        Job("nbeatsx", "nbeatsx", "nbeatsx"),
        Job("tsmixerx", "tsmixerx", "tsmixerx"),
        Job("patchtst", "patchtst", "patchtst"),
        Job("itransformer", "itransformer", "itransformer"),
        Job("nhits+drv", "nhits", "nhits_drv", list(drv)),
        Job("tft+drv", "tft", "tft_drv", list(drv)),
        Job("nbeatsx+drv", "nbeatsx", "nbeatsx_drv", list(drv)),
        Job("tsmixerx+drv", "tsmixerx", "tsmixerx_drv", list(drv)),
        Job("nhits+quantile", "nhits", "nhits_q", ["--loss", "quantile"]),
        Job("tsmixerx+drv+q", "tsmixerx", "tsmixerx_drv_q", ["--loss", "quantile", *drv]),
    ]


def load_jobs(jobs_path: Optional[str]) -> list[Job]:
    """Load a declarative jobs JSON (list of {label,model,outdir,extra_args?,profile?})
    or fall back to default_jobs()."""
    if not jobs_path:
        return default_jobs()
    data = json.loads(Path(jobs_path).read_text(encoding="utf-8"))
    items = data["jobs"] if isinstance(data, dict) else data
    return [Job(label=j["label"], model=j["model"], outdir=j["outdir"],
                extra_args=list(j.get("extra_args", [])), profile=j.get("profile", "safe"))
            for j in items]


# ======================================================================================
# Orchestration entry
# ======================================================================================
def run_sweep(jobs: list[Job], args, logfn: Callable[[str], None] = print) -> list[JobResult]:
    """Top-level GPU sweep: power guard + wake-lock + GPU mutex around a sequential, single-tenant
    run of every job with per-job backoff. Writes the manifest after each job (crash-safe)."""
    results: list[JobResult] = []
    check_power_state(enforce=args.enforce_power, logfn=logfn)

    # CPU-only runs don't need the wake-lock for GPU integrity, but keeping the system awake is
    # still desirable for long runs; honour the explicit --no-wake-lock flag.
    with WakeLock(logfn=logfn, enabled=not args.no_wake_lock):
        with GpuLock(label="mbari_train sweep", logfn=logfn) as lock:
            lock.acquire(wait=args.wait_lock, poll_s=args.lock_poll_s,
                         timeout_s=args.lock_timeout_s)
            logfn(f"[lock] acquired GPU mutex (pid={os.getpid()})")

            for job in jobs:
                guard_reason = driver_guard_reason(job)
                job_to_run = job
                if guard_reason and not args.allow_unsafe_driver_models:
                    job_to_run = with_driver_guard_shape(job, args.driver_guard_tail_weeks)
                    logfn(
                        f">>> GUARD_BOUNDED {job.label}: {guard_reason} "
                        f"tail_weeks={args.driver_guard_tail_weeks:g}"
                    )

                # Preflight: wait while another (foreign) process holds significant VRAM.
                # Skipped entirely for CPU-only runs (they never contend for VRAM).
                if args.accel != "cpu" and not args.no_gpu_preflight:
                    try:
                        wait_for_gpu(busy_frac=args.gpu_busy_frac, poll_s=args.gpu_poll_s,
                                     timeout_s=args.gpu_wait_timeout_s, logfn=logfn)
                    except TimeoutError as e:
                        logfn(f"[gpu] preflight timeout for {job.label}: {e} -- skipping job")
                        r = JobResult(label=job_to_run.label, model=job_to_run.model, status="failed")
                        r.attempts.append({"profile": job_to_run.profile, "accel": "gpu",
                                           "exit_code": None, "minutes": 0.0,
                                           "reason": "gpu_preflight_timeout"})
                        results.append(r)
                        write_manifest(results, meta=_meta(args))
                        continue

                # If user forced a starting profile, override the job's.
                if args.profile:
                    job_to_run = Job(job_to_run.label, job_to_run.model, job_to_run.outdir, job_to_run.extra_args, args.profile)

                r = run_job_with_backoff(
                    job_to_run, results_dir=RESULTS_DIR, force=args.force,
                    max_retries=args.max_retries, cpu_fallback=not args.no_cpu_fallback,
                    stall_timeout_s=args.stall_timeout_s, start_accel=args.accel, logfn=logfn)
                results.append(r)
                write_manifest(results, meta=_meta(args))
    return results


def _meta(args) -> dict:
    return {"profile_override": args.profile, "force": args.force,
            "max_retries": args.max_retries, "stall_timeout_s": args.stall_timeout_s,
            "allow_unsafe_driver_models": args.allow_unsafe_driver_models,
            "driver_guard_tail_weeks": args.driver_guard_tail_weeks}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Robust orchestrator for MBARI NeuralForecast training (single source of truth).")
    ap.add_argument("--jobs", default=None, help="path to a declarative jobs JSON; default = Phase-2 list")
    ap.add_argument("--status", action="store_true", help="print the progress table and exit")
    ap.add_argument("--force", action="store_true", help="re-run jobs even if summary.json is up-to-date")
    ap.add_argument("--profile", choices=list(BATCH_PROFILES), default=None,
                    help="override the starting batch profile for all jobs (default: per-job, usually 'safe')")
    ap.add_argument("--enforce-power", action="store_true",
                    help="set AC sleep/hibernate idle timeouts to 0 (Never) if not already")
    ap.add_argument("--accel", choices=["gpu", "cpu"], default="gpu",
                    help="starting accelerator for every job (default gpu; 'cpu' = GPU-free run, "
                         "skips the VRAM preflight and wake-lock)")
    ap.add_argument("--no-wake-lock", action="store_true", help="do not hold the OS wake-lock")
    ap.add_argument("--no-cpu-fallback", action="store_true", help="do not fall back to CPU after GPU backoff")
    ap.add_argument("--allow-unsafe-driver-models", action="store_true",
                    help="allow full-scale driver jobs for guarded models; use only with reduced job shape")
    ap.add_argument("--driver-guard-tail-weeks", type=float, default=DEFAULT_DRIVER_GUARD_TAIL_WEEKS,
                    help="recent-history weeks used for guarded driver models (default 104; <=0 disables bounding)")
    ap.add_argument("--no-gpu-preflight", action="store_true", help="skip the nvidia-smi VRAM contention check")
    ap.add_argument("--max-retries", type=int, default=4, help="max attempts per job across backoff (default 4)")
    ap.add_argument("--stall-timeout-s", type=float, default=480.0,
                    help="kill+retry a job after this many seconds with no new log output (default 480 = 8 min)")
    # lock / gpu wait tuning
    ap.add_argument("--wait-lock", action="store_true", help="wait for the GPU lock instead of refusing")
    ap.add_argument("--lock-poll-s", type=float, default=5.0)
    ap.add_argument("--lock-timeout-s", type=float, default=None)
    ap.add_argument("--gpu-busy-frac", type=float, default=0.25,
                    help="treat the GPU as busy at/above this VRAM fraction (default 0.25)")
    ap.add_argument("--gpu-poll-s", type=float, default=15.0)
    ap.add_argument("--gpu-wait-timeout-s", type=float, default=None)
    ap.add_argument("--mem-guard-gb", type=float, default=40.0,
                    help="refuse to start if an Ollama model >= this many GB is resident "
                         "(default 40; blocks the 51GB-class local models that OOM the box)")
    ap.add_argument("--skip-mem-guard", action="store_true",
                    help="bypass the resident-Ollama-model memory guard")
    return ap


OLLAMA_PS_URL = "http://127.0.0.1:11434/api/ps"


def resident_big_models(threshold_gb: float) -> list[tuple[str, float]]:
    """[(name, GB)] for Ollama models resident at/above threshold_gb.

    Returns [] if Ollama is unreachable (guard fails open — never blocks a run just
    because the model server is down).
    """
    try:
        with urllib.request.urlopen(OLLAMA_PS_URL, timeout=3) as resp:
            data = json.load(resp)
    except Exception:
        return []
    out: list[tuple[str, float]] = []
    for m in data.get("models", []):
        gb = (m.get("size_vram") or m.get("size") or 0) / (1024 ** 3)
        if gb >= threshold_gb:
            out.append((m.get("name", "?"), gb))
    return out


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    jobs = load_jobs(args.jobs)

    if args.status:
        print_status(jobs)
        return 0

    # Memory guard: a GPU sweep on top of a resident 51GB-class local model overruns the
    # system commit limit and OOM-crashes the terminals. Covers ALL orchestrator paths
    # (ops/train.ps1, direct `python mbari_train.py`, the detached launcher).
    if args.accel != "cpu" and not args.skip_mem_guard:
        big = resident_big_models(args.mem_guard_gb)
        if big:
            print("REFUSING to start GPU sweep: large Ollama model(s) resident — training "
                  "alongside these risks a commit-limit OOM that crashes the terminals:",
                  file=sys.stderr)
            for name, gb in big:
                print(f"  - {name}: {gb:.1f} GB resident", file=sys.stderr)
            print("\nFree it first:  ollama stop <name>  (or wait for OLLAMA_KEEP_ALIVE),\n"
                  "then re-run. Override deliberately with --skip-mem-guard.", file=sys.stderr)
            return 2

    print(f"=== mbari_train sweep start {time.strftime('%Y-%m-%dT%H:%M:%S')} === ({len(jobs)} jobs)")
    results = run_sweep(jobs, args)
    write_manifest(results, meta=_meta(args))
    done = sum(1 for r in results if r.status == "done")
    skipped = sum(1 for r in results if r.status == "skipped")
    failed = sum(1 for r in results if r.status == "failed")
    print(f"=== mbari_train sweep end === done={done} skipped={skipped} failed={failed}")
    print(f"manifest: {MANIFEST_PATH}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
