#!/usr/bin/env python3
"""Launch-time GPU admission guard for the Monterey Bay AI Lab workstation.

The goal is to reject the newest GPU job before it allocates VRAM when the RTX
5090 is already too full. Runtime safety monitors catch OOM-ish conditions after
launch; this guard is the earlier gate.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable


DEFAULT_MAX_USED_PCT = 0.85
DEFAULT_RESERVE_MIB = 4096


@dataclass(frozen=True)
class GpuState:
    used_mib: int
    total_mib: int
    free_mib: int
    util_pct: int | None = None
    name: str = "GPU"

    @property
    def used_pct(self) -> float:
        return self.used_mib / self.total_mib if self.total_mib else 0.0


@dataclass(frozen=True)
class VramProcess:
    pid: int
    name: str
    dedicated_mib: float
    command_line: str = ""


class GpuAdmissionError(RuntimeError):
    """Raised when a launch would overcommit GPU memory."""


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default


def _nvidia_smi() -> str | None:
    exe = shutil.which("nvidia-smi")
    if exe:
        return exe
    win_smi = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "System32", "nvidia-smi.exe")
    return win_smi if os.path.exists(win_smi) else None


def query_gpu_state() -> GpuState | None:
    """Return current VRAM state, or None when nvidia-smi is unavailable."""
    exe = _nvidia_smi()
    if not exe:
        return None
    cmd = [
        exe,
        "--query-gpu=name,memory.used,memory.total,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=8, check=False)
    except Exception:
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    line = proc.stdout.strip().splitlines()[0]
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 5:
        return None
    try:
        return GpuState(
            name=parts[0],
            used_mib=int(float(parts[1])),
            total_mib=int(float(parts[2])),
            free_mib=int(float(parts[3])),
            util_pct=int(float(parts[4])),
        )
    except ValueError:
        return None


def _powershell_json(script: str) -> object | None:
    if sys.platform != "win32":
        return None
    exe = shutil.which("powershell.exe") or shutil.which("pwsh.exe")
    if not exe:
        return None
    try:
        proc = subprocess.run(
            [exe, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
            capture_output=True,
            text=True,
            timeout=12,
            check=False,
        )
    except Exception:
        return None
    raw = proc.stdout.strip()
    if proc.returncode != 0 or not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def query_vram_processes(limit: int = 8) -> list[VramProcess]:
    """Return top Windows dedicated-VRAM holders from performance counters."""
    script = rf"""
$samples = Get-Counter '\GPU Process Memory(*)\Dedicated Usage' -ErrorAction SilentlyContinue |
  Select-Object -ExpandProperty CounterSamples
$mem = @{{}}
foreach ($s in $samples) {{
  if ($s.CookedValue -le 0) {{ continue }}
  if ($s.InstanceName -match 'pid_(\d+)_') {{
    $pidText = $matches[1]
    if (-not $mem.ContainsKey($pidText)) {{ $mem[$pidText] = 0.0 }}
    $mem[$pidText] += [double]$s.CookedValue
  }}
}}
$rows = @()
foreach ($pidText in $mem.Keys) {{
  $procId = [int]$pidText
  $p = Get-Process -Id $procId -ErrorAction SilentlyContinue
  $w = Get-CimInstance Win32_Process -Filter "ProcessId = $procId" -ErrorAction SilentlyContinue
  $rows += [pscustomobject]@{{
    pid = $procId
    name = if ($p) {{ $p.ProcessName }} else {{ "unknown" }}
    dedicated_mib = [math]::Round($mem[$pidText] / 1MB, 1)
    command_line = if ($w) {{ $w.CommandLine }} else {{ "" }}
  }}
}}
$rows | Sort-Object dedicated_mib -Descending | Select-Object -First {int(limit)} |
  ConvertTo-Json -Compress
"""
    data = _powershell_json(script)
    if data is None:
        return []
    if isinstance(data, dict):
        data = [data]
    out: list[VramProcess] = []
    for row in data if isinstance(data, list) else []:
        try:
            out.append(
                VramProcess(
                    pid=int(row.get("pid", 0)),
                    name=str(row.get("name") or "unknown"),
                    dedicated_mib=float(row.get("dedicated_mib") or 0.0),
                    command_line=str(row.get("command_line") or ""),
                )
            )
        except (TypeError, ValueError):
            continue
    return out


def _short_cmd(command_line: str, max_len: int = 150) -> str:
    text = re.sub(r"\s+", " ", command_line or "").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def format_denial(
    *,
    label: str,
    state: GpuState,
    request_mib: int,
    reserve_mib: int,
    max_used_pct: float,
    reasons: Iterable[str],
    processes: Iterable[VramProcess],
) -> str:
    lines = [
        f"GPU_ADMISSION_DENIED: refusing to start {label}",
        "",
        (
            f"Current {state.name}: {state.used_mib}/{state.total_mib} MiB used "
            f"({state.used_pct * 100:.1f}%), {state.free_mib} MiB free, "
            f"utilization {state.util_pct if state.util_pct is not None else '?'}%."
        ),
        (
            f"Requested footprint: ~{request_mib} MiB plus {reserve_mib} MiB reserve "
            f"(needs {request_mib + reserve_mib} MiB free before launch)."
        ),
        "",
        "Reason:",
    ]
    lines.extend(f"  - {reason}" for reason in reasons)
    procs = list(processes)
    if procs:
        lines += ["", "Top VRAM holders:"]
        for proc in procs:
            cmd = _short_cmd(proc.command_line)
            suffix = f" :: {cmd}" if cmd else ""
            lines.append(f"  - pid {proc.pid} {proc.name}: {proc.dedicated_mib:.1f} MiB{suffix}")
    lines += [
        "",
        "Free VRAM first by stopping/finishing one of those jobs, or deliberately override with:",
        "  set MBARI_GPU_GUARD_DISABLE=1",
    ]
    return "\n".join(lines)


def check_gpu_admission(
    *,
    label: str,
    request_mib: int,
    reserve_mib: int | None = None,
    max_used_pct: float | None = None,
) -> str | None:
    """Return a denial message, or None if the launch is admitted."""
    if os.environ.get("MBARI_GPU_GUARD_DISABLE", "").lower() in {"1", "true", "yes"}:
        return None
    reserve_mib = _env_int("MBARI_GPU_GUARD_RESERVE_MIB", DEFAULT_RESERVE_MIB) if reserve_mib is None else reserve_mib
    max_used_pct = _env_float("MBARI_GPU_GUARD_MAX_USED_PCT", DEFAULT_MAX_USED_PCT) if max_used_pct is None else max_used_pct
    if max_used_pct > 1.0:
        max_used_pct = max_used_pct / 100.0

    state = query_gpu_state()
    if state is None:
        # Fail open for CPU/test environments or broken nvidia-smi; the runtime will still
        # fail later if CUDA is genuinely unavailable.
        return None

    reasons: list[str] = []
    need_free = max(0, request_mib) + max(0, reserve_mib)
    if state.used_pct >= max_used_pct:
        reasons.append(
            f"current VRAM use is {state.used_pct * 100:.1f}%, above the configured "
            f"{max_used_pct * 100:.1f}% launch ceiling"
        )
    if state.free_mib < need_free:
        reasons.append(
            f"only {state.free_mib} MiB is free, but this launch needs {need_free} MiB "
            "free for its estimated footprint plus reserve"
        )

    if not reasons:
        return None
    return format_denial(
        label=label,
        state=state,
        request_mib=request_mib,
        reserve_mib=reserve_mib,
        max_used_pct=max_used_pct,
        reasons=reasons,
        processes=query_vram_processes(),
    )


def enforce_gpu_admission(
    *,
    label: str,
    request_mib: int,
    reserve_mib: int | None = None,
    max_used_pct: float | None = None,
    enabled: bool = True,
) -> None:
    if not enabled:
        return
    denial = check_gpu_admission(
        label=label,
        request_mib=request_mib,
        reserve_mib=reserve_mib,
        max_used_pct=max_used_pct,
    )
    if denial:
        print(denial, file=sys.stderr)
        raise SystemExit(2)


def estimate_run_production_mib(batch_size: int, context_window: int) -> int:
    """Conservative estimate calibrated from observed RTX 5090 runs."""
    batch_size = max(1, int(batch_size))
    context_window = max(1, int(context_window))
    base = 24_576  # observed production runner footprint at batch=2/context=168.
    extra_batch = max(0, batch_size - 2) * 1536
    extra_context = max(0, context_window - 168) * batch_size * 4
    return min(26_624, base + extra_batch + extra_context)


def estimate_smoke_test_mib(batch_size: int, context_window: int) -> int:
    batch_size = max(1, int(batch_size))
    context_window = max(1, int(context_window))
    base = 18_432
    extra_batch = max(0, batch_size - 2) * 1024
    extra_context = max(0, context_window - 720) * batch_size * 2
    return min(24_576, base + extra_batch + extra_context)


def estimate_neural_forecast_mib(
    *,
    model: str,
    smoke: bool = False,
    loss: str = "mae",
    drivers: bool = False,
) -> int:
    if smoke:
        return 4096
    model_l = model.lower()
    heavy = {"mbari_moe", "tft", "itransformer", "patchtst"}
    medium = {"nhits", "nbeatsx", "tsmixerx"}
    request = 8192 if model_l in heavy else 6144 if model_l in medium else 4096
    if loss == "quantile":
        request += 1024
    if drivers:
        request += 2048
    return min(request, 12_288)


def estimate_mbari_train_mib(profile: str | None = None) -> int:
    profile = (profile or "safe").lower()
    return {"safe": 8192, "medium": 10_240, "large": 12_288}.get(profile, 8192)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Refuse GPU launches that would overcommit VRAM.")
    parser.add_argument("--label", required=True)
    parser.add_argument("--request-mib", type=int, required=True)
    parser.add_argument("--reserve-mib", type=int, default=None)
    parser.add_argument("--max-used-pct", type=float, default=None)
    args = parser.parse_args(argv)

    denial = check_gpu_admission(
        label=args.label,
        request_mib=args.request_mib,
        reserve_mib=args.reserve_mib,
        max_used_pct=args.max_used_pct,
    )
    if denial:
        print(denial, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
