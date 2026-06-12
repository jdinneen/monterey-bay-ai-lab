#!/usr/bin/env python3
"""
Unit tests for the Monterey Bay AI Lab training orchestrator (mbal_train.py).

ALL tests are CPU/mock only -- they NEVER launch a GPU job, never run the real trainer (except
the optional opt-in CPU smoke), and never touch power settings. Every timeout / sleep / pid /
nvidia-smi / subprocess dependency is injected so the suite runs in well under a second.

Run with pytest if available, else as a plain script (asserts):
    python -m pytest tests/test_mbal_train.py -q
    python tests/test_mbal_train.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mbal_train as mt  # noqa: E402


# ----------------------------------------------------------------------------------------
# 3. nvidia-smi CSV parsing (pure, sample strings)
# ----------------------------------------------------------------------------------------
def test_parse_nvidia_smi_normal():
    csv = ("memory.used [MiB], memory.total [MiB], utilization.gpu [%]\n"
           "20195 MiB, 32607 MiB, 63 %\n")
    st = mt.parse_nvidia_smi(csv)
    assert st is not None
    assert st.mem_used_mib == 20195 and st.mem_total_mib == 32607 and st.util_pct == 63
    assert abs(st.mem_used_frac - 20195 / 32607) < 1e-9


def test_parse_nvidia_smi_no_units_and_idle():
    st = mt.parse_nvidia_smi("memory.used, memory.total, utilization.gpu\n512, 32607, 0\n")
    assert st.mem_used_mib == 512 and st.util_pct == 0


def test_parse_nvidia_smi_garbage():
    assert mt.parse_nvidia_smi("") is None
    assert mt.parse_nvidia_smi("nonsense\n") is None


def test_gpu_busy_threshold():
    busy = mt.GpuState(mem_used_mib=20000, mem_total_mib=32607, util_pct=63)
    idle = mt.GpuState(mem_used_mib=300, mem_total_mib=32607, util_pct=0)
    assert mt.gpu_is_busy(busy) is True
    assert mt.gpu_is_busy(idle) is False
    assert mt.gpu_is_busy(None) is False  # no nvidia-smi -> proceed (CPU env)


def test_wait_for_gpu_polls_until_free():
    # First query busy, second query free -> should return after one sleep.
    states = [mt.GpuState(30000, 32607, 90), mt.GpuState(200, 32607, 0)]
    calls = {"q": 0, "sleep": 0}

    def fake_query():
        s = states[min(calls["q"], len(states) - 1)]
        calls["q"] += 1
        return s

    def fake_sleep(_):
        calls["sleep"] += 1

    out = mt.wait_for_gpu(busy_frac=0.25, poll_s=1, timeout_s=100,
                          query=fake_query, logfn=lambda *_: None, sleep=fake_sleep)
    assert out.mem_used_mib == 200
    assert calls["sleep"] == 1


def test_wait_for_gpu_timeout():
    def fake_query():
        return mt.GpuState(30000, 32607, 90)
    try:
        mt.wait_for_gpu(busy_frac=0.25, poll_s=1, timeout_s=1,
                        query=fake_query, logfn=lambda *_: None, sleep=lambda _: None)
        assert False, "expected TimeoutError"
    except TimeoutError:
        pass


# ----------------------------------------------------------------------------------------
# 2. powercfg AC index parsing
# ----------------------------------------------------------------------------------------
def test_parse_ac_index_never():
    sample = ("    GUID Alias: STANDBYIDLE\n"
              "    Current AC Power Setting Index: 0x00000000\n"
              "    Current DC Power Setting Index: 0x00000258\n")
    assert mt.parse_ac_index(sample) == 0


def test_parse_ac_index_thirty_min():
    sample = "Current AC Power Setting Index: 0x00000708\n"  # 1800 s
    assert mt.parse_ac_index(sample) == 1800


def test_parse_ac_index_decimal_and_missing():
    assert mt.parse_ac_index("Current AC Power Setting Index: 1800\n") == 1800
    assert mt.parse_ac_index("no setting here") is None


# ----------------------------------------------------------------------------------------
# 1. GPU lock: acquire / steal-stale / refuse-live / wait
# ----------------------------------------------------------------------------------------
def test_lock_acquire_and_release(tmp_path):
    p = tmp_path / ".gpu.lock"
    lock = mt.GpuLock(p, label="t", logfn=lambda *_: None)
    assert lock.acquire(wait=False) is True
    assert p.exists()
    info = json.loads(p.read_text())
    assert info["pid"] == __import__("os").getpid()
    lock.release()
    assert not p.exists()


def test_lock_steals_stale_dead_pid(tmp_path):
    p = tmp_path / ".gpu.lock"
    p.write_text(json.dumps({"pid": 999999, "label": "ghost", "start": "x"}))
    lock = mt.GpuLock(p, pid_alive=lambda pid: False, logfn=lambda *_: None)
    assert lock.acquire(wait=False) is True  # stole the stale lock
    assert json.loads(p.read_text())["pid"] == __import__("os").getpid()


def test_lock_refuses_live_foreign_pid(tmp_path):
    p = tmp_path / ".gpu.lock"
    p.write_text(json.dumps({"pid": 4242, "label": "live", "start": "x"}))
    lock = mt.GpuLock(p, pid_alive=lambda pid: True, logfn=lambda *_: None)
    try:
        lock.acquire(wait=False)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_lock_wait_then_acquire(tmp_path):
    p = tmp_path / ".gpu.lock"
    p.write_text(json.dumps({"pid": 4242, "label": "live", "start": "x"}))
    state = {"alive": True, "sleeps": 0}

    def pid_alive(_):
        return state["alive"]

    def sleeper(_):
        state["sleeps"] += 1
        if state["sleeps"] >= 2:
            state["alive"] = False  # the holder dies -> lock becomes stale

    lock = mt.GpuLock(p, pid_alive=pid_alive, logfn=lambda *_: None, sleep=sleeper)
    assert lock.acquire(wait=True, poll_s=1) is True
    assert state["sleeps"] >= 2


# ----------------------------------------------------------------------------------------
# Batch profile selection / backoff ordering
# ----------------------------------------------------------------------------------------
def test_smaller_profile_order():
    assert mt.smaller_profile("large") == "medium"
    assert mt.smaller_profile("medium") == "safe"
    assert mt.smaller_profile("safe") is None
    assert mt.smaller_profile("unknown") is None


def test_safe_profile_values():
    # The proven-stable profile from the runbook: 12/64/128/64.
    assert mt.BATCH_PROFILES["safe"] == {
        "MBAL_BATCH_SIZE": 12, "MBAL_WINDOWS_BATCH": 64,
        "MBAL_INFER_WINDOWS_BATCH": 128, "MBAL_VALID_BATCH": 64}


def test_is_cuda_error():
    assert mt.is_cuda_error("torch.AcceleratorError: CUDA error: unknown error")
    assert mt.is_cuda_error("CUDA out of memory. Tried to allocate 81 GiB")
    assert not mt.is_cuda_error("DONE nhits in 12.3 min")


# ----------------------------------------------------------------------------------------
# 4/5. Backoff state machine via a fake run_attempt
# ----------------------------------------------------------------------------------------
def _job(tmp_path, label="j", model="nhits", profile="large"):
    return mt.Job(label=label, model=model, outdir=str(tmp_path / label), profile=profile)


def _writer_summary(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "summary.json").write_text(json.dumps({"cache_version": mt.CACHE_VERSION}))


def make_fake_run(script):
    """script: list of reasons to return in order. On 'ok', also writes summary.json."""
    calls = []

    def fake_run(job, outdir, profile, accel, **kw):
        reason = script[len(calls)]
        calls.append({"profile": profile, "accel": accel, "reason": reason})
        if reason == "ok":
            _writer_summary(Path(outdir))
            return mt.RunOutcome(exit_code=0, reason="ok", minutes=0.1)
        code = -9 if reason == "stall" else 1
        return mt.RunOutcome(exit_code=code, reason=reason, minutes=0.1)

    fake_run.calls = calls
    return fake_run


def test_backoff_cuda_then_success(tmp_path):
    # large blows up (cuda) -> medium succeeds.
    job = _job(tmp_path, profile="large")
    fake = make_fake_run(["cuda_error", "ok"])
    r = mt.run_job_with_backoff(job, results_dir=tmp_path, run_attempt=fake,
                                logfn=lambda *_: None)
    assert r.status == "done"
    assert [c["profile"] for c in fake.calls] == ["large", "medium"]
    assert r.final_accel == "gpu" and r.final_profile == "medium"


def test_backoff_walks_to_cpu(tmp_path):
    # cuda at large, medium, safe -> then CPU fallback succeeds.
    job = _job(tmp_path, profile="large")
    fake = make_fake_run(["cuda_error", "cuda_error", "cuda_error", "ok"])
    r = mt.run_job_with_backoff(job, results_dir=tmp_path, run_attempt=fake, max_retries=6,
                                logfn=lambda *_: None)
    assert r.status == "done"
    accels = [c["accel"] for c in fake.calls]
    profs = [c["profile"] for c in fake.calls]
    assert profs == ["large", "medium", "safe", "safe"]
    assert accels == ["gpu", "gpu", "gpu", "cpu"]


def test_stall_triggers_backoff(tmp_path):
    job = _job(tmp_path, profile="medium")
    fake = make_fake_run(["stall", "ok"])
    r = mt.run_job_with_backoff(job, results_dir=tmp_path, run_attempt=fake,
                                logfn=lambda *_: None)
    assert r.status == "done"
    assert [c["profile"] for c in fake.calls] == ["medium", "safe"]


def test_nonzero_non_cuda_error_does_not_retry(tmp_path):
    job = _job(tmp_path, profile="safe")
    fake = make_fake_run(["nonzero_exit"])  # a real bug, not CUDA -> stop immediately
    r = mt.run_job_with_backoff(job, results_dir=tmp_path, run_attempt=fake,
                                logfn=lambda *_: None)
    assert r.status == "failed"
    assert len(fake.calls) == 1


def test_capacity_error_does_not_retry(tmp_path):
    job = _job(tmp_path, profile="safe")
    fake = make_fake_run(["capacity_error", "ok"])
    r = mt.run_job_with_backoff(job, results_dir=tmp_path, run_attempt=fake,
                                max_retries=5, logfn=lambda *_: None)
    assert r.status == "failed"
    assert len(fake.calls) == 1


def test_driver_job_does_not_fall_back_to_cpu(tmp_path):
    job = mt.Job(
        label="nhits+drv",
        model="nhits",
        outdir=str(tmp_path / "nhits_drv"),
        extra_args=["--drivers-parquet", "drivers.parquet", "--drivers-manifest", "drivers.json"],
        profile="safe",
    )
    fake = make_fake_run(["cuda_error", "ok"])
    r = mt.run_job_with_backoff(job, results_dir=tmp_path, run_attempt=fake,
                                max_retries=5, cpu_fallback=True, logfn=lambda *_: None)
    assert r.status == "failed"
    assert len(fake.calls) == 1
    assert fake.calls[0]["accel"] == "gpu"


def test_retries_capped(tmp_path):
    job = _job(tmp_path, profile="safe")  # smallest; no smaller profile
    fake = make_fake_run(["cuda_error", "cuda_error"])  # safe -> cpu -> giveup
    r = mt.run_job_with_backoff(job, results_dir=tmp_path, run_attempt=fake, max_retries=5,
                                cpu_fallback=True, logfn=lambda *_: None)
    assert r.status == "failed"
    # safe(gpu) -> cpu(safe) -> no more moves
    assert len(fake.calls) == 2
    assert fake.calls[1]["accel"] == "cpu"


def test_subprocess_runner_kills_silent_stall(tmp_path):
    raw_log = tmp_path / "silent.log"
    cmd = [sys.executable, "-c", "import time; time.sleep(5)"]
    r = mt._subprocess_runner(
        cmd, __import__("os").environ.copy(), raw_log,
        stall_timeout_s=0.2, poll_s=0.05, logfn=lambda *_: None)
    assert r.reason == "stall"
    assert r.exit_code == -9


# ----------------------------------------------------------------------------------------
# 5. Idempotent resume
# ----------------------------------------------------------------------------------------
def test_job_done_detects_current_cache_version(tmp_path):
    out = tmp_path / "j"
    out.mkdir()
    (out / "summary.json").write_text(json.dumps({"cache_version": mt.CACHE_VERSION}))
    assert mt.job_done(out) is True


def test_job_done_rejects_stale_or_missing(tmp_path):
    out = tmp_path / "j"
    out.mkdir()
    assert mt.job_done(out) is False  # missing
    (out / "summary.json").write_text(json.dumps({"cache_version": "old_version"}))
    assert mt.job_done(out) is False  # stale


def test_skip_when_done_and_force_overrides(tmp_path):
    job = _job(tmp_path, profile="safe")
    _writer_summary(Path(job.resolved_outdir(tmp_path)))
    # Not forced -> skipped, run_attempt never called.
    fake = make_fake_run(["ok"])
    r = mt.run_job_with_backoff(job, results_dir=tmp_path, run_attempt=fake, force=False,
                                logfn=lambda *_: None)
    assert r.status == "skipped" and len(fake.calls) == 0
    # Forced -> runs anyway.
    fake2 = make_fake_run(["ok"])
    r2 = mt.run_job_with_backoff(job, results_dir=tmp_path, run_attempt=fake2, force=True,
                                 logfn=lambda *_: None)
    assert r2.status == "done" and len(fake2.calls) == 1


# ----------------------------------------------------------------------------------------
# 6. Manifest + job loading
# ----------------------------------------------------------------------------------------
def test_manifest_roundtrip(tmp_path):
    r = mt.JobResult(label="x", model="nhits", status="done", minutes=3.2,
                     final_profile="safe", final_accel="gpu")
    p = tmp_path / "run_manifest.json"
    mt.write_manifest([r], path=p, meta={"force": False})
    data = json.loads(p.read_text())
    assert data["jobs"][0]["label"] == "x" and data["jobs"][0]["status"] == "done"
    assert data["cache_version"] == mt.CACHE_VERSION


def test_load_jobs_default():
    jobs = mt.load_jobs(None)
    labels = [j.label for j in jobs]
    assert "nhits" in labels and "tsmixerx+drv+q" in labels
    assert len(jobs) == 12


def test_driver_guard_marks_known_bad_models():
    jobs = {j.label: j for j in mt.load_jobs(None)}
    assert mt.driver_guard_reason(jobs["nhits+drv"])
    assert mt.driver_guard_reason(jobs["tft+drv"])
    assert mt.driver_guard_reason(jobs["nbeatsx+drv"])
    assert mt.driver_guard_reason(jobs["tsmixerx+drv"])
    assert mt.driver_guard_reason(jobs["tsmixerx+drv+q"])


def test_driver_guard_shape_adds_tail_weeks():
    jobs = {j.label: j for j in mt.load_jobs(None)}
    bounded = mt.with_driver_guard_shape(jobs["nhits+drv"], 104)
    assert "--tail-weeks" in bounded.extra_args
    i = bounded.extra_args.index("--tail-weeks")
    assert bounded.extra_args[i + 1] == "104"
    assert "--drivers-parquet" in bounded.extra_args
    tsmix_bounded = mt.with_driver_guard_shape(jobs["tsmixerx+drv"], 104)
    assert "--tail-weeks" in tsmix_bounded.extra_args


def test_resolve_known_path_args_relative_to_root(tmp_path):
    args = ["--drivers-parquet", "nn_cache/drivers.parquet", "--loss", "mae"]
    out = mt.resolve_known_path_args(args, tmp_path)
    assert out[1] == str(tmp_path / "nn_cache" / "drivers.parquet")
    assert out[2:] == ["--loss", "mae"]


def test_status_uses_disk_done_when_manifest_missing(tmp_path):
    job = mt.Job("x", "nhits", "x")
    _writer_summary(tmp_path / "x")
    lines = []
    mt.print_status([job], results_dir=tmp_path, manifest_path=tmp_path / "missing.json", logfn=lines.append)
    assert any("x" in line and "done" in line for line in lines)


def test_load_jobs_from_file(tmp_path):
    f = tmp_path / "jobs.json"
    f.write_text(json.dumps({"jobs": [
        {"label": "a", "model": "dlinear", "outdir": "a", "extra_args": ["--smoke"], "profile": "medium"}]}))
    jobs = mt.load_jobs(str(f))
    assert len(jobs) == 1 and jobs[0].profile == "medium" and jobs[0].extra_args == ["--smoke"]


# ----------------------------------------------------------------------------------------
# WakeLock no-op safety (must never raise on any platform)
# ----------------------------------------------------------------------------------------
def test_wakelock_disabled_is_noop():
    with mt.WakeLock(logfn=lambda *_: None, enabled=False):
        pass  # must not raise


# ----------------------------------------------------------------------------------------
# Plain-script runner (no pytest required)
# ----------------------------------------------------------------------------------------
def _run_all():
    import inspect
    import tempfile
    fns = [(n, f) for n, f in globals().items()
           if n.startswith("test_") and callable(f)]
    passed = failed = 0
    for name, fn in fns:
        params = inspect.signature(fn).parameters
        try:
            if "tmp_path" in params:
                with tempfile.TemporaryDirectory() as d:
                    fn(Path(d))
            else:
                fn()
            passed += 1
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{passed} passed, {failed} failed")
    return failed


if __name__ == "__main__":
    raise SystemExit(1 if _run_all() else 0)

