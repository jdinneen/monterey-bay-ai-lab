from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "ops" / "mbal_dashboard.py"


def _source() -> str:
    return DASHBOARD.read_text(encoding="utf-8")


def test_dashboard_is_decision_first_and_project_named():
    source = _source()
    compile(source, str(DASHBOARD), "exec")
    assert 'page_title="Monterey Bay AI Lab Command Center"' in source
    assert "Monterey Bay AI Lab Command Center" in source
    assert "Live AI lab command center" in source
    assert "What matters right now" in source
    assert "Running discovery models" in source
    assert "MBAI" not in source
    assert '"Decision"' in source
    assert '"Visibility"' in source


def test_dashboard_uses_artifact_backed_decision_helpers():
    source = _source()
    assert "def headline_decision(" in source
    assert "def decision_items(" in source
    assert "def blocker_frame(" in source
    assert "def active_work_frame(" in source
    assert "def artifact_freshness_frame(" in source
    assert "def recent_artifact_frame(" in source
    assert "EXCLUDE_SAN_DIEGO" in source
    assert "calibrated_deploy_ready" in source
    assert "calibrated_lift" in source
    assert "raw_ap_lift" in source
    assert 'd3.metric("Calibrated AP"' in source
    assert 'd3.metric("Model AP"' not in source


def test_dashboard_surfaces_stale_data_and_current_activity():
    source = _source()
    assert "WATCHED_ARTIFACTS" in source
    assert '"Freshness"' in source
    assert '"Stale / Missing"' in source
    assert '"Newest important artifacts"' in source
    assert "Checked now:" in source
    assert "is {item['state']}" in source
    assert '"AGING INPUTS"' in source
    assert "aging_count" in source
    assert "Calibrated deployable model does not beat" in source
    assert "elif not deploy_ready and beats" in source


def test_smoke_button_uses_safe_launcher():
    source = _source()
    assert '"ops\\\\run_safe.py"' in source
    assert '"--task"' in source
    assert '"dashboard-smoke-verifier"' in source
    assert '"--gpu-mib"' in source


def test_dashboard_has_ai_gpu_snapshot_button():
    source = _source()
    assert "def build_ai_snapshot(" in source
    assert "def gpu_snapshot_frame(" in source
    assert "def ai_process_frame(" in source
    assert "def run_metrics_frame(" in source
    assert 'st.button("Snapshot"' in source
    assert "AI / GPU Snapshot" in source
    assert "Active Work" in source
    assert "vram_used_mib" in source


def test_dashboard_snapshot_explains_live_model_runs():
    source = _source()
    assert "RUN_EXPLAINERS" in source
    assert "def render_run_cards(" in source
    assert "def model_run_snapshot_frame(" in source
    assert "Running Models" in source
    assert "What we expect to learn" in source
    assert "masked denoising autoencoder" in source
    assert "weak binary label head" in source
    assert "4096 -> 1024 -> 128 -> 1024 -> 4096" in source
