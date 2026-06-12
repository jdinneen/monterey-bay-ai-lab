"""Smoke example for the Monterey Bay AI Lab experiment tracker.

Run from the repository root:
    python mbal_experiments\\examples\\smoke_record_run.py
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mbal_experiments import ExperimentTracker  # noqa: E402


def main() -> None:
    tracker = ExperimentTracker()
    record = tracker.record_run(
        name="smoke-xgboost-mbal-forecast",
        metrics={
            "rmse": 0.211,
            "mae": 0.146,
            "r2": 0.542,
            "persistence_rmse": 0.268,
        },
        params={
            "model_family": "xgboost",
            "tree_method": "hist",
            "device": "cuda",
            "forecast_horizon_hours": 24,
        },
        tags={
            "station": "M1",
            "target": "temp_d100p0",
            "validation": "walk_forward",
        },
        dataset_paths=[
            "mbal_history/opendap/*.parquet",
            "mbal_sota_results/*.csv",
        ],
        notes="Example only; metrics are illustrative smoke values.",
    )
    print(f"Recorded {record.run_id}")
    print(record.run_dir)


if __name__ == "__main__":
    main()

