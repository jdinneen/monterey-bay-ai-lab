from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from research.model_lab.guided_interaction_search import SearchConfig, numeric_feature_frame, run_search


def test_guided_interaction_search_finds_bounded_pair_signal(tmp_path: Path) -> None:
    rows = []
    for i in range(240):
        a = 1.0 if i % 4 in {0, 1} else 0.0
        b = 1.0 if i % 6 in {0, 1, 2} else 0.0
        y = int(a * b > 0)
        rows.append(
            {
                "sample_date": f"2024-01-{(i % 28) + 1:02d}",
                "target": y,
                "a": a,
                "b": b,
                "noise": float(i % 7),
            }
        )
    path = tmp_path / "toy.parquet"
    out = tmp_path / "out"
    pd.DataFrame(rows).to_parquet(path, index=False)

    summary = run_search(
        path,
        out,
        SearchConfig(
            target="target",
            max_rows=240,
            top_features=3,
            top_triple_features=0,
            max_pair_candidates=3,
            max_triple_candidates=0,
            min_valid_rows=20,
        ),
    )

    assert summary["destructive"] is False
    assert summary["pair_candidates_scored"] > 0
    assert summary["best_interaction_ap"] is not None
    assert (out / "top_interactions.csv").exists()

    top = pd.read_csv(out / "top_interactions.csv")
    assert "a*b" in set(top["interaction"]) or "b*a" in set(top["interaction"])
    payload = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert "bounded exhaustive" in payload["honesty_note"]


def test_numeric_feature_frame_drops_obvious_leakage_names_by_default() -> None:
    df = pd.DataFrame(
        {
            "target": [0, 1] * 6,
            "enterococcus": [1.0, 999.0] * 6,
            "model_probability": [0.1, 0.9] * 6,
            "rain_3d": [0.0, 1.0, 0.2, 0.3] * 3,
        }
    )

    safe = numeric_feature_frame(df, "target", None, SearchConfig(target="target"))
    unsafe = numeric_feature_frame(
        df,
        "target",
        None,
        SearchConfig(target="target", allow_leakage_named_features=True),
    )

    assert list(safe.columns) == ["rain_3d"]
    assert "enterococcus" in unsafe.columns
    assert "model_probability" in unsafe.columns
