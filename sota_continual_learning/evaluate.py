#!/usr/bin/env python3
"""
Evaluate a trained SOTA continual-learning checkpoint on the MBAL gold data.

Produces real forecast skill in physical units: per-series RMSE / MAE / MAPE for the
model AND for the seasonal-naive (same-hour-yesterday) baseline, plus skill =
1 - rmse_model / rmse_naive. Per AGENTS.md, the bar a forecaster must clear is the
better-of-naive baseline, and skill is aggregated with the MEDIAN across series.

Note on leakage: training streams the whole series, so the most-recent windows used
here were also seen in training. This is an in-sample sanity eval vs a parameter-free
baseline, not a strict temporal holdout. A model that cannot beat naive even in-sample
is a clear negative result; a win should be read with that caveat.

Usage:
    python sota_continual_learning/evaluate.py \
        --checkpoint sota_continual_learning/output_production/checkpoints/final.pt \
        --context-window 168 --windows-per-series 8
"""

import argparse
import json
import logging
import numpy as np
import pandas as pd
import polars as pl
import torch
from pathlib import Path

from core import ContinualLearner

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def rmse(a, p):
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mae(a, p):
    return float(np.mean(np.abs(a - p)))


def mape(a, p, eps=1e-6):
    return float(np.mean(np.abs(a - p) / (np.abs(a) + eps)) * 100.0)


def default_output_json(checkpoint: str) -> Path:
    ckpt = Path(checkpoint)
    if ckpt.parent.name == "checkpoints":
        return ckpt.parent.parent / "eval" / "eval_summary.json"
    return ckpt.parent / "eval_summary.json"


def load_series_frames(parquet_path: str, target_col: str, max_series: int | None = None) -> list[tuple[str, np.ndarray]]:
    """Load deduplicated physical target series from gold parquet partitions."""
    root = Path(parquet_path)
    files = [root] if root.is_file() else sorted(root.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {parquet_path}")

    frames = []
    for file_path in files:
        try:
            frame = pl.read_parquet(file_path, columns=["unique_id", "ds", target_col])
        except Exception as exc:
            logger.warning("Skipping %s: %s", file_path, exc)
            continue
        if target_col != "y":
            frame = frame.rename({target_col: "y"})
        frames.append(frame)

    if not frames:
        raise RuntimeError(f"No readable parquet files with unique_id, ds, {target_col}")

    data = (
        pl.concat(frames, how="vertical_relaxed")
        .drop_nulls(["unique_id", "ds", "y"])
        .unique(subset=["unique_id", "ds"], keep="last")
        .sort(["unique_id", "ds"])
    )

    series = []
    for name, group in data.group_by("unique_id", maintain_order=True):
        uid = name[0] if isinstance(name, tuple) else name
        vals = group["y"].to_numpy().astype("float64")
        vals = vals[np.isfinite(vals)]
        series.append((str(uid), vals))
        if max_series and len(series) >= max_series:
            break
    return series


def finite_skill(model_rmse: float, baseline_rmse: float) -> float:
    return 1.0 - model_rmse / baseline_rmse if baseline_rmse > 0 else float("nan")


def main():
    parser = argparse.ArgumentParser(description='Evaluate SOTA continual-learning checkpoint')
    parser.add_argument('--checkpoint', type=str,
                        default='sota_continual_learning/output_production/checkpoints/final.pt')
    parser.add_argument('--parquet-path', type=str, default='lakehouse/gold/forecast_predictions')
    parser.add_argument('--context-window', type=int, default=168)
    parser.add_argument('--horizon', type=int, default=24, help='Must match the model forecast head (24)')
    parser.add_argument('--windows-per-series', type=int, default=8,
                        help='Number of most-recent non-overlapping windows per series to score')
    parser.add_argument('--hidden-dim', type=int, default=1024)
    parser.add_argument('--num-experts', type=int, default=32)
    parser.add_argument('--target-col', type=str, default='y')
    parser.add_argument('--max-series', type=int, default=0,
                        help='Optional smoke limit for fast evaluator checks')
    parser.add_argument('--output-json', type=str, default=None,
                        help='Where to persist eval results; defaults beside the checkpoint')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ctx, hz = args.context_window, args.horizon

    logger.info(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get('model_state', ckpt)

    model = ContinualLearner(input_dim=1, hidden_dim=args.hidden_dim, num_experts=args.num_experts).to(device)
    
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        logger.error("Failed to load checkpoint. Note: Old MoE/Latent checkpoints are fundamentally incompatible with the new High-Signal architecture.")
        raise
    model.eval()

    series_frames = load_series_frames(
        args.parquet_path,
        target_col=args.target_col,
        max_series=args.max_series or None,
    )
    rows = []

    logger.info("Loaded %d deduplicated series from %s", len(series_frames), args.parquet_path)
    for uid, vals in series_frames:
        if len(vals) < ctx + hz:
            continue

        # Most-recent non-overlapping windows: target blocks end at the series end.
        ends = []
        e = len(vals)
        while len(ends) < args.windows_per_series and e - hz - ctx >= 0:
            ends.append(e)
            e -= hz
        ends = ends[::-1]
        if not ends:
            continue

        m_a, seasonal_p, persistence_p = [], [], []  # all raw units
        xs, mus, sds = [], [], []
        for end in ends:
            t0 = end - hz
            context = vals[t0 - ctx:t0]
            future = vals[t0:end]
            mu, sd = context.mean(), context.std() + 1e-6
            xs.append((context - mu) / sd)
            mus.append(mu); sds.append(sd)
            m_a.append(future)
            # seasonal-naive: same-hour-yesterday == last `hz` context values (lag 24)
            seasonal_p.append(context[-hz:])
            persistence_p.append(np.repeat(context[-1], hz))

        x = torch.tensor(np.stack(xs), dtype=torch.float32, device=device).unsqueeze(-1)  # (W, ctx, 1)
        with torch.no_grad():
            pred_norm, _ = model(x)  # (W, 24)
        pred_norm = pred_norm[:, :hz].float().cpu().numpy()
        mus = np.array(mus)[:, None]; sds = np.array(sds)[:, None]
        pred_raw = pred_norm * sds + mus

        a = np.concatenate(m_a)
        mp = pred_raw.reshape(-1)
        sp = np.concatenate(seasonal_p)
        pp = np.concatenate(persistence_p)

        r_m, r_s, r_p = rmse(a, mp), rmse(a, sp), rmse(a, pp)
        r_b = min(r_s, r_p)
        rows.append({
            'series': uid, 'windows': len(ends),
            'rmse_model': r_m,
            'rmse_persistence': r_p,
            'rmse_seasonal_naive': r_s,
            'rmse_best_naive': r_b,
            'skill_vs_persistence': finite_skill(r_m, r_p),
            'skill_vs_seasonal_naive': finite_skill(r_m, r_s),
            'skill_vs_best_naive': finite_skill(r_m, r_b),
            'mae_model': mae(a, mp), 'mape_model_%': mape(a, mp),
            'mape_persistence_%': mape(a, pp),
            'mape_seasonal_naive_%': mape(a, sp),
        })

    if not rows:
        logger.error("No series produced evaluable windows.")
        return

    res = pd.DataFrame(rows).sort_values('skill_vs_best_naive', ascending=False)
    pd.set_option('display.width', 160)
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.float_format', lambda v: f'{v:.4f}')

    print("\n===== PER-SERIES FORECAST SKILL (physical units, vs best naive) =====")
    print(res.to_string(index=False))

    wins = int((res['skill_vs_best_naive'] > 0).sum())
    summary = {
        "checkpoint": args.checkpoint,
        "parquet_path": args.parquet_path,
        "context_window": ctx,
        "horizon": hz,
        "windows_per_series": args.windows_per_series,
        "series_evaluated": int(len(res)),
        "beat_best_naive": wins,
        "beat_best_naive_pct": float(100 * wins / len(res)),
        "median_skill_vs_best_naive": float(res['skill_vs_best_naive'].median()),
        "mean_skill_vs_best_naive": float(res['skill_vs_best_naive'].mean()),
        "median_skill_vs_seasonal_naive": float(res['skill_vs_seasonal_naive'].median()),
        "median_skill_vs_persistence": float(res['skill_vs_persistence'].median()),
        "median_model_mape_pct": float(res['mape_model_%'].median()),
        "median_persistence_mape_pct": float(res['mape_persistence_%'].median()),
        "median_seasonal_naive_mape_pct": float(res['mape_seasonal_naive_%'].median()),
        "note": "In-sample sanity eval against physical-unit naive baselines, not a temporal holdout.",
    }

    print("\n===== SUMMARY =====")
    print(f"Series evaluated         : {summary['series_evaluated']}")
    print(f"Beat best-naive          : {wins}/{len(res)} ({summary['beat_best_naive_pct']:.0f}%)")
    print(f"MEDIAN skill vs best     : {summary['median_skill_vs_best_naive']:+.4f}  "
          f"(>0 means better than both persistence and same-hour-yesterday)")
    print(f"MEAN   skill vs best     : {summary['mean_skill_vs_best_naive']:+.4f}  (reference only)")
    print(f"Median model MAPE        : {summary['median_model_mape_pct']:.2f}%")
    print(f"Median persistence MAPE  : {summary['median_persistence_mape_pct']:.2f}%")
    print(f"Median seasonal MAPE     : {summary['median_seasonal_naive_mape_pct']:.2f}%")

    output_json = Path(args.output_json) if args.output_json else default_output_json(args.checkpoint)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps({"summary": summary, "rows": res.to_dict(orient="records")}, indent=2),
        encoding="utf-8",
    )
    logger.info("Wrote evaluation results to %s", output_json)


if __name__ == '__main__':
    main()

