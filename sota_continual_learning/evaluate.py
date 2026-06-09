#!/usr/bin/env python3
"""
Evaluate a trained SOTA continual-learning checkpoint on the MBARI gold data.

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
import logging
import numpy as np
import pandas as pd
import torch

from core import ContinualLearner
from data import LakehouseDataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def rmse(a, p):
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mae(a, p):
    return float(np.mean(np.abs(a - p)))


def mape(a, p, eps=1e-6):
    return float(np.mean(np.abs(a - p) / (np.abs(a) + eps)) * 100.0)


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
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ctx, hz = args.context_window, args.horizon

    logger.info(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get('model_state', ckpt)

    model = ContinualLearner(input_dim=1, hidden_dim=args.hidden_dim, num_experts=args.num_experts).to(device)
    model.load_state_dict(state)
    model.eval()

    loader = LakehouseDataLoader(
        parquet_path=args.parquet_path,
        context_window=ctx,
        forecast_horizon=hz,
        min_rows_per_series=ctx + hz,
    )

    col = loader.target_cols[0]
    rows = []

    for uid in loader.valid_series:
        df = loader._load_time_series(uid)
        vals = pd.to_numeric(df[col], errors='coerce').to_numpy(dtype='float64')
        vals = vals[np.isfinite(vals)]
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

        m_a, m_p, n_p = [], [], []  # actual, model pred, naive pred (all raw units)
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
            n_p.append(context[-hz:])

        x = torch.tensor(np.stack(xs), dtype=torch.float32, device=device).unsqueeze(-1)  # (W, ctx, 1)
        with torch.no_grad():
            pred_norm, _ = model(x)  # (W, 24)
        pred_norm = pred_norm[:, :hz].float().cpu().numpy()
        mus = np.array(mus)[:, None]; sds = np.array(sds)[:, None]
        pred_raw = pred_norm * sds + mus

        a = np.concatenate(m_a)
        mp = pred_raw.reshape(-1)
        npd = np.concatenate(n_p)

        r_m, r_n = rmse(a, mp), rmse(a, npd)
        skill = 1.0 - r_m / r_n if r_n > 0 else float('nan')
        rows.append({
            'series': uid, 'windows': len(ends),
            'rmse_model': r_m, 'rmse_naive': r_n, 'skill_vs_naive': skill,
            'mae_model': mae(a, mp), 'mape_model_%': mape(a, mp),
            'mape_naive_%': mape(a, npd),
        })

    if not rows:
        logger.error("No series produced evaluable windows.")
        return

    res = pd.DataFrame(rows).sort_values('skill_vs_naive', ascending=False)
    pd.set_option('display.width', 160)
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.float_format', lambda v: f'{v:.4f}')

    print("\n===== PER-SERIES FORECAST SKILL (physical units, vs seasonal-naive) =====")
    print(res.to_string(index=False))

    wins = int((res['skill_vs_naive'] > 0).sum())
    print("\n===== SUMMARY =====")
    print(f"Series evaluated         : {len(res)}")
    print(f"Beat seasonal-naive      : {wins}/{len(res)} ({100*wins/len(res):.0f}%)")
    print(f"MEDIAN skill vs naive    : {res['skill_vs_naive'].median():+.4f}  "
          f"(>0 means better than same-hour-yesterday)")
    print(f"MEAN   skill vs naive    : {res['skill_vs_naive'].mean():+.4f}  (reference only)")
    print(f"Median model MAPE        : {res['mape_model_%'].median():.2f}%")
    print(f"Median naive MAPE        : {res['mape_naive_%'].median():.2f}%")


if __name__ == '__main__':
    main()
