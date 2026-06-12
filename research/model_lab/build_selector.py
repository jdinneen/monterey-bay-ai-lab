"""Honest per-series model selector with a foundation (Chronos) candidate.

Pipeline:
  1) Reuse deep cv_predictions (exact 24 series x 52 weekly cutoffs x 168 steps).
  2) Regenerate Chronos-Bolt row-level preds on the SAME (uid, cutoff, ds) grid (CPU).
  3) Candidates per (series, horizon): 6 deep models + chronos + persistence + seasonal-naive.
  4) Select the best candidate per (series, horizon) on EARLY cutoffs (validation),
     measure its skill on the held-out LATE cutoffs (test) vs the best naive baseline.
     No hindsight. Report selector vs always-persistence vs oracle.

Scoring matches mbal_neural_forecast.evaluate(): observed target points only;
persistence = obs at origin; seasonal-naive = obs at ds-24h; all on a common subset.
"""
import os, glob, sys, time
import numpy as np
import pandas as pd

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # force CPU: avoid GPU contention with running jobs
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# Lives at <root>/research/model_lab/; resolve project root 3 levels up (or env override).
ROOT = os.environ.get("MBAL_PROJECT_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NNC, NNR = os.path.join(ROOT, "nn_cache"), os.path.join(ROOT, "nn_results")
HORIZONS = [1, 6, 24, 72, 168]
CORE = ["itransformer", "nbeatsx", "nhits", "patchtst", "tft", "tsmixerx"]
CHRONOS_OUT = os.path.join(NNR, "chronos_full", "cv_predictions.parquet")
pd.set_option("display.width", 170); pd.set_option("display.max_rows", 300)

long_df = pd.read_parquet(os.path.join(NNC, "long_v2_past_only_fill_origin_observed_full.parquet"))
mask = pd.read_parquet(os.path.join(NNC, "mask_v2_past_only_fill_origin_observed_full.parquet"))
long_df["ds"] = pd.to_datetime(long_df["ds"]); mask["ds"] = pd.to_datetime(mask["ds"])
obs = long_df.merge(mask, on=["unique_id", "ds"], how="left")
obs = obs[obs["observed"].fillna(False)][["unique_id", "ds", "y"]]
obs_val = obs.set_index(["unique_id", "ds"])["y"]

def load_preds(md):
    df = pd.read_parquet(os.path.join(md, "cv_predictions.parquet"))
    df["ds"] = pd.to_datetime(df["ds"]); df["cutoff"] = pd.to_datetime(df["cutoff"])
    pcol = [c for c in df.columns if c not in ("unique_id","ds","cutoff","y","observed","cache_version")][0]
    return df, pcol

dirs = {os.path.basename(os.path.dirname(p)): os.path.dirname(p)
        for p in glob.glob(os.path.join(NNR, "*", "cv_predictions.parquet"))
        if os.path.basename(os.path.dirname(p)) in CORE}
print("Core deep models:", list(dirs))

# canonical grid from patchtst (any core model has the same grid)
grid, _ = load_preds(dirs["patchtst"])
grid = grid[["unique_id", "ds", "cutoff", "y"]].copy()

# ---------- Step 2: Chronos-Bolt row preds on the same grid (CPU) ----------
def gen_chronos(grid):
    if os.path.exists(CHRONOS_OUT):
        c = pd.read_parquet(CHRONOS_OUT); c["ds"] = pd.to_datetime(c["ds"]); c["cutoff"] = pd.to_datetime(c["cutoff"])
        if set(c["unique_id"].unique()) >= set(grid["unique_id"].unique()):
            print(f"[chronos] reuse cached {CHRONOS_OUT} ({len(c)} rows)"); return c
    import torch
    from chronos import BaseChronosPipeline
    t0 = time.time()
    pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-base", device_map="cpu", torch_dtype=torch.float32)
    H = max(HORIZONS)
    rows = []
    uids = sorted(grid["unique_id"].unique())
    for ui, uid in enumerate(uids):
        ys = long_df[long_df.unique_id == uid].set_index("ds")["y"].sort_index()
        ys = ys.ffill().fillna(ys.mean()).fillna(0.0)
        g = grid[grid.unique_id == uid]
        cutoffs = sorted(g["cutoff"].unique())
        for i in range(0, len(cutoffs), 64):
            batch = cutoffs[i:i+64]
            ctxs = []
            for t in batch:
                ctx = ys.loc[:t].tail(512).to_numpy(dtype=np.float32)
                if len(ctx) < 16:
                    ctx = np.pad(ctx, (16-len(ctx), 0), mode="edge") if len(ctx) else np.zeros(16, np.float32)
                ctxs.append(torch.from_numpy(ctx))
            with torch.no_grad():
                q, _ = pipe.predict_quantiles(ctxs, prediction_length=H, quantile_levels=[0.5])
            path = q[:, :, 0].cpu().numpy()  # (batch, H) median path
            for bi, t in enumerate(batch):
                for h in HORIZONS:
                    rows.append((uid, pd.Timestamp(t) + pd.Timedelta(hours=h), pd.Timestamp(t), path[bi, h-1]))
        print(f"[chronos] {ui+1}/{len(uids)} {uid}  ({time.time()-t0:.0f}s)", flush=True)
    c = pd.DataFrame(rows, columns=["unique_id", "ds", "cutoff", "chronos"])
    os.makedirs(os.path.dirname(CHRONOS_OUT), exist_ok=True)
    c.to_parquet(CHRONOS_OUT, index=False)
    print(f"[chronos] wrote {CHRONOS_OUT} ({len(c)} rows, {time.time()-t0:.0f}s)")
    return c

chronos = gen_chronos(grid)

# ---------- assemble candidate matrix on the grid ----------
cand = grid.copy()
for name, md in dirs.items():
    df, pcol = load_preds(md)
    cand = cand.merge(df[["unique_id","ds","cutoff",pcol]].rename(columns={pcol:name}), on=["unique_id","ds","cutoff"], how="left")
cand = cand.merge(chronos, on=["unique_id","ds","cutoff"], how="left")
cand["persistence"] = obs_val.reindex(pd.MultiIndex.from_arrays([cand.unique_id, cand.cutoff])).to_numpy()
cand["seasonal"]    = obs_val.reindex(pd.MultiIndex.from_arrays([cand.unique_id, cand.ds - pd.Timedelta(hours=24)])).to_numpy()
cand = cand.merge(mask, on=["unique_id","ds"], how="left")
cand = cand[cand["observed"].fillna(False)].copy()
cand["horizon_h"] = ((cand.ds - cand.cutoff)/pd.Timedelta(hours=1)).round().astype(int)
cand = cand[cand.horizon_h.isin(HORIZONS)]

CANDS = list(dirs) + ["chronos", "persistence", "seasonal"]
NAIVE = ["persistence", "seasonal"]
cand = cand.dropna(subset=CANDS + ["y"])  # common subset where every candidate exists

# early/late split by cutoff time (60/40)
cutoff_order = np.sort(cand["cutoff"].unique())
split = cutoff_order[int(len(cutoff_order)*0.6)]
cand["fold"] = np.where(cand["cutoff"] < split, "val", "test")
print(f"\nFolds: {len(cutoff_order)} cutoffs -> val<{pd.Timestamp(split).date()} | test>=")

def rmse(g, col): return float(np.sqrt(np.mean((g[col].to_numpy()-g["y"].to_numpy())**2)))

# ---------- selector ----------
recs = []
for (uid, hh), g in cand.groupby(["unique_id", "horizon_h"]):
    val, test = g[g.fold=="val"], g[g.fold=="test"]
    if len(val) < 3 or len(test) < 3:
        continue
    val_rmse = {c: rmse(val, c) for c in CANDS}
    pick = min(val_rmse, key=val_rmse.get)                       # selected on validation only
    test_rmse = {c: rmse(test, c) for c in CANDS}
    best_naive = min(test_rmse[n] for n in NAIVE)
    oracle = min(test_rmse, key=test_rmse.get)                   # hindsight upper bound
    recs.append(dict(unique_id=uid, horizon_h=hh, pick=pick,
                     sel_skill=100*(1-test_rmse[pick]/best_naive),
                     persist_skill=100*(1-test_rmse["persistence"]/best_naive),
                     oracle_skill=100*(1-test_rmse[oracle]/best_naive), oracle=oracle,
                     n_test=len(test)))
R = pd.DataFrame(recs)

print("\n" + "="*78)
print("SELECTOR RESULT  — held-out TEST skill vs BEST-NAIVE (median across series), %")
print("(selector picks per series on validation only; no hindsight)")
print("="*78)
tab = R.groupby("horizon_h")[["sel_skill","persist_skill","oracle_skill"]].median().reindex(HORIZONS).round(1)
tab.columns = ["selector","always-persistence","oracle(hindsight)"]
print(tab.to_string())
print(f"\nSeries where selector beats best-naive (skill>0), by horizon:")
print((R.assign(w=R.sel_skill>0).groupby("horizon_h").w.mean()*100).reindex(HORIZONS).round(0).to_string())

print("\n" + "="*78)
print("WHAT THE SELECTOR PICKS (count), by horizon  — does it fall back to seasonal at long h?")
print("="*78)
print(pd.crosstab(R.horizon_h, R.pick).reindex(HORIZONS).fillna(0).astype(int).to_string())

print("\n" + "="*78)
print("PER-SERIES @6h (the product horizon): selected model + test skill vs best-naive")
print("="*78)
s6 = R[R.horizon_h==6].sort_values("sel_skill", ascending=False)
print(s6[["unique_id","pick","sel_skill","persist_skill","oracle","oracle_skill"]].round(1).to_string(index=False))
print(f"\n@6h: selector median {s6.sel_skill.median():.1f}% | beats best-naive in {int((s6.sel_skill>0).sum())}/{len(s6)} series"
      f" | oracle ceiling {s6.oracle_skill.median():.1f}%")
