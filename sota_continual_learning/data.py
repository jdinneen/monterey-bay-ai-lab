"""
High-Performance Data Ingestion for Monterey Bay AI Lab.
Optimized for RTX 5090 Throughput.

Implements:
- Polars-based vectorized Parquet scanning.
- Sharded IterableDataset for multi-worker prefetching.
- CUDA-stream asynchronous transfers (via CudaPrefetcher).
- Google-quality telemetry (throughput tracking).
"""

import torch
import polars as pl
import numpy as np
import os
import random
import logging
import time
from typing import Tuple, Optional, List, Iterator, Dict
from torch.utils.data import IterableDataset, DataLoader
from pathlib import Path
from collections import OrderedDict

logger = logging.getLogger(__name__)

# Constants
PROJECT_ROOT = Path(os.environ.get("MBAL_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()

class PolarsFastDataset(IterableDataset):
    """
    High-performance time-series dataset using Polars for vectorized scanning.
    Shards partitions across workers for parallel prefetching.
    """
    def __init__(
        self,
        parquet_path: str,
        context_window: int = 720,
        forecast_horizon: int = 24,
        target_col: str = 'y',
        stride: int = 12,
        shard_seed: int = 42,
        eval_holdout_steps: int = 192
    ):
        self.parquet_path = Path(PROJECT_ROOT) / parquet_path.lstrip('/\\')
        self.context_window = context_window
        self.forecast_horizon = forecast_horizon
        self.target_col = target_col
        self.stride = stride
        self.shard_seed = shard_seed
        self.eval_holdout_steps = eval_holdout_steps
        
        # Discovery
        self.all_files = list(self.parquet_path.rglob("*.parquet"))
        if not self.all_files:
            raise FileNotFoundError(f"No parquet files found at {self.parquet_path}")
            
        logger.info(f"Initialized PolarsFastDataset with {len(self.all_files)} partitions.")

    def _get_worker_shards(self) -> List[Path]:
        """Shard files across workers to avoid redundant processing."""
        worker_info = torch.utils.data.get_worker_info()
        
        # Shuffle for diverse coverage across workers
        shuffled = self.all_files.copy()
        random.seed(self.shard_seed)
        random.shuffle(shuffled)
        
        if worker_info is None:
            return shuffled
        
        # Partition
        return shuffled[worker_info.id::worker_info.num_workers]

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        shards = self._get_worker_shards()
        total_window = self.context_window + self.forecast_horizon
        
        for file_path in shards:
            try:
                # Polars: Scan Parquet lazily
                q = pl.scan_parquet(file_path).select([
                    'unique_id', 'ds', self.target_col
                ]).drop_nulls().sort(['unique_id', 'ds'])
                
                df = q.collect(streaming=True)
                
                # Group by unique_id
                for name_tuple, group in df.group_by('unique_id'):
                    vals = group[self.target_col].to_numpy().astype(np.float32)
                    
                    if self.eval_holdout_steps > 0:
                        vals = vals[:-self.eval_holdout_steps]
                        
                    if len(vals) < total_window:
                        continue
                    
                    # Slide windows
                    for i in range(0, len(vals) - total_window + 1, self.stride):
                        window = vals[i : i + total_window]
                        context = window[:self.context_window]
                        future = window[self.context_window:]
                        
                        # SCIENTIFIC FIX: Local per-window normalization (Matches evaluate.py)
                        # Prevents the "training/eval target mismatch"
                        mu = context.mean()
                        std = context.std() + 1e-6
                        
                        x_norm = (context - mu) / std
                        y_norm = (future - mu) / std
                        
                        # SCIENTIFIC FIX: Prevent normalization instability / bad outliers
                        # If a sensor flatlines (std ~ 0), the next real value explodes to infinity.
                        y_norm = np.clip(y_norm, -20.0, 20.0)
                        
                        x = torch.from_numpy(x_norm).unsqueeze(-1)
                        y = torch.from_numpy(y_norm)
                        yield x, y
                        
            except Exception as e:
                logger.warning(f"Worker failed to read {file_path}: {e}")
                continue

class CudaPrefetcher:
    """
    Overlaps Host-to-Device transfers with GPU compute using a dedicated CUDA stream.
    The 'Firehose' for RTX 5090.
    """
    def __init__(self, loader: DataLoader, device: torch.device, ema_alpha: float = 0.05):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        self.next_batch = None
        
        # CRITIC FIX: Exponential Moving Average (EMA) for real-time throughput telemetry
        self.ema_alpha = ema_alpha
        self.throughput_ema = 0.0
        self.last_time = time.time()
        
        self.metrics = {"batches": 0, "start_time": time.time()}
        self._preload()

    def _preload(self):
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return

        with torch.cuda.stream(self.stream):
            # Asynchronous non-blocking transfer
            self.next_batch = [b.to(device=self.device, non_blocking=True) for b in self.next_batch]

    def next(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        # Wait for the HtoD stream to finish before returning to the compute stream
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        
        if batch is not None:
            # Update EMA Throughput
            now = time.time()
            dt = now - self.last_time
            current_throughput = 1.0 / (dt + 1e-6)
            if self.throughput_ema == 0:
                self.throughput_ema = current_throughput
            else:
                self.throughput_ema = (1 - self.ema_alpha) * self.throughput_ema + self.ema_alpha * current_throughput
            self.last_time = now
            
            # Record that this batch is being used in the current stream
            for b in batch:
                b.record_stream(torch.cuda.current_stream())
            self.metrics["batches"] += 1
        
        self._preload()
        return batch

    def get_throughput(self) -> float:
        """Returns batches per second (Exponential Moving Average)."""
        return self.throughput_ema

def create_high_performance_loader(
    parquet_path: str,
    batch_size: int = 1024,
    num_workers: int = 4,
    context_window: int = 720,
    forecast_horizon: int = 24,
    stride: int = 1,
    eval_holdout_steps: int = 192
) -> DataLoader:
    """Factory for the high-performance data pipeline."""
    dataset = PolarsFastDataset(
        parquet_path=parquet_path,
        context_window=context_window,
        forecast_horizon=forecast_horizon,
        stride=stride,
        eval_holdout_steps=eval_holdout_steps
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,  # Critical for CUDA prefetching
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False
    )

class LakehouseDataLoader:
    """
    Legacy compatibility wrapper for LakehouseDataLoader using the high-performance Polars engine.
    """
    def __init__(
        self,
        parquet_path: str = 'lakehouse/gold/forecast_predictions',
        context_window: int = 720,
        forecast_horizon: int = 24,
        target_cols: Optional[List[str]] = None,
        min_rows_per_series: int = 800,
    ):
        self.target_cols = target_cols if target_cols else ['y']
        self.dataset = PolarsFastDataset(
            parquet_path=parquet_path,
            context_window=context_window,
            forecast_horizon=forecast_horizon,
            target_col=self.target_cols[0]
        )
        # For legacy scripts that check this property
        self.valid_series = self.dataset.all_files

    def stream_batched(self, batch_size: int = 32) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Legacy interface for batched streaming."""
        loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True
        )
        for batch in loader:
            yield batch
