"""
Data loading and preprocessing utilities for continual learning.
Supports streaming, synthetic data generation, and diversity-preserving sampling.

Lakehouse integration: streams silver/forecast_splits and gold/forecast_predictions Parquet files.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List, Iterator
from torch.utils.data import Dataset, DataLoader, IterableDataset
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler
import random
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pandas as pd
import numpy as np


import logging
import os

logger = logging.getLogger(__name__)

# Get project root (two directories up from this file)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class SyntheticContinualDataset(Dataset):
    """
    Generates synthetic data for continual learning benchmarks.

    Each task has distinct concept drift to test catastrophic forgetting.
    """

    def __init__(
        self,
        num_tasks: int = 10,
        samples_per_task: int = 1000,
        input_dim: int = 64,
        task_shift: float = 2.0
    ):
        self.num_tasks = num_tasks
        self.samples_per_task = samples_per_task
        self.input_dim = input_dim
        self.task_shift = task_shift

        # Generate all data upfront (small scale for demo)
        self.data, self.labels, self.tasks = self._generate_data()

    def _generate_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate synthetic continual learning dataset."""
        all_x = []
        all_y = []
        all_tasks = []

        # Base offset for each task (concept drift)
        base_shifts = torch.linspace(0, self.task_shift * (self.num_tasks - 1), self.num_tasks)

        for task_id in range(self.num_tasks):
            shift = base_shifts[task_id].item()

            # Create distinct classification task
            x, y = make_classification(
                n_samples=self.samples_per_task,
                n_features=self.input_dim,
                n_informative=self.input_dim // 2,
                n_redundant=self.input_dim // 4,
                classes=2,
                class_sep=1.5,
                random_state=task_id * 42
            )

            # Apply task-specific shift for concept drift
            x = x + shift

            all_x.append(torch.tensor(x, dtype=torch.float32))
            all_y.append(torch.tensor(y, dtype=torch.long))
            all_tasks.append(torch.full((self.samples_per_task,), task_id, dtype=torch.long))

        return torch.cat(all_x), torch.cat(all_y), torch.cat(all_tasks)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx], self.tasks[idx]

    def get_task_data(self, task_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get data for a specific task."""
        mask = (self.tasks == task_id)
        return self.data[mask], self.labels[mask]


class DataDriftGenerator:
    """
    Generates streaming data with concept drift.

    Useful for online learning scenarios where the data distribution shifts over time.
    """

    def __init__(
        self,
        input_dim: int = 128,
        drift_frequency: int = 1000,  # Steps between drift changes
        noise_level: float = 0.1
    ):
        self.input_dim = input_dim
        self.drift_frequency = drift_frequency
        self.noise_level = noise_level

        # Current concept weights
        self.weights = torch.randn(input_dim, input_dim)
        self.bias = torch.randn(input_dim)

        self.step_count = 0

    def _update_concept(self) -> None:
        """Update the underlying data distribution (concept drift)."""
        # Add drift to weights
        drift = torch.randn_like(self.weights) * 0.1
        self.weights += drift

        # Normalize to prevent explosion
        self.weights = F.normalize(self.weights, p=2, dim=1)

        # Update bias slightly
        self.bias += torch.randn_like(self.bias) * 0.01

    def generate_batch(self, batch_size: int) -> torch.Tensor:
        """Generate a batch of data with current concept."""
        # Check if concept should drift
        if self.step_count > 0 and self.step_count % self.drift_frequency == 0:
            self._update_concept()

        self.step_count += 1

        # Sample from current distribution
        x = torch.randn(batch_size, self.input_dim)
        x = x @ self.weights.t() + self.bias

        # Normalize for stable training (z-score)
        x = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-8)

        # Add noise
        x += torch.randn_like(x) * self.noise_level

        return x


class LakehouseDataLoader:
    """
    Streams actual MBARI data from lakehouse silver/forecast_splits partitions.
    
    Produces sequential time-series batches suitable for TFT input shape:
    (batch, seq_len=720, input_dim)
    
    Usage:
        loader = LakehouseDataLoader(
            parquet_path='lakehouse/silver/forecast_splits',
            context_window=720,  # 30 days hourly
            target_cols=['TSMixerx', 'y']  # Prediction targets
        )
        for batch_x, batch_target in loader.stream_batched(batch_size=32):
            # batch_x: (batch, seq_len, input_dim)
            # batch_target: (batch, forecast_horizon)
            outputs = model(batch_x, compute_loss=True, target=batch_target)
    """
    
    def __init__(
        self,
        parquet_path: str = 'lakehouse/silver/forecast_splits',
        context_window: int = 720,  # 30 days hourly
        forecast_horizon: int = 24,  # Predict next 24 hours
        target_cols: Optional[List[str]] = None,
        min_rows_per_series: int = 200,  # Minimum historical data per series (unique timestamps)
    ):
        """
        Initialize Lakehouse data loader.

        Args:
            parquet_path: Path to silver/forecast_splits directory (relative to project root)
            context_window: Number of historical timesteps (default 720h = 30 days)
            forecast_horizon: Number of future timesteps to predict
            target_cols: Column names to use as targets (if None, use TSMixerx)
            min_rows_per_series: Minimum rows required per unique_id/time series
        """
        # Convert relative path to absolute
        parquet_path = os.path.join(PROJECT_ROOT, parquet_path.lstrip('/\\'))
        
        self.parquet_path = parquet_path
        self.context_window = context_window
        self.forecast_horizon = forecast_horizon
        # Forecast the ACTUAL observed value `y` (autoregressive), NOT another model's
        # prediction column. The model sees past y, predicts future y.
        self.target_cols = target_cols or ['y']
        self.min_rows_per_series = min_rows_per_series

        # Load dataset from Parquet partitions (use filesystem directly to avoid merge issues)
        try:
            # Use pyarrow.dataset (auto-coalesces schemas for partitioned data)
            self.gold_dataset = ds.dataset(parquet_path, format='parquet',
                                          exclude_invalid_files=True)
            self.schema = self.gold_dataset.schema
            total_rows = self.gold_dataset.count_rows()
            logger.info(f"Loaded {total_rows} rows from lakehouse data")
            
            # Convert to pandas DataFrame for easier processing (only needed once)
            # Read all unique_ids using the dataset directly (PyArrow handles schema coalescing)
            table = self.gold_dataset.to_table(columns=['unique_id'])
            # Convert ChunkedArray to list properly
            import pyarrow.compute as pc
            unique_values = pc.unique(table['unique_id']).to_pylist()
            self.unique_ids = unique_values
            
            logger.info(f"Found {len(self.unique_ids)} unique series IDs")
            
            self.valid_series = self._filter_valid_series(self.unique_ids)
            logger.info(f"Found {len(self.valid_series)} valid series with sufficient historical data")
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset from {parquet_path}: {e}")

    def _get_unique_ids(self) -> List[str]:
        """Get unique series IDs from_dataset (sample for efficiency)."""
        import pyarrow.compute as pc

        # Read a sample of files and extract unique_ids
        unique_values = set()

        # Get file paths - use only valid files that can be read
        all_files = self.gold_dataset.files
        
        # Try to read from just one partition directory (they should have consistent schema)
        if not all_files:
            return []
            
        # Use first file's parent directory instead of individual files
        try:
            test_path = os.path.dirname(all_files[0])
            single_partition = ds.dataset(test_path, format='parquet')
            table = single_partition.to_table(columns=['unique_id'])
            unique_values.update(table['unique_id'].unique().tolist())
        except Exception as e1:
            # Fall back to reading sample of files
            for f in all_files[:20]:  # Sample up to 20 files
                try:
                    table = pq.read_table(f, columns=['unique_id'])
                    unique_values.update(table['unique_id'].tolist())
                except Exception as e:
                    continue
        
        return list(unique_values)[:100]  # Limit for quick test
    
    def _filter_valid_series(self, unique_ids: List[str]) -> List[str]:
        """Return only series with sufficient historical data."""
        valid = []
        for uid in unique_ids:
            try:
                # Count rows for this series - convert uid to proper type
                table = self.gold_dataset.to_table(filter=ds.field('unique_id') == str(uid))
                if len(table) >= self.min_rows_per_series:
                    valid.append(uid)
            except Exception as e:
                logger.warning(f"Could not read {uid}: {e}")
        return valid[:100]  # Limit to first 100 series for quick test
    
    def _load_time_series(self, unique_id: str) -> pd.DataFrame:
        """
        Load one series as a clean, deduplicated timeline of actual observations.

        The gold table stores many overlapping forecast runs, so each (series, ds)
        appears multiple times. The observed value `y` is identical across runs, so we
        keep the first occurrence per timestamp to recover a true hourly series.
        """
        col = self.target_cols[0]
        table = self.gold_dataset.to_table(
            filter=ds.field('unique_id') == str(unique_id),
            columns=['ds', col]
        )
        df = table.to_pandas()
        df = df.dropna(subset=[col])
        df = df.sort_values('ds').drop_duplicates(subset=['ds'], keep='first')
        return df.reset_index(drop=True)
    
    def _extract_window(
        self, 
        df: pd.DataFrame, 
        target_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract context window and forecast horizon from time series.

        Args:
            df: Sorted DataFrame with ds column
            target_idx: Index where forecast should start

        Returns:
            (context_window, forecast_target)
            - context_window shape: (seq_len=720, input_dim)
            - forecast_target shape: (forecast_horizon,)
        """
        seq_len = self.context_window
        horizon = self.forecast_horizon

        # Get total rows available
        n_rows = len(df)

        if target_idx < seq_len or target_idx + horizon > n_rows:
            return None, None  # Not enough data for this window

        # Extract context (preceding seq_len rows)
        start_idx = target_idx - seq_len
        end_idx = target_idx

        context_df = df.iloc[start_idx:end_idx]
        target_df = df.iloc[end_idx:end_idx + horizon]

        if len(context_df) < seq_len or len(target_df) < horizon:
            return None, None  # Incomplete window

        col = self.target_cols[0]

        # Input is the ACTUAL past values of the series; target is its future values.
        context_vals = pd.to_numeric(context_df[col], errors='coerce').to_numpy(dtype='float64')
        future_vals = pd.to_numeric(target_df[col], errors='coerce').to_numpy(dtype='float64')

        context_vals = np.nan_to_num(context_vals, nan=0.0, posinf=0.0, neginf=0.0)
        future_vals = np.nan_to_num(future_vals, nan=0.0, posinf=0.0, neginf=0.0)

        if len(context_vals) < seq_len or len(future_vals) < horizon:
            return None, None

        # Per-window z-score using context statistics (target normalized on the SAME
        # scale so the model learns/predicts in normalized space). Eval de-normalizes
        # with (mu, sd) to report errors in physical units.
        mu = float(context_vals.mean())
        sd = float(context_vals.std())

        # Guard degenerate windows. A near-constant context (stuck sensor or a
        # gap-filled flat segment — common in this data) has sd≈0. The previous
        # `sd + 1e-6` floor then amplified micro-noise by ~1e6×, producing inputs
        # that overflow FP16 autocast → Inf → NaN that poisons the entire run after
        # a few hundred steps. Such windows carry no forecastable signal, so drop
        # them (the caller already skips None windows) rather than normalize them.
        if not np.isfinite(sd) or sd < 1e-3 or sd < 1e-3 * abs(mu):
            return None, None

        context_features = (context_vals - mu) / sd
        target_values = (future_vals - mu) / sd

        # Belt-and-suspenders: never emit a non-finite window downstream.
        if not (np.isfinite(context_features).all() and np.isfinite(target_values).all()):
            return None, None

        return context_features, target_values
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Stream batches of sequential time-series data.
        
        Yields:
            (batch_x, batch_target)
            - batch_x: (batch, seq_len=720, input_dim=1) context window
            - batch_target: (batch, forecast_horizon) values to predict
        """
        # Shuffle series order for better training
        series_ids = self.valid_series.copy()
        random.shuffle(series_ids)
        
        for unique_id in series_ids:
            try:
                df = self._load_time_series(unique_id)
                
                # Slide window across the time series
                target_idx = self.context_window
                while target_idx + self.forecast_horizon <= len(df):
                    context, target = self._extract_window(df, target_idx)
                    
                    if context is not None and target is not None:
                        # Add feature dimension (input_dim=1 for univariate)
                        batch_x = torch.from_numpy(context).unsqueeze(-1).float()  # (seq_len, 1)
                        batch_target = torch.from_numpy(target).float()  # (forecast_horizon,)
                        
                        yield batch_x, batch_target
                    
                    target_idx += self.forecast_horizon // 2  # Overlap by 50%
                    
            except Exception as e:
                logger.warning(f"Error processing series {unique_id}: {e}")
                continue
    
    def stream_batched(
        self,
        batch_size: int = 32
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Stream batched time-series data.
        
        Args:
            batch_size: Number of sequences per batch
            
        Yields:
            (batch_x, batch_target) where:
            - batch_x: (batch_size, seq_len=720, input_dim)
            - batch_target: (batch_size, forecast_horizon)
        """
        batch_x_list = []
        batch_target_list = []
        
        for x, target in self:
            batch_x_list.append(x)
            batch_target_list.append(target)
            
            if len(batch_x_list) == batch_size:
                # Stack into batch tensors
                batch_x = torch.stack(batch_x_list)  # (batch, seq_len, input_dim)
                batch_target = torch.stack(batch_target_list)  # (batch, horizon)
                
                yield batch_x, batch_target
                
                batch_x_list = []
                batch_target_list = []


class StreamingReplayDataset(IterableDataset):
    """
    Dataset that streams data while maintaining replay buffer.

    Perfect for online continual learning scenarios.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        replay_fraction: float = 0.3,
        max_buffer_size: int = 1000
    ):
        self.base_dataset = base_dataset
        self.replay_fraction = replay_fraction
        self.max_buffer_size = max_buffer_size

        # Replay buffer (in-memory)
        self.buffer_x: List[torch.Tensor] = []
        self.buffer_y: List[torch.Tensor] = []

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:  # Single-process
            indices = list(range(len(self.base_dataset)))
            random.shuffle(indices)
        else:
            # Multi-worker sharding
            per_worker = len(self.base_dataset) // worker_info.num_workers
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, len(self.base_dataset))
            indices = list(range(start, end))

        for idx in indices:
            x, y, task_id = self.base_dataset[idx]

            # Decide if this should go to replay buffer
            if len(self.buffer_x) < self.max_buffer_size and random.random() < 0.5:
                self.buffer_x.append(x.clone())
                self.buffer_y.append(y.clone())

            yield x, y, task_id

        # Clear buffer between epochs
        self.buffer_x.clear()
        self.buffer_y.clear()


def create_dataloaders(
    dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True
) -> DataLoader:
    """Create data loader with appropriate settings for RTX 5090."""

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle if not isinstance(dataset, IterableDataset) else False,
        pin_memory=True,  # Faster GPU transfers
        persistent_workers=num_workers > 0,  # Avoids reloading data
        prefetch_factor=2  # Preload batches
    )


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    
    logger = logging.getLogger(__name__)
    
    # Quick test with synthetic data
    dataset = SyntheticContinualDataset(num_tasks=5, samples_per_task=200)

    print(f"Total samples: {len(dataset)}")
    print(f"Input dim: {dataset.input_dim}")

    # Test task-specific data retrieval
    x, y = dataset.get_task_data(0)
    print(f"Task 0 samples: {len(x)}")

    # Create dataloader
    loader = create_dataloaders(dataset, batch_size=32)

    for i, (x, y, task) in enumerate(loader):
        print(f"Batch {i}: x.shape={x.shape}, unique tasks={torch.unique(task)}")
        if i >= 2:
            break
    
    # Test LakehouseDataLoader with gold layer (5.8M rows)
    try:
        lake_loader = LakehouseDataLoader(
            parquet_path='lakehouse/gold/forecast_predictions',
            context_window=168,  # Test with smaller window first
            forecast_horizon=24,
            min_rows_per_series=200
        )
        
        print(f"Loaded {len(lake_loader.valid_series)} valid series")
        
        batch_count = 0
        for batch_x, batch_target in lake_loader.stream_batched(batch_size=8):
            print(f"Batch shape: x={batch_x.shape}, target={batch_target.shape}")
            batch_count += 1
            if batch_count >= 2:
                break
        
    except Exception as e:
        logger.warning(f"LakehouseDataLoader test failed (expected if no data): {e}")
