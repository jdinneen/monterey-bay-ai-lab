import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import re
import os
from typing import Optional, List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

PROJECT_ROOT = os.environ.get("MBAL_PROJECT_ROOT", os.getcwd())

# Mission Control directory for TensorBoard logs.
LOG_DIR = os.path.join(PROJECT_ROOT, "mbal_forecast_v2_results", "logs")

class FeatureSemanticEncoder(nn.Module):
    """
    Upgraded: Encodes the semantic meaning of MBAL features using 
    learned variable embeddings and Sinusoidal Depth Encodings.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        # Learned embeddings for variable types
        self.var_types = ['temp', 'sal', 'air', 'wind', 'current', 'eastward', 'northward', 'relh', 'other']
        self.var_embedding = nn.Embedding(len(self.var_types), d_model // 2)
        
    def _sinusoidal_depth_encoding(self, depths: torch.Tensor) -> torch.Tensor:
        D = self.d_model - (self.d_model // 2)
        div_term = torch.exp(torch.arange(0, D, 2).float() * -(np.log(1000.0) / D)).to(depths.device)
        pe = torch.zeros(depths.size(0), D, device=depths.device)
        pe[:, 0::2] = torch.sin(depths * div_term)
        pe[:, 1::2] = torch.cos(depths * div_term)
        return pe

    def forward(self, semantic_info: List[Tuple[str, float]]):
        device = self.var_embedding.weight.device
        var_indices = []
        depth_values = []
        for v_type, depth in semantic_info:
            var_indices.append(self.var_types.index(v_type if v_type in self.var_types else 'other'))
            depth_values.append([depth])
        var_idx_t = torch.tensor(var_indices, device=device)
        depth_val_t = torch.tensor(depth_values, device=device, dtype=torch.float32)
        v_emb = self.var_embedding(var_idx_t)
        d_emb = self._sinusoidal_depth_encoding(depth_val_t)
        return torch.cat([v_emb, d_emb], dim=-1)

class MBAL_Ocean_Transformer(nn.Module):
    def __init__(
        self, 
        n_features: int, 
        semantic_info: List[Tuple[str, float]],
        d_model: int = 128, 
        n_heads: int = 8, 
        e_layers: int = 3, 
        d_ff: int = 256, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.n_features = n_features
        self.feature_projection = nn.Linear(1, d_model)
        self.semantic_encoder = FeatureSemanticEncoder(d_model)
        self.semantic_info = semantic_info
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, 
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.feature_projection(x)
        s_emb = self.semantic_encoder(self.semantic_info)
        x = x + s_emb.unsqueeze(0)
        x = self.transformer(x)
        x = torch.mean(x, dim=1)
        return self.output_head(x)


class MBAL_MoE_Transformer(nn.Module):
    """
    Research-only Transformer with a DynamicMoE head.

    This class remains importable because existing NeuralForecast wrappers and
    tests depend on the model contract. Production promotion is still gated by
    shared-split evidence; restoring the class does not make MoE production-ready.
    """
    def __init__(
        self,
        n_features: int,
        semantic_info: List[Tuple[str, float]],
        h: int = 1,
        d_model: int = 128,
        n_heads: int = 8,
        e_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        num_experts: int = 8,
    ):
        super().__init__()
        from ops.mbal_sota_utils import DynamicMoE, MoEConfig

        self.h = h
        self.n_features = n_features
        self.feature_projection = nn.Linear(1, d_model)
        self.semantic_encoder = FeatureSemanticEncoder(d_model)
        self.semantic_info = semantic_info

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)

        moe_config = MoEConfig(
            num_experts=num_experts,
            hidden_dim=d_model,
            expert_hidden_dim=d_ff,
            num_regular_experts=1,
        )
        self.moe_head = DynamicMoE(moe_config)
        self.final_proj = nn.Linear(d_model, h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        x = self.feature_projection(x)
        s_emb = self.semantic_encoder(self.semantic_info)
        x = x + s_emb.unsqueeze(0)
        x = self.transformer(x)
        x, _ = self.moe_head(x)
        x = torch.mean(x, dim=1)
        return self.final_proj(x)


def parse_mbal_feature_semantics(columns: List[str]) -> List[Tuple[str, float]]:
    semantics = []
    for col in columns:
        v_type = 'other'
        depth = 0.0
        if 'temp' in col: v_type = 'temp'
        elif 'sal' in col: v_type = 'sal'
        elif 'air_temp' in col: v_type = 'air'
        elif 'air_pres' in col: v_type = 'air'
        elif 'wind' in col: v_type = 'wind'
        elif 'current' in col: v_type = 'current'
        elif 'eastward' in col: v_type = 'eastward'
        elif 'northward' in col: v_type = 'northward'
        elif 'relh' in col: v_type = 'relh'
        depth_match = re.search(r'd(\d+)p(\d+)', col)
        if depth_match:
            depth = float(f"{depth_match.group(1)}.{depth_match.group(2)}")
        elif '_z' in col: 
            z_match = re.search(r'_z(-?\d+\.?\d*)', col)
            if z_match: depth = abs(float(z_match.group(1)))
        semantics.append((v_type, depth))
    return semantics

class MBALTrainer:
    def __init__(self, device: str = 'cuda', epochs: int = 15, lr: float = 0.0005, batch_size: int = 1024, model_type: str = 'ocean'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.model_type = model_type
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.model = None

    def fit_predict(self, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, horizon: int) -> Tuple[np.ndarray, Optional[pd.DataFrame]]:
        x_tr_arr = self.imputer.fit_transform(x_train.values)
        x_te_arr = self.imputer.transform(x_test.values)
        y_tr_arr = y_train.values.reshape(-1, 1)
        
        x_tr_s = self.scaler_x.fit_transform(x_tr_arr).astype(np.float32)
        x_te_s = self.scaler_x.transform(x_te_arr).astype(np.float32)
        y_tr_s = self.scaler_y.fit_transform(y_tr_arr).astype(np.float32)
        
        semantics = parse_mbal_feature_semantics(x_train.columns.tolist())
        dataset = TensorDataset(torch.from_numpy(x_tr_s), torch.from_numpy(y_tr_s))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        if self.model is None:
            self.model = MBAL_Ocean_Transformer(
                n_features=x_tr_s.shape[1], semantic_info=semantics,
                d_model=128, n_heads=8, e_layers=3
            ).to(self.device)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.HuberLoss() 

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                output = self.model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(loader)
            if (epoch+1) % 5 == 0: print(f"  [AI] Epoch {epoch+1}/{self.epochs} Loss: {avg_loss:.6f}")
            
        self.model.eval()
        with torch.no_grad():
            x_te_t = torch.from_numpy(x_te_s).to(self.device)
            preds_list = []
            for i in range(0, x_te_t.size(0), self.batch_size):
                preds_list.append(self.model(x_te_t[i:i+self.batch_size]))
            preds_scaled = torch.cat(preds_list, dim=0).cpu().numpy()
            preds = self.scaler_y.inverse_transform(preds_scaled)
        
        return preds.flatten(), None


def patchtst_forecast(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, horizon: int) -> Tuple[np.ndarray, Optional[pd.DataFrame]]:
    trainer = MBALTrainer(epochs=10, batch_size=2048, model_type='ocean') # Original transformer
    return trainer.fit_predict(x_train, y_train, x_test, horizon)

# Global cache for foundation models to prevent redundant disk I/O in the CV loop
_MODEL_CACHE = {}

def chronos_forecast(y_history: pd.Series, test_index: pd.DatetimeIndex, horizon: int) -> np.ndarray:
    """
    Zero-shot forecast using Amazon Chronos-Bolt.
    Takes the target history (y_history), and for each point in test_index, 
    predicts 'horizon' steps ahead and returns the point forecast.
    """
    from chronos import BaseChronosPipeline
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # The RTX 5090 is optimized for bfloat16.
    model_id = "amazon/chronos-bolt-base"
    
    if model_id not in _MODEL_CACHE:
        # Load once and keep in VRAM/RAM
        _MODEL_CACHE[model_id] = BaseChronosPipeline.from_pretrained(
            model_id, 
            device_map=device, 
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
        )
    
    pipe = _MODEL_CACHE[model_id]

    # Pre-process history: Chronos (Transformer) cannot handle NaNs.
    # We use forward-fill then global mean to ensure a dense input.
    y_history = y_history.sort_index()
    y_clean = y_history.ffill().fillna(y_history.mean()).fillna(0.0)
    
    context_len = 512
    preds = []
    
    # Batching to maximize GPU throughput
    batch_size = 64
    test_times = test_index.tolist()
    
    for i in range(0, len(test_times), batch_size):
        batch_times = test_times[i : i + batch_size]
        batch_contexts = []
        for t in batch_times:
            # Context is all history up to and including time t
            ctx = y_clean.loc[:t].tail(context_len).to_numpy(dtype=np.float32)
            if len(ctx) < 16:
                ctx = np.pad(ctx, (16 - len(ctx), 0), mode='edge') if len(ctx) > 0 else np.zeros(16, dtype=np.float32)
            batch_contexts.append(torch.from_numpy(ctx))
        
        # Predict
        with torch.no_grad():
            q, _ = pipe.predict_quantiles(
                batch_contexts, 
                prediction_length=horizon, 
                quantile_levels=[0.5]
            )
            # The value at time t+horizon is the last element of the forecast sequence
            batch_preds = q[:, horizon - 1, 0].cpu().numpy()
            preds.extend(batch_preds)
            
    return np.array(preds)
