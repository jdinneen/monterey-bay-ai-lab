#!/usr/bin/env python3
"""
LatentTSF Value Gate: Autoencoder Training and Latent Space Visualization

This script trains a 1D-CNN Autoencoder to compress raw M1 mooring data into a 
latent representation. It then projects the latent space down to 2D using UMAP 
and color-codes the points by their original physical temperature.

VALUE GATE REQUIREMENT: 
If the UMAP plot shows distinct physical clustering (e.g., a clear temperature 
gradient across the latent space), it proves the Autoencoder has learned abstract 
physics, justifying the full LatentTSF pipeline.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import umap

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOTA_DIR = PROJECT_ROOT / "sota_continual_learning"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SOTA_DIR) not in sys.path:
    sys.path.insert(0, str(SOTA_DIR))

from sota_continual_learning.data import create_high_performance_loader

class Conv1DAutoencoder(nn.Module):
    """Zero-Bloat 1D-CNN Autoencoder for time-series compression."""
    def __init__(self, seq_len=720, latent_dim=64):
        super().__init__()
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        
        # Encoder: (Batch, Channels, SeqLen) -> (Batch, Latent)
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Flatten()
        )
        
        # Calculate flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, seq_len)
            flat_size = self.encoder(dummy).shape[1]
            
        self.fc_encode = nn.Linear(flat_size, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, flat_size)
        
        self._spatial_shape = (64, seq_len // 8) # After 3 stride-2 layers
        
        # Decoder: (Batch, Latent) -> (Batch, Channels, SeqLen)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=4, stride=2, padding=1)
        )

    def encode(self, x):
        # x input shape from data loader: (Batch, SeqLen, 1)
        # Conv1d expects: (Batch, Channels, SeqLen)
        x = x.transpose(1, 2)
        h = self.encoder(x)
        z = self.fc_encode(h)
        return z

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(z.size(0), self._spatial_shape[0], self._spatial_shape[1])
        x_rec = self.decoder(h)
        # Output shape back to: (Batch, SeqLen, 1)
        return x_rec.transpose(1, 2)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

def train_and_visualize():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 1. Initialize Loader
    parquet_path = "lakehouse/gold/forecast_predictions"
    print(f"Initializing data loader from {parquet_path}...")
    loader = create_high_performance_loader(
        parquet_path=parquet_path,
        batch_size=256,
        num_workers=4,
        context_window=720,
        forecast_horizon=24
    )
    
    # 2. Initialize Autoencoder
    model = Conv1DAutoencoder(seq_len=720, latent_dim=64).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    # 3. Quick Burn-in Training
    train_steps = 1000
    print(f"Training Autoencoder for {train_steps} steps to map the latent space...")
    
    model.train()
    step = 0
    total_loss = 0
    
    for batch_x, _ in loader:
        if step >= train_steps:
            break
            
        batch_x = batch_x.to(device)
        
        optimizer.zero_grad()
        reconstruction = model(batch_x)
        loss = criterion(reconstruction, batch_x)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        step += 1
        
        if step % 100 == 0:
            print(f"Step {step}/{train_steps} | Recon Loss: {loss.item():.4f}")
            
    print(f"\nTraining Complete. Average Loss: {total_loss/train_steps:.4f}")
    
    # 4. Latent Extraction & UMAP Validation
    print("\nExtracting latent representations for validation gate...")
    model.eval()
    
    latent_vectors = []
    physical_temps = [] # We use the mean value of the window as a physical proxy
    
    extract_steps = 20
    step = 0
    
    with torch.no_grad():
        for batch_x, _ in loader:
            if step >= extract_steps:
                break
                
            batch_x = batch_x.to(device)
            z = model.encode(batch_x)
            
            latent_vectors.append(z.cpu().numpy())
            # Mean temperature of the context window
            physical_temps.append(batch_x.mean(dim=1).squeeze().cpu().numpy())
            step += 1

    Z = np.concatenate(latent_vectors, axis=0)
    T = np.concatenate(physical_temps, axis=0)
    
    print(f"Projecting {Z.shape[0]} latent vectors from {Z.shape[1]}D down to 2D via UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
    Z_2d = reducer.fit_transform(Z)
    
    # 5. Output Verification Data
    out_dir = PROJECT_ROOT / "research" / "model_lab" / "latent_tsf"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    plot_path = out_dir / "umap_latent_space.png"
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(Z_2d[:, 0], Z_2d[:, 1], c=T, cmap='coolwarm', s=5, alpha=0.8)
    plt.colorbar(scatter, label='Normalized Physical State (Mean Window Temp)')
    plt.title("Latent Space Disentanglement (UMAP projection of 64D -> 2D)")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"\nVALUE GATE VISUALIZATION SAVED TO: {plot_path}")
    print("If the plot shows a clear color gradient, the Autoencoder has successfully learned to disentangle physical states.")

if __name__ == "__main__":
    train_and_visualize()
