"""
Smoke test for MBAL MoE Transformer integration.
"""
import torch
import torch.nn as nn
import pytest
from mbal_deep_models import MBAL_MoE_Transformer

def test_moe_transformer_forward():
    # Setup dummy data
    n_features = 4
    batch_size = 2
    d_model = 64
    
    # Semantic info for 4 features
    semantic_info = [
        ('temp', 1.0),
        ('sal', 1.0),
        ('air', 0.0),
        ('wind', 0.0)
    ]
    
    # Instantiate model
    model = MBAL_MoE_Transformer(
        n_features=n_features,
        semantic_info=semantic_info,
        d_model=d_model,
        n_heads=4,
        e_layers=1,
        d_ff=128,
        num_experts=4
    )
    
    # Dummy input (batch_size, n_features)
    x = torch.randn(batch_size, n_features)
    
    # Forward pass
    output = model(x)
    
    # Assertions
    assert output.shape == (batch_size, 1), f"Expected shape {(batch_size, 1)}, got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaNs"

def test_moe_transformer_training_step():
    # Setup dummy data
    n_features = 4
    batch_size = 2
    d_model = 64
    semantic_info = [('temp', 1.0), ('sal', 1.0), ('air', 0.0), ('wind', 0.0)]
    
    model = MBAL_MoE_Transformer(
        n_features=n_features,
        semantic_info=semantic_info,
        d_model=d_model,
        num_experts=4
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # 1 training step
    model.train()
    x = torch.randn(batch_size, n_features)
    y = torch.randn(batch_size, 1)
    
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    
    assert loss.item() >= 0, "Loss should be non-negative"

if __name__ == "__main__":
    pytest.main([__file__])
