import pytest
import torch
from ts_titans.config import TSTitansConfig

@pytest.fixture
def base_config():
    return TSTitansConfig(
        d_model=32,
        n_layers=2,
        n_heads=4,
        context_length=64,
        forecast_horizon=24,
        patch_size=8,
        stride=4,
        input_dim=5,
        num_features=5,
        batch_size=2,
        num_persistent_tokens=4,
        window_size=16,
        chunk_size=16,
        conv_kernel_size=3
    )

@pytest.fixture
def dummy_batch(base_config):
    B = base_config.batch_size
    L = base_config.context_length
    F = base_config.input_dim
    # Random tensor simulating a batch of time-series context
    return torch.randn(B, L, F)
