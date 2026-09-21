import torch
import pytest
from ts_titans.models.model import PatchEmbedding, TSTitans

def test_patch_embedding_shape(base_config, dummy_batch):
    embedder = PatchEmbedding(base_config)
    out = embedder(dummy_batch)
    
    # Expected num patches: (L - patch_size) // stride + 1
    expected_patches = (base_config.context_length - base_config.patch_size) // base_config.stride + 1
    
    assert out.shape == (base_config.batch_size, expected_patches, base_config.d_model)

def test_tstitans_forward_shape(base_config, dummy_batch):
    model = TSTitans(base_config)
    forecast = model(dummy_batch)
    
    # Expected output: [Batch, Horizon, Features]
    assert forecast.shape == (base_config.batch_size, base_config.forecast_horizon, base_config.num_features)
