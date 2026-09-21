import torch
import pytest
from ts_titans.models.memory import DeepNeuralMemory

def test_memory_forward_shape(base_config):
    memory = DeepNeuralMemory(base_config)
    B = base_config.batch_size
    T = 32 # number of patch tokens
    D = base_config.d_model
    
    q = torch.randn(B, T, D)
    k = torch.randn(B, T, D)
    v = torch.randn(B, T, D)
    
    out, next_state = memory(q, k, v)
    
    assert out.shape == (B, T, D)
    # State shape should be [B, n_heads, head_dim, head_dim]
    head_dim = D // base_config.n_heads
    assert next_state.shape == (B, base_config.n_heads, head_dim, head_dim)

def test_memory_stateful_accumulation(base_config):
    # Set conv_kernel_size to 1 to eliminate causal convolution temporal dependency at boundary
    base_config.conv_kernel_size = 1
    memory = DeepNeuralMemory(base_config)
    B = 2
    T_chunk = 16
    D = base_config.d_model
    
    q1, k1, v1 = torch.randn(B, T_chunk, D), torch.randn(B, T_chunk, D), torch.randn(B, T_chunk, D)
    q2, k2, v2 = torch.randn(B, T_chunk, D), torch.randn(B, T_chunk, D), torch.randn(B, T_chunk, D)
    
    # Process sequentially
    _, state1 = memory(q1, k1, v1)
    _, state2 = memory(q2, k2, v2, past_state=state1)
    
    # Process together (concatenated)
    q_all = torch.cat([q1, q2], dim=1)
    k_all = torch.cat([k1, k2], dim=1)
    v_all = torch.cat([v1, v2], dim=1)
    
    _, state_all = memory(q_all, k_all, v_all)
    
    # The final state should theoretically be very close, though floating point 
    # math across chunk boundaries might introduce minor differences.
    torch.testing.assert_close(state2, state_all, rtol=1e-3, atol=1e-3)
