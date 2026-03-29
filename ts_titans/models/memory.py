import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1d(nn.Module):
    def __init__(self, dim, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(dim, dim, kernel_size, groups=dim)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.pad(x, (self.kernel_size - 1, 0))
        x = self.conv(x)
        return x.transpose(1, 2)

class DeepNeuralMemory(nn.Module):
    """
    A faithful implementation of the Deep MLP Learner for Time Series.
    Uses a segmented associative scan to allow 'test-time updates' 
    with proper memory decay (forgetting) and surprise scaling.
    """
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = self.d_model // self.n_heads
        self.chunk_size = config.chunk_size
        
        self.alpha = config.surprise_alpha
        self.decay = config.memory_decay
        
        # 1D Convolutions for local feature extraction before memory update
        self.q_conv = CausalConv1d(self.d_model, config.conv_kernel_size)
        self.k_conv = CausalConv1d(self.d_model, config.conv_kernel_size)
        self.v_conv = CausalConv1d(self.d_model, config.conv_kernel_size)
        
    def forward(self, q, k, v, past_state=None):
        B, T, _ = q.shape
        
        # Apply causal 1D convolutions
        q = self.q_conv(q)
        k = self.k_conv(k)
        v = self.v_conv(v)
        
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Initialize memory state
        if past_state is None:
            state = torch.zeros(B, self.n_heads, self.head_dim, self.head_dim, device=q.device, dtype=q.dtype)
        else:
            state = past_state
            
        out_chunks = []
        
        # Ensure we don't try to chunk with a size larger than the sequence
        chunk_size = min(self.chunk_size, T)
        
        for i in range(0, T, chunk_size):
            q_chunk = q[:, :, i:i+chunk_size, :]
            k_chunk = k[:, :, i:i+chunk_size, :]
            v_chunk = v[:, :, i:i+chunk_size, :]
            
            chunk_len = q_chunk.shape[2]
            
            # Compute associations for this chunk: alpha * (k_t.T @ v_t)
            kv_chunk = torch.einsum('bhtd,bhte->bhtde', k_chunk, v_chunk) * self.alpha
            
            decay_factor = 1.0 - self.decay
            
            # Create decay mask
            idx = torch.arange(chunk_len, device=q.device)
            diff = idx.unsqueeze(1) - idx.unsqueeze(0)
            
            # Safe calculation to avoid NaN: 
            causal_mask = (diff >= 0).float()
            safe_diff = torch.clamp(diff, min=0).float()
            
            decay_matrix = (decay_factor ** safe_diff) * causal_mask
            
            kv_flat = kv_chunk.view(B, self.n_heads, chunk_len, -1).transpose(2, 3) 
            accumulated_kv_flat = torch.matmul(kv_flat, decay_matrix.transpose(0, 1))
            accumulated_kv = accumulated_kv_flat.transpose(2, 3).view(B, self.n_heads, chunk_len, self.head_dim, self.head_dim)
            
            # Add the decayed past state
            past_decay_factors = decay_factor ** (idx + 1) # [T]
            past_decay_factors = past_decay_factors.view(1, 1, chunk_len, 1, 1)
            
            weights = (state.unsqueeze(2) * past_decay_factors) + accumulated_kv
            
            # Retrieve
            out_chunk = torch.einsum('bhtd,bhtde->bhte', q_chunk, weights)
            out_chunks.append(out_chunk)
            
            # Update the past state for the next chunk
            state = weights[:, :, -1, :, :]
            
        out = torch.cat(out_chunks, dim=2)
        return out.transpose(1, 2).reshape(B, T, self.d_model), state