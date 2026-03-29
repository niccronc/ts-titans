import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

# Try to import FlexAttention (PyTorch 2.5+)
FLEX_ATTENTION_AVAILABLE = False
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    flex_attention = None

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class CausalConv1d(nn.Module):
    def __init__(self, dim, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        # Depthwise convolution: groups=dim for local feature extraction
        self.conv = nn.Conv1d(dim, dim, kernel_size, groups=dim)
        
    def forward(self, x):
        # x shape: [B, T, C]
        x = x.transpose(1, 2) # [B, C, T]
        # Pad on the left to ensure strict causality
        x = F.pad(x, (self.kernel_size - 1, 0))
        x = self.conv(x)
        return x.transpose(1, 2) # [B, T, C]

class SlidingWindowAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = self.d_model // self.n_heads
        self.window_size = config.window_size
        self.use_rope = config.use_rope
        self.use_flex_attn = config.use_flex_attn and FLEX_ATTENTION_AVAILABLE
        self.num_persistent_tokens = config.num_persistent_tokens
        
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        
        # Causal 1D convolutions for local feature extraction
        self.q_conv = CausalConv1d(self.d_model, config.conv_kernel_size)
        self.k_conv = CausalConv1d(self.d_model, config.conv_kernel_size)
        self.v_conv = CausalConv1d(self.d_model, config.conv_kernel_size)
        
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        
        if self.use_rope:
            inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
            self.register_buffer("inv_freq", inv_freq)
            self._cos_cached = None
            self._sin_cached = None

    def _create_flex_mask(self, seq_len, device):
        def titan_mask(b, h, q_idx, kv_idx):
            # Persistent tokens at the beginning are fully visible
            is_persist = kv_idx < self.num_persistent_tokens
            
            # Causal + Sliding Window
            is_causal = q_idx >= kv_idx
            is_in_window = (q_idx - kv_idx) < self.window_size
            
            return is_persist | (is_causal & is_in_window)

        return create_block_mask(titan_mask, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device)

    def _get_cos_sin(self, seq_len, device, dtype):
        if self._cos_cached is not None and self._cos_cached.shape[0] >= seq_len:
            return self._cos_cached[:seq_len], self._sin_cached[:seq_len]
        
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self._cos_cached = emb.cos().to(dtype)
        self._sin_cached = emb.sin().to(dtype)
        return self._cos_cached, self._sin_cached

    def forward(self, x, mask=None, past_kv=None):
        B, T, C = x.shape
        
        q = self.q_conv(self.q_proj(x))
        k = self.k_conv(self.k_proj(x))
        v = self.v_conv(self.v_proj(x))
        
        # Reshape for multi-head attention
        q = rearrange(q, 'b t (h d) -> b h t d', h=self.n_heads)
        k = rearrange(k, 'b t (h d) -> b h t d', h=self.n_heads)
        v = rearrange(v, 'b t (h d) -> b h t d', h=self.n_heads)
        
        # Handle KV caching for inference
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            
            # For sliding window during inference, truncate the cache if it exceeds the window
            if k.shape[2] > self.window_size + self.num_persistent_tokens:
                # Keep persistent tokens, then the most recent window_size tokens
                k_persist = k[:, :, :self.num_persistent_tokens, :]
                k_window = k[:, :, -(self.window_size-1):, :] # -1 because we are adding the new token
                k = torch.cat([k_persist, k_window], dim=2)
                
                v_persist = v[:, :, :self.num_persistent_tokens, :]
                v_window = v[:, :, -(self.window_size-1):, :]
                v = torch.cat([v_persist, v_window], dim=2)
                
        next_kv = (k, v)
        
        # FlexAttention path (Gated by hardware support, config, and CUDA requirement)
        if self.use_flex_attn and x.is_cuda and past_kv is None:
            q_flex = rearrange(q, 'b h t d -> b t h d')
            k_flex = rearrange(k, 'b h t d -> b t h d')
            v_flex = rearrange(v, 'b h t d -> b t h d')
            
            if self.use_rope:
                cos, sin = self._get_cos_sin(T, x.device, x.dtype)
                cos, sin = cos[None, :, None, :], sin[None, :, None, :]
                q_flex, k_flex = apply_rotary_pos_emb(q_flex, k_flex, cos, sin)
            
            block_mask = self._create_flex_mask(T, x.device)
            out = flex_attention(q_flex, k_flex, v_flex, block_mask=block_mask)
            out = rearrange(out, 'b t h d -> b t (h d)')
            return self.out_proj(out), next_kv

        # Fallback path (Manual Sliding Window & Inference)
        if self.use_rope:
            seq_len_with_past = k.shape[2]
            cos, sin = self._get_cos_sin(seq_len_with_past, x.device, x.dtype)
            
            # Extract only the cos/sin corresponding to the new query tokens
            q_cos = cos[None, None, -T:, :]
            q_sin = sin[None, None, -T:, :]
            
            k_cos = cos[None, None, :, :]
            k_sin = sin[None, None, :, :]
            
            q, _ = apply_rotary_pos_emb(q, q, q_cos, q_sin) # We only need rotated q
            k, _ = apply_rotary_pos_emb(k, k, k_cos, k_sin) # Rotate full k

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = (q @ k.transpose(-2, -1)) * scale
        
        # Apply sliding window causal mask
        K_len = k.shape[2]
        
        if past_kv is None:
            causal_mask = torch.tril(torch.ones(T, K_len, device=x.device))
            window_mask = torch.triu(torch.ones(T, K_len, device=x.device), diagonal=-self.window_size + 1)
            combined_mask = causal_mask * window_mask
            
            # Ensure persistent tokens are visible
            combined_mask[:, :self.num_persistent_tokens] = 1.0
            combined_mask = combined_mask * causal_mask
        else:
            # Inference mode. Query can see all of K.
            combined_mask = torch.ones(T, K_len, device=x.device)
            
        if mask is not None:
            combined_mask = combined_mask * mask.view(B, 1, 1, K_len)
            
        attn_weights = attn_weights.masked_fill(combined_mask == 0, float('-inf'))
        
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        out = attn_probs @ v
        out = rearrange(out, 'b h t d -> b t (h d)')
        
        return self.out_proj(out), next_kv