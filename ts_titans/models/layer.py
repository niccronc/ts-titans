import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import SlidingWindowAttention
from .memory import DeepNeuralMemory

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.silu(gate)

class TitanLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.integration_type = config.integration_type
        
        # Short-Term Memory (Core)
        self.attention = SlidingWindowAttention(config)
        self.norm1 = RMSNorm(self.d_model)
        
        # Long-Term Memory (NLM)
        self.memory = DeepNeuralMemory(config)
        self.norm2 = RMSNorm(self.d_model)
        
        # Feed-Forward Network with SwiGLU
        hidden_dim = int(config.d_ff * 2 / 3) 
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, hidden_dim * 2),
            SwiGLU(),
            nn.Linear(hidden_dim, self.d_model),
            nn.Dropout(config.dropout)
        )
        self.norm3 = RMSNorm(self.d_model)
        
        if self.integration_type == "mag":
            self.gate = nn.Linear(self.d_model * 2, self.d_model)
        
    def forward(self, x, past_layer_state=None):
        past_attn_kv = None
        past_mem_state = None
        
        if past_layer_state is not None:
            past_attn_kv, past_mem_state = past_layer_state
            
        m_out, next_mem_state = self.memory(x, x, x, past_state=past_mem_state)
        
        # 2. Integration & Attention
        if self.integration_type == "mac":
            x = x + m_out
            x = self.norm1(x)
            attn_out, next_attn_kv = self.attention(x, past_kv=past_attn_kv)
            x = x + attn_out
            
        elif self.integration_type == "mag":
            attn_out, next_attn_kv = self.attention(self.norm1(x), past_kv=past_attn_kv)
            combined = torch.cat([attn_out, m_out], dim=-1)
            g = torch.sigmoid(self.gate(combined))
            x = x + g * attn_out + (1 - g) * m_out
            
        elif self.integration_type == "mal":
            x = x + m_out
            x = self.norm2(x)
            attn_out, next_attn_kv = self.attention(x, past_kv=past_attn_kv)
            x = x + attn_out
            
        x = self.norm3(x)
        x = x + self.ffn(x)
        
        next_layer_state = (next_attn_kv, next_mem_state)
        return x, next_layer_state
