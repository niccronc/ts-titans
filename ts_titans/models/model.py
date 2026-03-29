import torch
import torch.nn as nn
from .layer import TitanLayer
import math

class PatchEmbedding(nn.Module):
    """
    Transforms continuous time-series data into discrete patches.
    """
    def __init__(self, config):
        super().__init__()
        self.patch_size = config.patch_size
        self.stride = config.stride
        self.input_dim = config.input_dim
        self.d_model = config.d_model
        
        # Linear projection for the flattened patch
        self.projection = nn.Linear(self.patch_size * self.input_dim, self.d_model)
        
    def forward(self, x):
        # x shape: [Batch, Time, Features]
        B, T, F = x.shape
        
        # We handle padding inside the dataloader to ensure T is a valid multiple 
        # of the stride/patch_size, but we can assert here for safety
        
        # Unfold the sequence into patches
        # Shape becomes: [Batch, Num_Patches, Patch_Size, Features]
        patches = x.unfold(dimension=1, size=self.patch_size, step=self.stride)
        
        # Rearrange to [Batch, Num_Patches, Patch_Size * Features]
        patches = patches.contiguous().view(B, -1, self.patch_size * F)
        
        # Project to d_model
        return self.projection(patches)

class TSTitans(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        
        # Time-Series Patching
        self.patch_embedding = PatchEmbedding(config)
        
        # Persistent Memory
        # Learnable prefix tokens to store general task dynamics
        self.persistent_memory = nn.Parameter(
            torch.randn(1, config.num_persistent_tokens, self.d_model)
        )
        
        self.layers = nn.ModuleList([
            TitanLayer(config) for _ in range(config.n_layers)
        ])
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Direct Multi-Step Forecasting Head
        # Calculates the number of patches resulting from the context length
        num_patches = (config.context_length - config.patch_size) // config.stride + 1
        
        # Flatten the sequence of patches and project directly to the forecast horizon
        self.flatten_dim = num_patches * self.d_model
        self.forecast_head = nn.Linear(self.flatten_dim, config.forecast_horizon * config.num_features)
        
    def forward(self, seq, past_state=None, return_state=False):
        """
        seq: [Batch, Context_Length, Input_Dim]
        """
        # 1. Patch Tokenization
        x = self.patch_embedding(seq) # -> [B, Num_Patches, d_model]
        B, Num_Patches, _ = x.shape
        
        # 2. Prepend Persistent Memory
        p_mem = self.persistent_memory.expand(B, -1, -1)
        x = torch.cat([p_mem, x], dim=1)
        
        x = self.dropout(x)
        
        # 3. Core Processing
        next_states = []
        for i, layer in enumerate(self.layers):
            layer_state = past_state[i] if past_state is not None else None
            x, next_layer_state = layer(x, past_layer_state=layer_state)
            next_states.append(next_layer_state)
            
        # 4. Strip Persistent Memory before forecasting
        x = x[:, self.config.num_persistent_tokens:, :]
        
        # 5. Direct Multi-Step Forecast
        # Flatten the patch sequence: [B, Num_Patches, d_model] -> [B, Num_Patches * d_model]
        x_flat = x.reshape(B, -1)
        
        # Project to forecast horizon: [B, H * F]
        forecast = self.forecast_head(x_flat)
        
        # Reshape to time-series output: [B, H, F]
        forecast = forecast.view(B, self.config.forecast_horizon, self.config.num_features)
        
        if return_state:
            return forecast, next_states
        return forecast
