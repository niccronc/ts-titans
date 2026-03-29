from dataclasses import dataclass

@dataclass
class TSTitansConfig:
    # Data Dimensions
    num_features: int = 7        # E.g., ETTh1 has 7 features
    input_dim: int = 14          # num_features * 2 (indicator mask)
    
    # Time-Series Tokenization (Patching)
    patch_size: int = 16         # Number of timesteps per patch
    stride: int = 8              # Overlap stride between patches
    
    # Forecasting Task
    context_length: int = 512    # Lookback window length (L) in timesteps
    forecast_horizon: int = 96   # Steps to predict (H) in timesteps
    
    # Model Dimensions
    d_model: int = 128           # Hidden dimension
    n_heads: int = 4             # Number of attention heads
    n_layers: int = 3            # Number of Titan layers
    d_ff: int = 512              # Feed-forward network dimension
    
    # Short-Term Memory (Core)
    window_size: int = 32        # Sliding window size (in patches)
    use_rope: bool = True
    use_flex_attn: bool = True
    conv_kernel_size: int = 4    # 1D Conv kernel for local feature extraction
    
    # Persistent Memory
    num_persistent_tokens: int = 16
    
    # Neural Long-Term Memory (NLM)
    surprise_alpha: float = 0.1
    memory_momentum: float = 0.9
    memory_decay: float = 0.01
    chunk_size: int = 16         # Associative scan segment size (in patches)
    
    # Integration Variant ("mac", "mag", "mal")
    integration_type: str = "mac"
    
    # Optimization
    learning_rate: float = 1e-4
    dropout: float = 0.1
    weight_decay: float = 1e-5
    
    # Data Pipeline
    data_path: str = "data/raw/dataset.csv"
    batch_size: int = 32
