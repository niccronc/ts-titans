# TS-Titans: Generalized Time-Series Forecasting with Neural Memory

A professional PyTorch implementation of the **Titans** architecture (Google DeepMind) specialized for **Long-Horizon Time-Series Forecasting (LTSF)**.

This repository demonstrates how the **MIRAS** framework and Test-Time Training (TTT) can be applied to multivariate numerical data to achieve state-of-the-art results on long-context benchmarks.

## Key Features

- **Neural Long-Term Memory (NLM):** Captures multi-year baselines and regime shifts using a deep MLP learner that updates its weights during inference.
- **Time-Series Patching:** Groups windows of timesteps into discrete tokens (similar to PatchTST) for massive speedups on high-frequency data.
- **Missing Data Handling:** Built-in **Forward-Fill + Indicator Mask** logic to explicitly teach the model how to handle sensor dropouts and null values.
- **Direct Multi-step Forecasting:** Predicts the entire future horizon in a single shot, avoiding the error accumulation of autoregressive models.
- **Stateful Continuous Inference:** A specialized evaluation loop that "warms up" the neural memory on history and carries state forward, proving the power of long-term memory.

## Installation

This project uses `uv` for lightning-fast dependency management.

```bash
git clone https://github.com/niccronc/ts-titans.git
cd ts-titans
uv sync
```

## Usage

### 1. Prepare Your Data
The model expects a `.csv` file with features. If no file is found, it will automatically generate a synthetic sine-wave dataset at `data/raw/dataset.csv`.

### 2. Training
Train the model on the LTSF benchmarks (e.g., ETTh1 or Weather):
```bash
uv run python train.py --context_length 512 --forecast_horizon 96 --patch_size 16 --max_epochs 10
```

### 3. Stateful Evaluation (Test-Time Training)
To run the stateful inference loop that leverages the Neural Memory's ability to learn from history:
```bash
uv run python eval_stateful.py --checkpoint checkpoints/ts-titans-best.ckpt
```

## Architecture Details

- **Input:** `[Batch, Context, 2 * Features]` (includes missingness mask).
- **Encoder:** Linear patching layer + RoPE-enhanced Sliding Window Attention.
- **Memory:** Segmented Associative Scan with explicit forgetting/decay.
- **Head:** Linear projection to `[Batch, Horizon, Features]`.

## References
- *Titans: Learning to Memorize at Test Time* (Behrouz et al., Google Research, 2024).
- *PatchTST: A Time Series is Worth 64 Words* (Nie et al., 2023).
