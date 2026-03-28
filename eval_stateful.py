import torch
import pandas as pd
import numpy as np
from ts_titans.config import TSTitansConfig
from ts_titans.models.lightning import TSTitansLightningModule
from ts_titans.data import TimeSeriesDataset
import argparse
import os

def stateful_inference(model, dataset, config, device):
    """
    Simulates a real-world scenario where the model 'warms up' on history
    and then performs continuous, stateful forecasting.
    """
    model.eval()
    model.to(device)
    
    # 1. Warm-up Phase: Process all available history to build the Neural Memory state
    print("Warming up Neural Memory on dataset history...")
    state = None
    
    # We use a batch size of 1 for sequential state accumulation
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    all_mses = []
    
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            
            # Forward pass: Get forecast AND the updated memory state
            # Note: During warm-up, we are essentially 'teaching' the model the baseline
            forecast, state = model(x, past_state=state, return_state=True)
            
            # Detach state to prevent graph buildup (standard stateful inference)
            state = [(s[0].detach(), s[1].detach()) for s in state]
            
            if i % 100 == 0:
                mse = torch.mean((forecast - y)**2).item()
                all_mses.append(mse)
                print(f"Step {i}/{len(loader)} | Running MSE: {np.mean(all_mses):.4f}")
                
    print("\nStateful Inference Complete.")
    print(f"Final Average MSE: {np.mean(all_mses):.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best checkpoint")
    parser.add_argument("--data_path", type=str, default="data/raw/dataset.csv")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model and its saved config
    print(f"Loading checkpoint from {args.checkpoint}...")
    lightning_module = TSTitansLightningModule.load_from_checkpoint(args.checkpoint)
    model = lightning_module.model
    config = TSTitansConfig(**lightning_module.hparams)
    
    # Load dataset (using the same scaler logic)
    dataset = TimeSeriesDataset(args.data_path, config, fit_scaler=True)
    
    stateful_inference(model, dataset, config, device)

if __name__ == "__main__":
    main()
