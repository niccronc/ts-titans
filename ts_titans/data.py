import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import lightning as L

class TimeSeriesDataset(Dataset):
    def __init__(self, csv_path, config, fit_scaler=True, scaler=None):
        self.config = config
        
        # Create dummy data if it doesn't exist
        if not os.path.exists(csv_path):
            self._create_dummy_data(csv_path, config.num_features)
            
        # Load data (assuming shape [Time, Features])
        df = pd.read_csv(csv_path)
        
        # We assume the first column is 'date' or 'time' and drop it
        if 'date' in df.columns:
            df = df.drop('date', axis=1)
        if 'time' in df.columns:
            df = df.drop('time', axis=1)
            
        # Ensure only numeric data remains
        raw_data = df.select_dtypes(include=[np.number]).values.astype(np.float32)
        
        # Induce some artificial NaNs for testing the missingness logic
        # In real datasets, these would naturally exist
        mask = np.random.rand(*raw_data.shape) < 0.05
        raw_data[mask] = np.nan
        
        # 1. Indicator Mask Generation (x_nan)
        # 1 if null, 0 if present
        is_missing = np.isnan(raw_data).astype(np.float32)
        
        # 2. Forward Fill Imputation
        df_imputed = pd.DataFrame(raw_data).ffill().bfill().values
        
        # 3. Scaling
        self.scaler = StandardScaler() if scaler is None else scaler
        if fit_scaler:
            df_imputed = self.scaler.fit_transform(df_imputed)
        else:
            df_imputed = self.scaler.transform(df_imputed)
            
        # 4. Concatenate Imputed Data with Missingness Mask
        # [Time, F] and [Time, F] -> [Time, 2F]
        self.data_x = np.concatenate([df_imputed, is_missing], axis=1)
        
        # Target data is just the raw (but scaled and imputed) features
        # We don't predict the missingness indicators.
        self.data_y = df_imputed
        
        self.L = config.context_length
        self.H = config.forecast_horizon
        
        # Calculate how many sliding windows we can extract
        self.num_samples = len(self.data_x) - self.L - self.H + 1

    def _create_dummy_data(self, path, num_features):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Generate 10000 timesteps of sine waves with some noise
        t = np.linspace(0, 100, 10000)
        data = {f"feat_{i}": np.sin(t + i) + np.random.randn(10000) * 0.1 for i in range(num_features)}
        data["date"] = pd.date_range("2026-01-01", periods=10000, freq="h")
        pd.DataFrame(data).to_csv(path, index=False)

    def __len__(self):
        return max(0, self.num_samples)
        
    def __getitem__(self, idx):
        # Input context
        x = self.data_x[idx : idx + self.L]
        
        # Forecast horizon target
        y = self.data_y[idx + self.L : idx + self.L + self.H]
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class TSDataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.csv_path = config.data_path
        
    def setup(self, stage=None):
        # We use a single dataset here for demonstration.
        # In reality, you'd split df into train (70%), val (10%), test (20%).
        self.train_dataset = TimeSeriesDataset(self.csv_path, self.config, fit_scaler=True)
        self.val_dataset = TimeSeriesDataset(self.csv_path, self.config, fit_scaler=False, scaler=self.train_dataset.scaler)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True)
        
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config.batch_size, shuffle=False)
