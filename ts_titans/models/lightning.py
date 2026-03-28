import lightning as L
import torch
import torch.nn.functional as F
from .model import TSTitans

class TSTitansLightningModule(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config.__dict__)
        self.model = TSTitans(config)
        self.config = config
        
    def forward(self, seq, past_state=None, return_state=False):
        return self.model(seq, past_state=past_state, return_state=return_state)
        
    def _compute_loss(self, batch):
        x, y = batch
        # x shape: [B, context_length, input_dim]
        # y shape: [B, forecast_horizon, num_features]
        forecast = self.model(x)
        
        # We compute loss only against the true numerical values (num_features), 
        # not the missingness indicators.
        mse_loss = F.mse_loss(forecast, y)
        mae_loss = F.l1_loss(forecast, y)
        
        return mse_loss, mae_loss

    def training_step(self, batch, batch_idx):
        mse_loss, mae_loss = self._compute_loss(batch)
        self.log("train_mse", mse_loss, prog_bar=True)
        return mse_loss
        
    def validation_step(self, batch, batch_idx):
        mse_loss, mae_loss = self._compute_loss(batch)
        self.log("val_mse", mse_loss, prog_bar=True)
        self.log("val_mae", mae_loss, prog_bar=True)
        return mse_loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        return optimizer
