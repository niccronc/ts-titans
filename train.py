import lightning as L
from ts_titans.config import TSTitansConfig
from ts_titans.data import TSDataModule
from ts_titans.models.lightning import TSTitansLightningModule
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import argparse

def main():
    parser = argparse.ArgumentParser()
    # Training args
    parser.add_argument("--max_epochs", type=int, default=5, help="Maximum number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    
    # TS Task args
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--forecast_horizon", type=int, default=96)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    
    # Model args
    parser.add_argument("--integration", type=str, default="mac", choices=["mac", "mag", "mal"])
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=2)
    
    args = parser.parse_args()
    
    config = TSTitansConfig(
        d_model=args.d_model, 
        n_layers=args.n_layers, 
        integration_type=args.integration,
        learning_rate=args.learning_rate,
        context_length=args.context_length,
        forecast_horizon=args.forecast_horizon,
        patch_size=args.patch_size,
        stride=args.stride,
        batch_size=args.batch_size
    )
    
    dm = TSDataModule(config)
    model = TSTitansLightningModule(config)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="ts-titans-best",
        monitor="val_mse",
        mode="min",
        save_top_k=1,
        enable_version_counter=False
    )
    
    logger = CSVLogger("logs", name="ts_titans_training")
    
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback],
        logger=logger,
        accelerator="auto",
        devices=1,
        enable_checkpointing=True
    )
    
    print(f"Starting Time-Series Training | L: {args.context_length}, H: {args.forecast_horizon}")
    trainer.fit(model, datamodule=dm)
    print("Training complete.")

if __name__ == "__main__":
    main()
