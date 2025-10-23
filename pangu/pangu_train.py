import argparse
import os
import numpy as np
import xarray as xr
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from Pangu.Pangu_Weather_Prediction_Model.pangu.pangu_model import Pangu_lite
from Pangu.Pangu_Weather_Prediction_Model.pangu.data_utils import (
    ZarrWeatherDataset, 
    surface_transform, 
    upper_air_transform, 
)


def train_step(model, dataloader, surface_criterion, upper_air_criterion, optimizer, device, 
               scaler, train_dataset, accumulation_steps=1):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    
    for i, batch in enumerate(dataloader):
        # Unpack and move to device in a memory-efficient way
        input_surface = batch['surface'].to(device, non_blocking=True)
        input_upper_air = batch['upper_air'].to(device, non_blocking=True)
        static = batch['static'].to(device, non_blocking=True)
        target_surface = batch['surface_target'].to(device, non_blocking=True)
        target_upper_air = batch['upper_air_target'].to(device, non_blocking=True)

        batch_size = input_surface.size(0)
        
        # Mixed precision training
        with autocast():
            pred_surf, pred_upper = model(input_surface, static, input_upper_air)
            
            # Compute losses
            loss_surf = surface_criterion(pred_surf, target_surface)
            loss_upper = upper_air_criterion(pred_upper, target_upper_air)
            loss = (loss_surf * 0.25 + loss_upper) / accumulation_steps

        # Scaled backward pass
        scaler.scale(loss).backward()
        
        if (i + 1) % accumulation_steps == 0:
            # Unscale gradients and clip
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step with scaling
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * batch_size

        # Clear cache periodically
        if i % 10 == 0:
            torch.cuda.empty_cache()

    # Calculate average loss for the epoch
    total_samples = len(train_dataset)  # Get total number of samples directly from dataset
    epoch_loss = running_loss / total_samples
    return epoch_loss

@torch.no_grad()
def val_step(model, dataloader, surface_criterion, upper_air_criterion, device, val_dataset):
    model.eval()
    metrics = {
        'val_loss': 0.0,
        'surface_mse': 0.0,
        'upper_air_mse': 0.0
    }
    
    for batch in dataloader:
        # Efficient device transfer with non_blocking
        input_surface = batch['surface'].to(device, non_blocking=True)
        input_upper_air = batch['upper_air'].to(device, non_blocking=True)
        static = batch['static'].to(device, non_blocking=True)
        target_surface = batch['surface_target'].to(device, non_blocking=True)
        target_upper_air = batch['upper_air_target'].to(device, non_blocking=True)

        batch_size = input_surface.size(0)

        # Use mixed precision for inference
        with autocast():
            pred_surf, pred_upper = model(input_surface, static, input_upper_air)

            # Compute losses efficiently
            loss_surf = surface_criterion(pred_surf, target_surface)
            loss_upper = upper_air_criterion(pred_upper, target_upper_air)
            loss = loss_surf * 0.25 + loss_upper

        # Update metrics
        metrics['val_loss'] += loss.item() * batch_size
        
        # Compute MSE efficiently using torch operations
        metrics['surface_mse'] += torch.mean((pred_surf - target_surface) ** 2).item() * batch_size
        metrics['upper_air_mse'] += torch.mean((pred_upper - target_upper_air) ** 2).item() * batch_size

        # Clear some memory
        del pred_surf, pred_upper
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Normalize metrics by total number of samples
    total_samples = len(val_dataset)  # Get total number of samples directly from dataset
    metrics = {k: v / total_samples for k, v in metrics.items()}
    
    return metrics['val_loss'], metrics['surface_mse'], metrics['upper_air_mse']

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop

def train(model, train_loader, val_loader, surface_criterion, upper_air_criterion, 
          optimizer, device, num_epochs, log_dir, accumulation_steps=2):
    writer = SummaryWriter(log_dir=log_dir)
    scaler = GradScaler()
    early_stopping = EarlyStopping(patience=10, min_delta=1e-4)
    best_val_loss = float('inf')
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5e-4,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader) // accumulation_steps,
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    for epoch in range(1, num_epochs + 1):
        # Training phase
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} - Training")
        train_loss = train_step(
            model, train_bar, surface_criterion, upper_air_criterion,
            optimizer, device, scaler, train_dataset, accumulation_steps
        )
        
        # Validation phase
        val_loss, surface_mse, upper_air_mse = val_step(
            model, val_loader, surface_criterion, upper_air_criterion, device, val_dataset
        )

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('MSE/surface', surface_mse, epoch)
        writer.add_scalar('MSE/upper_air', upper_air_mse, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)

        # Log model gradients and weights less frequently to save memory
        if epoch % 5 == 0:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(f'grad/{name}', param.grad, epoch)
                writer.add_histogram(f'weight/{name}', param, epoch)

        # Model checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
            }
            torch.save(model_state, os.path.join(log_dir, 'best_model.pth'))
            print(f"✓ Best model saved (val_loss: {val_loss:.4f})")

        # Print metrics
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Surface MSE: {surface_mse:.4f}, Upper Air MSE: {upper_air_mse:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Early stopping check
        if early_stopping(val_loss):
            print("Early stopping triggered")
            break
            
        # Clear memory
        torch.cuda.empty_cache()
    
    writer.close()
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="/home/bedartha/public/datasets/as_downloaded/weatherbench2/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr", help="Path to the Zarr dataset")
    parser.add_argument("--surface_variables", nargs='+', default=["2m_temperature","mean_sea_level_pressure","10m_u_component_of_wind","10m_v_component_of_wind"], help="Surface variables")
    parser.add_argument("--upper_air_variables", nargs='+', default=["geopotential","specific_humidity","temperature", "u_component_of_wind", "v_component_of_wind"], help="Upper air variables")
    parser.add_argument("--pLevels", nargs='+', default=[50,100,150,200,250,300,400,500,600,700,850,925,1000], type=int, help="Pressure levels for upper air variables")
    parser.add_argument("--static_variables", nargs='+', default=["land_sea_mask", "soil_type"], help="Static variables")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--log_dir", default="runs/trial", help="Directory to save logs")
    parser.add_argument("--transform_dir", type=str, default="/storage/arpit/Pangu/Pangu_Weather_Prediction_Model/pangu/data", help="Directory containing transforms")
    parser.add_argument("--accumulation_steps", type=int, default=2, help="Number of gradient accumulation steps")
    parser.add_argument("--num_workers", type=int, default=6, help="Number of data loading workers per GPU")
    opt = parser.parse_args()

    print("Preparing indices...")
    dataset = xr.open_zarr(opt.data)
    time_len = dataset.sizes['time'] - 1
    indices = np.arange(time_len)
    
    # Stratified split to ensure temporal consistency
    train_indices = indices[:int(0.8 * time_len)]
    val_indices = indices[int(0.8 * time_len):]

    print("Creating datasets and dataloaders...")
    # Create the transforms
    surface_normalizer, _ = surface_transform(
        os.path.join(opt.transform_dir, "surface_mean.pkl"),
        os.path.join(opt.transform_dir, "surface_std.pkl")
    )

    upper_air_normalizer, _, _ = upper_air_transform(
        os.path.join(opt.transform_dir, "upper_air_mean.pkl"),
        os.path.join(opt.transform_dir, "upper_air_std.pkl")
    )

    # Create datasets with optimized chunk size
    chunk_size = opt.batch_size * opt.accumulation_steps
    train_dataset = ZarrWeatherDataset(
        zarr_path=opt.data, 
        surface_vars=opt.surface_variables,
        upper_air_vars=opt.upper_air_variables,
        plevels=opt.pLevels,
        static_vars=opt.static_variables,
        indices=train_indices,
        surface_transform=surface_normalizer,  
        upper_air_transform=upper_air_normalizer,
        chunk_size=chunk_size
    )

    val_dataset = ZarrWeatherDataset(
        zarr_path=opt.data,
        surface_vars=opt.surface_variables,
        upper_air_vars=opt.upper_air_variables,
        plevels=opt.pLevels,
        static_vars=opt.static_variables,
        indices=val_indices,
        surface_transform=surface_normalizer,
        upper_air_transform=upper_air_normalizer,
        chunk_size=chunk_size
    )

    # Create optimized dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    print("Setting up device and model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    pangu = Pangu_lite().to(device)

    # Loss functions with label smoothing
    surface_criterion = nn.L1Loss()
    upper_air_criterion = nn.L1Loss()

    # Optimizer with gradient clipping
    optimizer = torch.optim.AdamW(
        pangu.parameters(),
        lr=5e-4,
        weight_decay=3e-6,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    print("Starting training...")
    results = train(
        pangu, 
        train_loader, 
        val_loader, 
        surface_criterion, 
        upper_air_criterion, 
        optimizer, 
        device, 
        opt.num_epochs, 
        opt.log_dir,
        opt.accumulation_steps
    )
    print("Training complete.")

    # Save final model
    model_state = {
        'epoch': opt.num_epochs,
        'model_state_dict': pangu.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(model_state, os.path.join(opt.log_dir, 'final_model.pth'))
    print("Final model saved.")

