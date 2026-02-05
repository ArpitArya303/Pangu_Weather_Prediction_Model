import argparse
import os
import numpy as np
import xarray as xr
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
from accelerate.utils import set_seed

from Pangu.Pangu_Weather_Prediction_Model.pangu.pangu_model import Pangu_lite
from Pangu.Pangu_Weather_Prediction_Model.pangu.data_utils import (
    ZarrWeatherDataset, 
    surface_transform, 
    upper_air_transform, 
)


def train_step(model, dataloader, surface_criterion, upper_air_criterion, optimizer, accelerator, 
               train_dataset, accumulation_steps):
    """Run one epoch of training and collect per-parameter gradient statistics.

    Returns:
        epoch_loss (float), grad_stats (dict[str, float])
    """
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()

    # Accumulate mean-absolute gradients per parameter across optimizer steps
    grad_acc = {name: 0.0 for name, _ in model.named_parameters()}
    grad_count = {name: 0 for name, _ in model.named_parameters()}

    for i, batch in enumerate(dataloader):
        # Data is already on the correct device thanks to accelerator
        input_surface = batch['surface']
        input_upper_air = batch['upper_air']
        static = batch['static']
        target_surface = batch['surface_target']
        target_upper_air = batch['upper_air_target']

        batch_size = input_surface.size(0)
        
        # Forward pass with automatic mixed precision
        with accelerator.autocast():
            pred_surf, pred_upper = model(input_surface, static, input_upper_air)
            
            # Compute losses
            loss_surf = surface_criterion(pred_surf, target_surface)
            loss_upper = upper_air_criterion(pred_upper, target_upper_air)
            loss = (loss_surf * 0.25 + loss_upper) / accumulation_steps

        # Backward pass with accelerator
        accelerator.backward(loss)
        
        if (i + 1) % accumulation_steps == 0:
            # Clip gradients
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Before zeroing, record mean-absolute gradient for each parameter
            for name, param in model.named_parameters():
                if param.grad is not None:
                    try:
                        mag = param.grad.abs().mean().item()
                    except Exception:
                        # Fallback: convert to CPU then compute
                        mag = param.grad.detach().cpu().abs().mean().item()
                    grad_acc[name] += mag
                    grad_count[name] += 1

            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * batch_size

        # Clear cache periodically
        if i % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Gather loss from all processes
    total_loss = accelerator.gather(torch.tensor(running_loss, device=accelerator.device)).sum().item()
    total_samples = len(train_dataset)
    epoch_loss = total_loss / total_samples

    # Compute average gradient magnitude per parameter (skip params never updated)
    grad_stats = {}
    for name in grad_acc:
        if grad_count[name] > 0:
            grad_stats[name] = grad_acc[name] / float(grad_count[name])
    return epoch_loss, grad_stats

@torch.no_grad()
def val_step(model, dataloader, surface_criterion, upper_air_criterion, accelerator, val_dataset):
    model.eval()
    metrics = {
        'val_loss': 0.0,
        'surface_mse': 0.0,
        'upper_air_mse': 0.0
    }
    
    for batch in dataloader:
        # Data is already on the correct device
        input_surface = batch['surface']
        input_upper_air = batch['upper_air']
        static = batch['static']
        target_surface = batch['surface_target']
        target_upper_air = batch['upper_air_target']

        batch_size = input_surface.size(0)

        # Use automatic mixed precision for inference
        with accelerator.autocast():
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

    # Gather metrics from all processes
    for key in metrics:
        metrics[key] = accelerator.gather(torch.tensor(metrics[key], device=accelerator.device)).sum().item()
    
    # Normalize metrics by total number of samples
    total_samples = len(val_dataset)
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

def train(model, train_loader, val_loader, train_data, val_data, surface_criterion, upper_air_criterion, 
          optimizer, accelerator, num_epochs, log_dir, accumulation_steps, stop_patience):
    train_log_dir = os.path.join(log_dir, 'train_logs')
    val_log_dir = os.path.join(log_dir, 'val_logs')
    
    # Only create writers on main process
    if accelerator.is_main_process:
        writer_train = SummaryWriter(log_dir=train_log_dir)
        writer_val = SummaryWriter(log_dir=val_log_dir)
    
    early_stopping = EarlyStopping(patience=stop_patience, min_delta=1e-4)
    best_val_loss = float('inf')
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5e-4,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader) // accumulation_steps,
        anneal_strategy='cos'
    )
    
    for epoch in range(1, num_epochs + 1):
        # Training phase
        if accelerator.is_main_process:
            train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} - Training")
        else:
            train_bar = train_loader
            
        train_loss, grad_stats = train_step(
            model, train_bar, surface_criterion, upper_air_criterion,
            optimizer, accelerator, train_data, accumulation_steps
        )
        
        # Validation phase
        val_loss, surface_mse, upper_air_mse = val_step(
            model, val_loader, surface_criterion, upper_air_criterion, accelerator, val_data
        )

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Log metrics only on main process
        if accelerator.is_main_process:
            writer_train.add_scalar('Loss', train_loss, epoch)
            writer_val.add_scalar('Loss', val_loss, epoch)
            writer_val.add_scalar('MSE/surface', surface_mse, epoch)
            writer_val.add_scalar('MSE/upper_air', upper_air_mse, epoch)
            writer_train.add_scalar('Learning_Rate', current_lr, epoch)

            # Log gradient scalars (mean-abs) for each parameter collected during epoch
            for name, mag in grad_stats.items():
                writer_train.add_scalar(f'Gradients/{name}', mag, epoch)

            # Log model weights histograms less frequently to save memory
            if epoch % 5 == 0:
                for name, param in model.named_parameters():
                    writer_train.add_histogram(f'Weight/{name}', param.detach().cpu().numpy(), epoch)

        # Model checkpointing (only on main process)
        if accelerator.is_main_process:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                
                # Save unwrapped model for DDP
                unwrapped_model = accelerator.unwrap_model(model)
                model_state = {
                    'epoch': epoch,
                    'model_state_dict': unwrapped_model.state_dict(),
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

        # # Early stopping check
        # if early_stopping(val_loss):
        #     print("Early stopping triggered")
        #     break

        # Wait for all processes
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        writer_train.close()
        writer_val.close()
    
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
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    opt = parser.parse_args()

    # Initialize Accelerator with mixed precision
    accelerator = Accelerator(
        gradient_accumulation_steps=opt.accumulation_steps,
        mixed_precision='fp16',  # or 'bf16' if your GPU supports it
        log_with="tensorboard",
        project_dir=opt.log_dir
    )

    # Set seed for reproducibility
    set_seed(opt.seed)

    if accelerator.is_main_process:
        print("Preparing dataset...")

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
        year_range=(1959, 2017),
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
        year_range=(2018,2020),
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

    if accelerator.is_main_process:
        print("Setting up device and model...")
    
    # Initialize model
    pangu = Pangu_lite()

    # Loss functions with label smoothing
    surface_criterion = nn.L1Loss()
    upper_air_criterion = nn.L1Loss()

    # Optimizer
    optimizer = torch.optim.AdamW(
        pangu.parameters(),
        lr=5e-4,
        weight_decay=3e-6,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Prepare everything with accelerator
    pangu, optimizer, train_loader, val_loader = accelerator.prepare(
        pangu, optimizer, train_loader, val_loader
    )
    
    if accelerator.is_main_process:
        print(f"Training on {accelerator.num_processes} GPUs")
        print("Starting training...")
    
    results = train(
        pangu, 
        train_loader, 
        val_loader,
        train_dataset,
        val_dataset, 
        surface_criterion, 
        upper_air_criterion, 
        optimizer, 
        accelerator,
        opt.num_epochs, 
        opt.log_dir,
        opt.accumulation_steps,
        opt.patience
    )
    
    if accelerator.is_main_process:
        print("Training complete.")

        # Save final model
        unwrapped_model = accelerator.unwrap_model(pangu)
        model_state = {
            'epoch': opt.num_epochs,
            'model_state_dict': unwrapped_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(model_state, os.path.join(opt.log_dir, 'final_model.pth'))
        print("Final model saved.")

