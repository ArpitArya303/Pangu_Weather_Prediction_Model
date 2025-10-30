import argparse
import os
import yaml
import numpy as np
import wandb
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import nn
from torch.cuda.amp import autocast, GradScaler

from Pangu.Pangu_Weather_Prediction_Model.pangu.pangu_model import Pangu_lite
from Pangu.Pangu_Weather_Prediction_Model.pangu.data_utils import (
    ZarrWeatherDataset, 
    surface_transform, 
    upper_air_transform, 
)


def train_step(model, dataloader, surface_criterion, upper_air_criterion, optimizer, device, 
               scaler, train_dataset, accumulation_steps=1, max_grad_norm=1.0, surface_loss_weight=0.25):
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
            loss = (loss_surf * surface_loss_weight + loss_upper) / accumulation_steps

        # Scaled backward pass
        scaler.scale(loss).backward()
        
        if (i + 1) % accumulation_steps == 0:
            # Unscale gradients and clip
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            
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
def val_step(model, dataloader, surface_criterion, upper_air_criterion, device, val_dataset, surface_loss_weight=0.25):
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
            loss = loss_surf * surface_loss_weight + loss_upper

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
          optimizer, device, config, train_dataset, val_dataset):
    """
    Training loop with all parameters from config dict.
    
    Args:
        config: Dictionary containing all training configuration including:
            - num_epochs, accumulation_steps, max_grad_norm, surface_loss_weight
            - early_stopping_patience, early_stopping_min_delta
            - scheduler_max_lr, scheduler_pct_start, scheduler_anneal_strategy
            - log_dir, log_histogram_frequency
            - wandb config (entity, project, run_name, notes, tags)
    """
    # Extract config values
    num_epochs = config['num_epochs']
    accumulation_steps = config['accumulation_steps']
    max_grad_norm = config['max_grad_norm']
    surface_loss_weight = config['surface_loss_weight']
    log_dir = config['log_dir']
    log_histogram_freq = config['log_histogram_frequency']
    
    scaler = GradScaler()
    early_stopping = EarlyStopping(
        patience=config['early_stopping_patience'], 
        min_delta=config['early_stopping_min_delta']
    )
    best_val_loss = float('inf')

    # Initialize W&B with minimal essential config
    wandb.init(
        entity=config['wandb_entity'],
        project=config['wandb_project'],
        name=config.get('run_name'),  # Set the actual run name
        config={
            # Core hyperparameters that affect training
            'batch_size': config['batch_size'],
            'accumulation_steps': config['accumulation_steps'],
            'learning_rate': config['learning_rate'],
            'weight_decay': config['weight_decay'],
            'max_grad_norm': config['max_grad_norm'],
            'surface_loss_weight': config['surface_loss_weight'],
            'num_epochs': config['num_epochs'],
            # Model identifier
            'model': config['model_name'],
            # Data split info
            'train_years': config['train_years'],
            'val_years': config['val_years'],
        },
        notes=config.get('wandb_notes', ''),
        tags=config.get('wandb_tags', [])
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['scheduler_max_lr'],
        epochs=num_epochs,
        steps_per_epoch=len(train_loader) // accumulation_steps,
        pct_start=config['scheduler_pct_start'],
        anneal_strategy=config['scheduler_anneal_strategy']
    )
    
    for epoch in range(1, num_epochs + 1):
        # Training phase
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} - Training")
        train_loss = train_step(
            model, train_bar, surface_criterion, upper_air_criterion,
            optimizer, device, scaler, train_dataset, accumulation_steps,
            max_grad_norm, surface_loss_weight
        )
        
        # Validation phase
        val_loss, surface_mse, upper_air_mse = val_step(
            model, val_loader, surface_criterion, upper_air_criterion, 
            device, val_dataset, surface_loss_weight
        )

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics to W&B
        log_dict = {
            'epoch': epoch,
            'train/loss': train_loss,
            'val/loss': val_loss,
            'val/surface_mse': surface_mse,
            'val/upper_air_mse': upper_air_mse,
            'learning_rate': current_lr,
        }
        
        wandb.log(log_dict)

        # Log model gradients and weights at configured frequency
        if epoch % log_histogram_freq == 0:
            # Log weight histograms
            for name, param in model.named_parameters():
                wandb.log({
                    f'weights/{name}': wandb.Histogram(param.data.cpu().numpy())
                })
                if param.grad is not None:
                    wandb.log({
                        f'gradients/{name}': wandb.Histogram(param.grad.data.cpu().numpy())
                    })

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
            
            # Save best model to W&B
            wandb.save(os.path.join(log_dir, 'best_model.pth'))
            print(f"✓ Best model saved (val_loss: {val_loss:.4f})")

        # Print metrics
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Surface MSE: {surface_mse:.4f}, Upper Air MSE: {upper_air_mse:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Early stopping check
        if early_stopping(val_loss):
            print("Early stopping triggered")
            wandb.log({'early_stopping_epoch': epoch})
            break
            
        # Clear memory
        torch.cuda.empty_cache()
    
    wandb.finish()
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/storage/arpit/Pangu/Pangu_Weather_Prediction_Model/pangu/config.yml", help="Path to config YAML file")
    parser.add_argument("--run_name", type=str, default=None, help="Optional run name for W&B")
    args = parser.parse_args()

    # Load configuration from YAML
    config_path = args.config
    if not os.path.isabs(config_path):
        # If relative path, assume it's in the same directory as this script
        config_path = os.path.join(os.path.dirname(__file__), config_path)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Loading configuration from: {config_path}")
    
    # Extract configuration
    dataset_cfg = config['dataset']
    dataloader_cfg = config['dataloader']
    training_cfg = config['training']
    logging_cfg = config['logging']
    wandb_cfg = config['wandb']
    device_cfg = config['device']

    print("Creating datasets and dataloaders...")
    # Create the transforms
    surface_normalizer, _ = surface_transform(
        os.path.join(dataset_cfg['transform_dir'], "surface_mean.pkl"),
        os.path.join(dataset_cfg['transform_dir'], "surface_std.pkl")
    )

    upper_air_normalizer, _, _ = upper_air_transform(
        os.path.join(dataset_cfg['transform_dir'], "upper_air_mean.pkl"),
        os.path.join(dataset_cfg['transform_dir'], "upper_air_std.pkl")
    )

    # Create datasets with optimized chunk size
    chunk_size = dataloader_cfg['batch_size'] * training_cfg['accumulation_steps']

    train_dataset = ZarrWeatherDataset(
        zarr_path=dataset_cfg['path'], 
        surface_vars=dataset_cfg['surface_variables'],
        upper_air_vars=dataset_cfg['upper_air_variables'],
        plevels=dataset_cfg['pressure_levels'],
        static_vars=dataset_cfg['static_variables'],
        year_range=tuple(dataset_cfg['train_years']),
        surface_transform=surface_normalizer,  
        upper_air_transform=upper_air_normalizer,
        chunk_size=chunk_size
    )

    val_dataset = ZarrWeatherDataset(
        zarr_path=dataset_cfg['path'],
        surface_vars=dataset_cfg['surface_variables'],
        upper_air_vars=dataset_cfg['upper_air_variables'],
        plevels=dataset_cfg['pressure_levels'],
        static_vars=dataset_cfg['static_variables'],
        year_range=tuple(dataset_cfg['val_years']),
        surface_transform=surface_normalizer,
        upper_air_transform=upper_air_normalizer,
        chunk_size=chunk_size
    )

    # Create optimized dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=dataloader_cfg['batch_size'],
        shuffle=dataloader_cfg['shuffle'],
        num_workers=dataloader_cfg['num_workers'],
        pin_memory=dataloader_cfg['pin_memory'],
        persistent_workers=dataloader_cfg['persistent_workers']
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=dataloader_cfg['batch_size'],
        shuffle=False,
        num_workers=dataloader_cfg['num_workers'],
        pin_memory=dataloader_cfg['pin_memory'],
        persistent_workers=dataloader_cfg['persistent_workers']
    )

    print("Setting up device and model...")
    device = torch.device("cuda" if (device_cfg['cuda'] and torch.cuda.is_available()) else "cpu")
    
    # Set seed for reproducibility
    torch.manual_seed(device_cfg['seed'])
    np.random.seed(device_cfg['seed'])
    
    # Initialize model
    pangu = Pangu_lite().to(device)

    # Loss functions
    surface_criterion = nn.L1Loss()
    upper_air_criterion = nn.L1Loss()

    # Optimizer with gradient clipping
    optimizer = torch.optim.AdamW(
        pangu.parameters(),
        lr=training_cfg['learning_rate'],
        weight_decay=training_cfg['weight_decay'],
        betas=training_cfg['optimizer_betas'],
        eps=training_cfg['optimizer_eps']
    )
    
    print("Starting training...")
    
    # Create unified config dict for training function
    train_config = {
        # Training parameters
        'num_epochs': training_cfg['num_epochs'],
        'accumulation_steps': training_cfg['accumulation_steps'],
        'max_grad_norm': training_cfg['max_grad_norm'],
        'surface_loss_weight': training_cfg['surface_loss_weight'],
        'learning_rate': training_cfg['learning_rate'],
        'weight_decay': training_cfg['weight_decay'],
        'batch_size': dataloader_cfg['batch_size'],
        
        # Early stopping
        'early_stopping_patience': training_cfg['early_stopping_patience'],
        'early_stopping_min_delta': training_cfg['early_stopping_min_delta'],
        
        # Scheduler
        'scheduler_max_lr': training_cfg['scheduler_max_lr'],
        'scheduler_pct_start': training_cfg['scheduler_pct_start'],
        'scheduler_anneal_strategy': training_cfg['scheduler_anneal_strategy'],
        
        # Logging
        'log_dir': logging_cfg['log_dir'],
        'log_histogram_frequency': logging_cfg['log_histogram_frequency'],
        
        # W&B
        'wandb_entity': wandb_cfg['entity'],
        'wandb_project': wandb_cfg['project'],
        'wandb_notes': wandb_cfg.get('notes', ''),
        'wandb_tags': wandb_cfg.get('tags', []),
        'run_name': args.run_name or f"Pangu_lr{training_cfg['learning_rate']}_bs{dataloader_cfg['batch_size']}_acc{training_cfg['accumulation_steps']}",
        
        # Model and data info
        'model_name': config['model']['name'],
        'train_years': f"{dataset_cfg['train_years'][0]}-{dataset_cfg['train_years'][1]}",
        'val_years': f"{dataset_cfg['val_years'][0]}-{dataset_cfg['val_years'][1]}",
    }
    
    results = train(
        pangu, 
        train_loader, 
        val_loader, 
        surface_criterion, 
        upper_air_criterion, 
        optimizer, 
        device, 
        train_config,
        train_dataset,
        val_dataset
    )
    print("Training complete.")

    # Save final model
    model_state = {
        'epoch': training_cfg['num_epochs'],
        'model_state_dict': pangu.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(model_state, os.path.join(logging_cfg['log_dir'], 'final_model.pth'))
    
    print("Final model saved.")

