import argparse
import os
import numpy as np
import xarray as xr
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from Pangu.Pangu_Weather_Prediction_Model.pangu.pangu_model import Pangu_lite
from Pangu.Pangu_Weather_Prediction_Model.pangu.data_utils import (
    ZarrWeatherDataset, 
    surface_transform, 
    upper_air_transform, 
)


def train_step(model, dataloader, surface_criterion, upper_air_criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        # Unpack dictionary from dataset
        input_surface = batch['surface']
        input_upper_air = batch['upper_air']
        static = batch['static']
        target_surface = batch['surface_target']
        target_upper_air = batch['upper_air_target']

        # Move to device
        input_surface = input_surface.to(device)
        input_upper_air = input_upper_air.to(device)
        static = static.to(device)
        target_surface = target_surface.to(device)
        target_upper_air = target_upper_air.to(device)

        batch_size = input_surface.size(0)
        
        optimizer.zero_grad()
        pred_surf, pred_upper = model(input_surface, static, input_upper_air)

        loss_surf = surface_criterion(pred_surf, target_surface)
        loss_upper = upper_air_criterion(pred_upper, target_upper_air)
        loss = loss_surf*0.25 + loss_upper

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size

    epoch_loss = running_loss / len(dataloader.dataset)
    
    return epoch_loss

def val_step(model, dataloader, surface_criterion, upper_air_criterion, device, surface_inv_trans=None, upper_air_inv_trans=None):
    model.eval()
    val_loss = 0.0
    surface_mse = 0.0
    upper_air_mse = 0.0

    with torch.no_grad():
        for batch in dataloader:
            # Unpack dictionary from dataset
            input_surface = batch['surface']
            input_upper_air = batch['upper_air']
            static = batch['static']
            target_surface = batch['surface_target']
            target_upper_air = batch['upper_air_target']

            # Move to device
            input_surface = input_surface.to(device)
            input_upper_air = input_upper_air.to(device)
            static = static.to(device)
            target_surface = target_surface.to(device)
            target_upper_air = target_upper_air.to(device)

            batch_size = input_surface.size(0)

            pred_surf, pred_upper = model(input_surface, static, input_upper_air)

            loss_surf = surface_criterion(pred_surf, target_surface)
            loss_upper = upper_air_criterion(pred_upper, target_upper_air)
            loss = loss_surf*0.25 + loss_upper
            
            val_loss += loss.item() * batch_size
            surface_mse += ((pred_surf - target_surface) ** 2).mean().item() * batch_size
            upper_air_mse += ((pred_upper - target_upper_air) ** 2).mean().item() * batch_size

    epoch_val_loss = val_loss / len(dataloader.dataset)
    epoch_surface_mse = surface_mse / len(dataloader.dataset)
    epoch_upper_air_mse = upper_air_mse / len(dataloader.dataset)
    
    return epoch_val_loss, epoch_surface_mse, epoch_upper_air_mse

def train(model, train_loader, val_loader, surface_criterion, upper_air_criterion, optimizer, device, num_epochs, log_dir):
    writer = SummaryWriter(log_dir=log_dir)
    
    for epoch in range(1, num_epochs + 1):
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} - Training")
        train_loss = train_step(model, train_bar, surface_criterion, upper_air_criterion, optimizer, device)
        val_loss, surface_mse, upper_air_mse = val_step(model, val_loader, surface_criterion, upper_air_criterion, device)

        # Log metrics to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('MSE/surface', surface_mse, epoch)
        writer.add_scalar('MSE/upper_air', upper_air_mse, epoch)

        print(f"Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Surface MSE: {surface_mse:.4f}, Upper Air MSE: {upper_air_mse:.4f}")
        # history logging can be done here if needed
        for name, param in model.named_parameters():
            writer.add_histogram(name, param, epoch)
            writer.add_histogram(f'{name}.grad', param.grad, epoch)

        early_stopping_patience = 10
        best_train_loss = float("inf")
        epochs_no_improve = 0

        # Early stopping and model checkpointing can be implemented here if needed
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(log_dir, 'best_model.pth'))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print("Early stopping triggered")
                break
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="/home/bedartha/public/datasets/as_downloaded/weatherbench2/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr", help="Path to the Zarr dataset")
    parser.add_argument("--surface_variables", nargs='+', default=["2m_temperature","mean_sea_level_pressure","10m_u_component_of_wind","10m_v_component_of_wind"], help="Surface variables")
    parser.add_argument("--upper_air_variables", nargs='+', default=["geopotential","specific_humidity","temperature", "u_component_of_wind", "v_component_of_wind"], help="Upper air variables")
    parser.add_argument("--pLevels", nargs='+', default=[50,100,150,200,250,300,400,500,600,700,850,925,1000], type=int, help="Pressure levels for upper air variables")
    parser.add_argument("--static_variables", nargs='+', default=["land_sea_mask", "soil_type"], help="Static variables")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--log_dir", default="runs/trial", help="Directory to save logs")
    parser.add_argument("--transform_dir", type=str, default="/storage/arpit/Pangu/Pangu_Weather_Prediction_Model/pangu/data", help="Directory containing mean and std pickle files for transforms")
    opt = parser.parse_args()
    
    print("Preparing indices...")
    dataset = xr.open_zarr(opt.data)
    time_len = dataset.sizes['time'] - 1
    indices = np.arange(time_len)
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

    # Train dataset creation
    train_dataset = ZarrWeatherDataset(
        zarr_path=opt.data, 
        surface_vars=opt.surface_variables,
        upper_air_vars=opt.upper_air_variables,
        plevels=opt.pLevels,
        static_vars=opt.static_variables,
        indices=train_indices,
        surface_transform=surface_normalizer,  
        upper_air_transform=upper_air_normalizer  
    )

    # Create validation dataset
    val_dataset = ZarrWeatherDataset(
        zarr_path=opt.data,
        surface_vars=opt.surface_variables,
        upper_air_vars=opt.upper_air_variables,
        plevels=opt.pLevels,
        static_vars=opt.static_variables,
        indices=val_indices,
        surface_transform=surface_normalizer,
        upper_air_transform=upper_air_normalizer
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)

    print("Setting up device...")
    device = torch.device("cuda") # if torch.cuda.is_available() else "cpu")

    print("Initializing model, criteria, and optimizer...")
    pangu = Pangu_lite().to(device)

    surface_criterion = nn.L1Loss()
    upper_air_criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(pangu.parameters(), lr=5e-4, weight_decay=3e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.num_epochs)
    
    print("Starting training...")
    results = train(pangu, train_loader, val_loader, surface_criterion, upper_air_criterion, optimizer, device, opt.num_epochs, opt.log_dir)
    print("Training complete.")

    # Save final model
    torch.save(pangu.state_dict(), os.path.join(opt.log_dir, 'final_model.pth'))
    print("Final model saved.")

