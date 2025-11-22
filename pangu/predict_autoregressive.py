#!/usr/bin/env python3
# File: predict_autoregressive.py
# 
# Autoregressive multi-step prediction that stores forecasts at EVERY 6-hour interval.
# Output shape: (Steps, C, Lat, Lon) where Steps = lead_time_hours // 6
# 
# For single lead-time predictions, use prediction.py instead.
import argparse
import torch
import numpy as np
import os
from torch.utils.data import DataLoader

from Pangu.Pangu_Weather_Prediction_Model.pangu.pangu_model import Pangu_lite
from Pangu.Pangu_Weather_Prediction_Model.pangu.data_utils import (
    ZarrWeatherDataset, surface_transform, upper_air_transform,
    surface_inv_transform, upper_air_inv_transform
)

def load_model(model_path, device):
    pangu_lite = Pangu_lite()
    pangu_lite.eval()
    if device.type == "cuda":
        pangu_lite.cuda()
        checkpoint = torch.load(model_path)
    else:
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    pangu_lite.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    return pangu_lite

def predict_autoregressive(model, dataloader, device, surface_mask, surface_invTrans, upper_air_invTrans,
                           pLevels, num_samples, max_lead_time_hours):
    model.eval()
    steps = max_lead_time_hours // 6
    predictions = []
    sample_count = 0
    
    print(f"⚡ Running autoregressive inference for {steps} steps ({max_lead_time_hours} hours)...")

    with torch.no_grad():
        for batch in dataloader:
            if sample_count >= num_samples: break
            
            input_surface = batch['surface'].to(device)
            input_upper_air = batch['upper_air'].to(device)
            
            # List to store (Surface, Upper) for EACH step for this batch
            # Shape will be: Steps -> Batch -> (Vars, Lat, Lon)
            batch_steps_surface = []
            batch_steps_upper = []

            for step in range(1, steps + 1):
                # 1. Run Model
                output_surface, output_upper_air = model(input_surface, surface_mask, input_upper_air)
                
                # 2. Save Normed Output for next step
                input_surface = output_surface
                input_upper_air = output_upper_air
                
                # 3. Denormalize just for saving (keep on CPU to save GPU mem)
                # Surface
                denorm_surf = surface_invTrans(output_surface).detach().cpu()
                batch_steps_surface.append(denorm_surf) # (B, C, Lon, Lat)
                
                # Upper Air (Per level)
                denorm_upper = torch.stack([
                    upper_air_invTrans[pl](output_upper_air[:, :, i, :, :]) 
                    for i, pl in enumerate(pLevels)
                ], dim=2).detach().cpu()
                batch_steps_upper.append(denorm_upper) # (B, C, Pl, Lon, Lat)

            # Stack steps: (Batch, Steps, ...)
            # Transpose list of tensors to tensor
            # dim 0 is steps, we want (Batch, Steps)
            final_surf = torch.stack(batch_steps_surface, dim=1).numpy()      # (B, Steps, C, Lon, Lat)
            final_upper = torch.stack(batch_steps_upper, dim=1).numpy()       # (B, Steps, C, Pl, Lon, Lat)
            
            # Process each sample in batch
            batch_size = final_surf.shape[0]
            for b in range(min(batch_size, num_samples - sample_count)):
                # Transpose to (Steps, C, Lat, Lon) for plotting standards
                surf_sample = np.transpose(final_surf[b], (0, 1, 3, 2))
                upper_sample = np.transpose(final_upper[b], (0, 1, 2, 4, 3))
                
                predictions.append({
                    'surface': surf_sample,
                    'upper_air': upper_sample,
                    'sample_idx': sample_count,
                    'start_time_index': batch['start_time_index'][b] if 'start_time_index' in batch else None
                })
                sample_count += 1
                print(f"  Sample {sample_count}/{num_samples} completed.")

    return predictions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--transform_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--lead_time", type=int, default=120, help="Max forecast hours (multiple of 6)")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_years", type=int, nargs='+', default=[2021, 2022, 2023])
    
    # Standard vars (can be defaulted)
    parser.add_argument("--surface_variables", type=str, nargs='+', default=["2m_temperature", "mean_sea_level_pressure", "10m_u_component_of_wind", "10m_v_component_of_wind"])
    parser.add_argument("--upper_air_variables", type=str, nargs='+', default=["geopotential", "specific_humidity", "temperature", "u_component_of_wind", "v_component_of_wind"])
    parser.add_argument("--pressure_levels", type=int, nargs='+', default=[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000])
    parser.add_argument("--static_variables", type=str, nargs='+', default=["land_sea_mask", "soil_type"])

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Transforms (Sorted)
    surface_normalizer, surface_vars = surface_transform(os.path.join(args.transform_dir, "surface_mean.pkl"), os.path.join(args.transform_dir, "surface_std.pkl"))
    upper_air_normalizer, upper_vars, plevels_stats = upper_air_transform(os.path.join(args.transform_dir, "upper_air_mean.pkl"), os.path.join(args.transform_dir, "upper_air_std.pkl"))
    surface_invTrans, _ = surface_inv_transform(os.path.join(args.transform_dir, "surface_mean.pkl"), os.path.join(args.transform_dir, "surface_std.pkl"))
    upper_air_invTrans, _, pLevels = upper_air_inv_transform(os.path.join(args.transform_dir, "upper_air_mean.pkl"), os.path.join(args.transform_dir, "upper_air_std.pkl"))

    # 2. Dataset
    test_dataset = ZarrWeatherDataset(
        zarr_path=args.data_path,
        surface_vars=args.surface_variables,
        upper_air_vars=args.upper_air_variables,
        plevels=args.pressure_levels,
        static_vars=args.static_variables,
        year_range=tuple(args.test_years),
        surface_transform=surface_normalizer,
        upper_air_transform=upper_air_normalizer,
        chunk_size=args.batch_size
    )
    dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 3. Static Mask
    static_mask = next(iter(dataloader))['static'].to(device)

    # 4. Predict
    model = load_model(args.model_path, device)
    preds = predict_autoregressive(model, dataloader, device, static_mask, surface_invTrans, upper_air_invTrans, pLevels, args.num_samples, args.lead_time)

    # 5. Save
    os.makedirs(args.output_path, exist_ok=True)
    # Metadata
    np.savez(os.path.join(args.output_path, "metadata.npz"), 
             surface_vars=surface_vars, upper_air_vars=upper_vars, 
             pressure_levels=pLevels, lead_time_hours=args.lead_time)

    for p in preds:
        np.savez(os.path.join(args.output_path, f"pred_{p['sample_idx']}.npz"), 
                 surface=p['surface'], upper_air=p['upper_air'], 
                 sample_idx=p['sample_idx'], 
                 surface_vars=surface_vars, upper_air_vars=upper_vars, p_levels=pLevels)
    
    print(f"✅ Saved {len(preds)} autoregressive predictions.")

if __name__ == "__main__":
    main()