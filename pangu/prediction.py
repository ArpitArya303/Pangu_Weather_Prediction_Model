#!/usr/bin/env python3
# File: test.py - Model prediction and inference
import argparse
import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from Pangu.Pangu_Weather_Prediction_Model.pangu.pangu_model import Pangu_lite
from Pangu.Pangu_Weather_Prediction_Model.pangu.data_utils import (
    ZarrWeatherDataset,
    surface_transform, 
    upper_air_transform,
    surface_inv_transform, 
    upper_air_inv_transform
)


def load_model(model_path: str, device: torch.device):
    """Load trained model from checkpoint."""
    pangu_lite = Pangu_lite()
    pangu_lite.eval()
    
    if device.type == "cuda":
        pangu_lite.cuda()
        checkpoint = torch.load(model_path)
    else:
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        pangu_lite.load_state_dict(checkpoint['model_state_dict'])
    else:
        pangu_lite.load_state_dict(checkpoint)
    
    return pangu_lite


def predict(model, dataloader, device, surface_mask, surface_invTrans, upper_air_invTrans,
           pLevels, num_samples=5, lead_time_hours=6):
    """
    Run inference and return denormalized predictions.
    
    Args:
        model: Trained Pangu model
        dataloader: DataLoader for test dataset
        device: torch device (cuda/cpu)
        surface_mask: Static surface fields
        surface_invTrans: Inverse transform for surface variables
        upper_air_invTrans: Inverse transform dict for upper air variables
        pLevels: List of pressure levels
        num_samples: Number of samples to predict
        lead_time_hours: Lead time in hours (must be multiple of 6)
        
    Returns:
        List of dicts containing predictions with keys:
        - 'surface': numpy array (C, Lat, Lon)
        - 'upper_air': numpy array (C, Pl, Lat, Lon)
        - 'sample_idx': int
    """
    model.eval()
    
    num_iterations = lead_time_hours // 6
    assert lead_time_hours % 6 == 0, "lead_time_hours must be a multiple of 6"
    
    predictions = []
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if sample_count >= num_samples:
                break
            
            # Get input tensors
            input_surface = batch['surface'].to(device)
            input_upper_air = batch['upper_air'].to(device)
            
            # Iterative prediction for lead_time
            for _ in range(num_iterations):
                output_surface, output_upper_air = model(input_surface, surface_mask, input_upper_air)
                input_surface = output_surface
                input_upper_air = output_upper_air
            
            # Denormalize
            output_surface = surface_invTrans(output_surface)  # B, C, Lon, Lat
            
            # Stack upper air with inverse transform per pressure level
            output_upper_air = torch.stack(
                [upper_air_invTrans[pl](output_upper_air[:, :, i, :, :]) 
                 for i, pl in enumerate(pLevels)], 
                dim=2
            )  # B, C, Pl, Lon, Lat
            
            # Move to CPU
            output_surface = output_surface.detach().cpu().numpy()  # B, C, Lon, Lat
            output_upper_air = output_upper_air.detach().cpu().numpy()  # B, C, Pl, Lon, Lat
            
            # Process each sample in the batch
            batch_size = output_surface.shape[0]
            for b in range(min(batch_size, num_samples - sample_count)):
                # Extract single sample: (C, Lon, Lat)
                surface_sample = output_surface[b]
                upper_air_sample = output_upper_air[b]  # C, Pl, Lon, Lat
                
                # Transpose data from (C, Lon, Lat) to (C, Lat, Lon) for plotting
                surface_sample = np.transpose(surface_sample, (0, 2, 1))  # C, Lat, Lon
                upper_air_sample = np.transpose(upper_air_sample, (0, 1, 3, 2))  # C, Pl, Lat, Lon
                
                predictions.append({
                    'surface': surface_sample,
                    'upper_air': upper_air_sample,
                    'sample_idx': sample_count
                })
                
                sample_count += 1
    
    return predictions

def debug_channel_ranges(surface_tensor, surface_vars):
    arr = surface_tensor.detach().cpu().numpy()
    for i, v in enumerate(surface_vars):
        ch = arr[0, i]  # first sample
        print(f"[DEBUG] {v}: min={ch.min():.2f} max={ch.max():.2f} mean={ch.mean():.2f}")

def main():
    parser = argparse.ArgumentParser(description="Run weather prediction inference.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to Zarr dataset")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--transform_dir", type=str, required=True, help="Path to normalization stats directory")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to save predictions (.npz files)")
    parser.add_argument("--surface_variables", type=str, nargs='+', 
                       default=["2m_temperature", "mean_sea_level_pressure", 
                               "10m_u_component_of_wind", "10m_v_component_of_wind"],
                       help="List of surface variable names")
    parser.add_argument("--upper_air_variables", type=str, nargs='+',
                       default=["geopotential", "specific_humidity", "temperature",
                               "u_component_of_wind", "v_component_of_wind"],
                       help="List of upper-air variable names")
    parser.add_argument("--pressure_levels", type=int, nargs='+',
                       default=[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
                       help="List of pressure levels for upper-air variables")
    parser.add_argument("--static_variables", type=str, nargs='+',
                       default=["land_sea_mask", "soil_type"],
                       help="List of static variables")
    parser.add_argument("--test_years", type=int, nargs='+', default=[2021, 2023],
                       help="Year range for test data")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--num_samples", type=int, default=5, help="Maximum number of samples to predict")
    parser.add_argument("--lead_time", type=int, default=24, help="Lead time in hours (must be multiple of 6)")

    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_gpu = device.type == "cuda"
    print(f"Using device: {device}")

    # Load normalization transforms
    print("\nLoading normalization statistics...")
    surface_normalizer, surface_vars = surface_transform(
        os.path.join(args.transform_dir, "surface_mean.pkl"),
        os.path.join(args.transform_dir, "surface_std.pkl")
    )
    
    upper_air_normalizer, upper_air_vars, plevels = upper_air_transform(
        os.path.join(args.transform_dir, "upper_air_mean.pkl"),
        os.path.join(args.transform_dir, "upper_air_std.pkl")
    )
    
    # Load inverse transforms
    print("Loading inverse transforms...")
    surface_invTrans, inv_surface_vars = surface_inv_transform(
        os.path.join(args.transform_dir, "surface_mean.pkl"),
        os.path.join(args.transform_dir, "surface_std.pkl")
    )
    print("[DEBUG] Surface variable ranges after inverse transform:")

    debug_channel_ranges(surface_invTrans(torch.zeros(1, len(inv_surface_vars), 64, 32)), inv_surface_vars)

    upper_air_invTrans, inv_upper_vars, pLevels = upper_air_inv_transform(
        os.path.join(args.transform_dir, "upper_air_mean.pkl"),
        os.path.join(args.transform_dir, "upper_air_std.pkl")
    )

    # Create test dataset
    print(f"\nLoading test data ({args.test_years[0]}-{args.test_years[1]})...")
    print(f"Pressure levels: {args.pressure_levels}")
    
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
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=is_gpu
    )
    
    # Get surface mask (constant fields)
    print("Loading surface mask...")
    first_batch = next(iter(test_loader))
    surface_mask = first_batch['static']
    if is_gpu:
        surface_mask = surface_mask.cuda()
    
    print(f"Test dataset size: {len(test_dataset)} samples")
    print(f"Surface variables: {args.surface_variables}")
    print(f"Upper air variables: {args.upper_air_variables}")

    # Load model
    print(f"\nLoading model from {args.model_path}...")
    model = load_model(args.model_path, device)

    # Run predictions
    print(f"\nGenerating predictions (lead time: {args.lead_time}h)...")
    predictions = predict(
        model, 
        test_loader, 
        device, 
        surface_mask=surface_mask,
        surface_invTrans=surface_invTrans,
        upper_air_invTrans=upper_air_invTrans,
        pLevels=pLevels,
        num_samples=args.num_samples,
        lead_time_hours=args.lead_time
    )
    
    # Save predictions to disk
    os.makedirs(args.output_path, exist_ok=True)
    # Save a one-time metadata file for plotting/mapping
    meta_path = os.path.join(args.output_path, "metadata.npz")
    if not os.path.exists(meta_path):
        np.savez(
            meta_path,
            surface_vars=np.array(surface_vars, dtype=object),
            upper_air_vars=np.array(upper_air_vars, dtype=object),
            pressure_levels=np.array(pLevels, dtype=np.int32),
            lead_time_hours=args.lead_time,
        )
        print(f"Saved metadata: {meta_path}")

    for pred in predictions:
        sample_idx = pred['sample_idx']

        # These are already denormalized and transposed to (C, Lat, Lon)/(C, Pl, Lat, Lon)
        surface_arr = pred['surface']         # shape: (Cs, Lat, Lon)
        upper_air_arr = pred['upper_air']     # shape: (Cu, Pl, Lat, Lon)

        output_file = os.path.join(args.output_path, f'prediction_sample{sample_idx}.npz')
        np.savez(
            output_file,
            sample_idx=sample_idx,
            surface=surface_arr,
            upper_air=upper_air_arr,
            p_levels=np.array(pLevels, dtype=np.int32),
            surface_vars=np.array(surface_vars, dtype=object),
            upper_air_vars=np.array(upper_air_vars, dtype=object),
        )
        print(f"Saved prediction: {output_file}")

    print(f"\nPrediction complete! Saved {len(predictions)} predictions to {args.output_path}")

if __name__ == "__main__":
    main()
