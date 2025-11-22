#!/usr/bin/env python3
# File: evaluate_autoregressive.py
import argparse
import os
import numpy as np
import xarray as xr
import pandas as pd
import xskillscore as xs
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr_path", required=True)
    parser.add_argument("--prediction_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--climatology_path", default=None)
    parser.add_argument("--test_years", type=int, nargs='+', default=[2021, 2023],
                        help="Year range for test data (must match predict.py)")
    args = parser.parse_args()

    # 1. Load GT
    ds_gt = xr.open_zarr(args.zarr_path)
    
    # Get the master list of all possible test start times
    print(f"Finding test set start times ({args.test_years[0]}-{args.test_years[1]})...")
    test_slice = slice(f"{args.test_years[0]}-01-01", f"{args.test_years[1]}-12-31")
    test_times_da = ds_gt.sel(time=test_slice).time
    all_test_times = test_times_da.values 
    print(f"  Found {len(all_test_times)} test timestamps starting from {all_test_times[0]}")

    # 2. Load Climatology (Optional)
    clim_ds = xr.open_dataset(args.climatology_path) if args.climatology_path else None

    files = sorted([f for f in os.listdir(args.prediction_dir) if f.startswith("pred_")])
    all_stats = []

    print(f"🚀 Evaluating {len(files)} forecast streams...")

    for f in tqdm(files, desc="Processing Files"):
        data = np.load(os.path.join(args.prediction_dir, f), allow_pickle=True)
        
        sample_idx = int(data['sample_idx'])

        try:
            start_time = all_test_times[sample_idx]
        except IndexError:
            print(f"Warning: sample_idx {sample_idx} is out of bounds for test times. Skipping file {f}")
            continue
        
        surf_pred = data['surface']
        upper_pred = data['upper_air']
        steps = surf_pred.shape[0]
        lead_times = np.arange(1, steps + 1) * 6
        
        # Get actual spatial dimensions from prediction data
        # surf_pred shape: (steps, n_vars, latitude, longitude)
        pred_nlatitude = surf_pred.shape[-2]  # Should be 32
        pred_nlongitude = surf_pred.shape[-1]  # Should be 64
        
        print(f"Prediction shape: latitude={pred_nlatitude}, longitude={pred_nlongitude}")
        
        # Get coordinates from ground truth
        latitudes = ds_gt.latitude.values if 'latitude' in ds_gt else ds_gt.lat.values
        longitudes = ds_gt.longitude.values if 'longitude' in ds_gt else ds_gt.lon.values
        
        print(f"GT coordinates: latitude={len(latitudes)}, longitude={len(longitudes)}")
        
        # Make sure we have the right number of coordinates
        assert len(latitudes) == pred_nlatitude, f"latitude mismatch: GT has {len(latitudes)}, pred has {pred_nlatitude}"
        assert len(longitudes) == pred_nlongitude, f"longitude mismatch: GT has {len(longitudes)}, pred has {pred_nlongitude}"
        
        # Create weights based on correct latitude dimension
        # Broadcast to (latitude, longitude) shape
        weights = np.cos(np.deg2rad(latitudes))
        weights_2d = np.broadcast_to(weights[:, np.newaxis], (pred_nlatitude, pred_nlongitude))
        weights_da = xr.DataArray(
            weights_2d, 
            coords={"latitude": latitudes, "longitude": longitudes}, 
            dims=("latitude", "longitude"), 
            name="weights"
        )

        # --- Evaluate Surface ---
        for i, var in enumerate(data['surface_vars']):
            da_pred = xr.DataArray(
                surf_pred[:, i, ...], 
                coords={"lead_time": lead_times, "latitude": latitudes, "longitude": longitudes}, 
                dims=("lead_time", "latitude", "longitude")
            )

            target_list = []
            clim_list = []
            
            for step in lead_times:
                valid_time = start_time + np.timedelta64(step, 'h')
                
                t_da = ds_gt[var].sel(time=valid_time, method="nearest").squeeze()
                
                # Transpose if dimensions are in wrong order
                # Check if the data array dimensions match expected (latitude, longitude)
                latitude_dim = 'latitude' if 'latitude' in t_da.dims else 'lat'
                longitude_dim = 'longitude' if 'longitude' in t_da.dims else 'lon'
                
                # Ensure correct dimension order (latitude, longitude)
                t_da = t_da.transpose(latitude_dim, longitude_dim)
                target_list.append(t_da.values)

                if clim_ds:
                    ts = pd.Timestamp(valid_time) 
                    c = clim_ds[var].sel(dayofyear=ts.dayofyear)
                    if 'hour' in c.dims: c = c.sel(hour=ts.hour)
                    # Transpose climatology too
                    latitude_dim_c = 'latitude' if 'latitude' in c.dims else 'lat'
                    longitude_dim_c = 'longitude' if 'longitude' in c.dims else 'lon'
                    c = c.transpose(latitude_dim_c, longitude_dim_c)
                    clim_list.append(c.values)
                else:
                    clim_list.append(t_da.mean().values)

            da_target = xr.DataArray(np.stack(target_list), coords=da_pred.coords, dims=da_pred.dims)
            da_clim = xr.DataArray(np.stack(clim_list), coords=da_pred.coords, dims=da_pred.dims)

            acc = xs.pearson_r(da_pred - da_clim, da_target - da_clim, dim=["latitude", "longitude"], weights=weights_da)

            # This is calculated on the raw forecast error (pred vs target)
            rmse = xs.rmse(da_pred, da_target, dim=["latitude", "longitude"], weights=weights_da)

            # Store both metrics
            for step, acc_val, rmse_val in zip(lead_times, acc.values, rmse.values):
                all_stats.append({
                    "lead_time": step, "variable": var, "level": "surface", 
                    "acc": acc_val, "rmse": rmse_val
                })
        # --- Evaluate Upper Air ---
        for i, var in enumerate(data['upper_air_vars']):
            for j, pl in enumerate(data['p_levels']):
                da_pred = xr.DataArray(
                    upper_pred[:, i, j, ...], 
                    coords={"lead_time": lead_times, "latitude": latitudes, "longitude": longitudes}, 
                    dims=("lead_time", "latitude", "longitude")
                )
                
                target_list = []
                clim_list = []
                for step in lead_times:
                    valid_time = start_time + np.timedelta64(step, 'h')
                    level_name = 'level' if 'level' in ds_gt.dims else 'pressure_level'
                    t_da = ds_gt[var].sel(time=valid_time, method="nearest", **{level_name: pl}).squeeze()
                    
                    # Transpose if dimensions are in wrong order
                    latitude_dim = 'latitude' if 'latitude' in t_da.dims else 'lat'
                    longitude_dim = 'longitude' if 'longitude' in t_da.dims else 'lon'
                    t_da = t_da.transpose(latitude_dim, longitude_dim)
                    target_list.append(t_da.values)
                    
                    if clim_ds:
                        ts = pd.Timestamp(valid_time)
                        c = clim_ds[var].sel(dayofyear=ts.dayofyear)
                        if 'hour' in c.dims: c = c.sel(hour=ts.hour)
                        level_dim_c = 'level' if 'level' in c.dims else 'pressure_level'
                        c = c.sel(**{level_dim_c: pl})
                        # Transpose climatology too
                        latitude_dim_c = 'latitude' if 'latitude' in c.dims else 'lat'
                        longitude_dim_c = 'longitude' if 'longitude' in c.dims else 'lon'
                        c = c.transpose(latitude_dim_c, longitude_dim_c)
                        clim_list.append(c.values)
                    else:
                        clim_list.append(t_da.mean().values)

                da_target = xr.DataArray(np.stack(target_list), coords=da_pred.coords, dims=da_pred.dims)
                da_clim = xr.DataArray(np.stack(clim_list), coords=da_pred.coords, dims=da_pred.dims)
                
                acc = xs.pearson_r(da_pred - da_clim, da_target - da_clim, dim=["latitude", "longitude"], weights=weights_da)
                
                # *** NEW: RMSE (Root Mean Squared Error) ***
                rmse = xs.rmse(da_pred, da_target, dim=["latitude", "longitude"], weights=weights_da)

                # Store both metrics
                for step, acc_val, rmse_val in zip(lead_times, acc.values, rmse.values):
                    all_stats.append({
                        "lead_time": step, "variable": var, "level": pl, 
                        "acc": acc_val, "rmse": rmse_val
                    })
    os.makedirs(args.output_path, exist_ok=True)
    output_csv = os.path.join(args.output_path, "acc_lead_time.csv")
    pd.DataFrame(all_stats).to_csv(output_csv, index=False)
    print(f"✅ Results saved to {output_csv}")

if __name__ == "__main__":
    main()