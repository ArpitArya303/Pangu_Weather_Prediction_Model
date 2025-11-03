#!/usr/bin/env python3
# file: plotting.py - Comprehensive plotting utilities for Pangu weather predictions
import argparse
import numpy as np
import os
import sys
import xarray as xr
import matplotlib.pyplot as plt

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.util import add_cyclic_point
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("Warning: cartopy not installed. Using basic matplotlib plots.")


def get_variable_units(var_name):
    """Get standard units for common meteorological variables."""
    units_map = {
        '2m_temperature': 'K',
        'mean_sea_level_pressure': 'Pa',
        '10m_u_component_of_wind': 'm/s',
        '10m_v_component_of_wind': 'm/s',
        'geopotential': 'm²/s²',
        'specific_humidity': 'kg/kg',
        'temperature': 'K',
        'u_component_of_wind': 'm/s',
        'v_component_of_wind': 'm/s',
    }
    return units_map.get(var_name, '')


def plot_with_cartopy(data, lat, lon, var_name, output_path, cmap='viridis', title='', units=''):
    """Plot single data field on geographic map with cartopy."""
    if not HAS_CARTOPY:
        print("Cartopy not available, using basic plot instead")
        plot_basic(data, var_name, output_path, cmap, title, units)
        return
    
    # Add cyclic point to avoid white line at dateline
    data_cyclic, lon_cyclic = add_cyclic_point(data, coord=lon)
    
    # Create meshgrid
    lon_grid, lat_grid = np.meshgrid(lon_cyclic, lat)
    
    # Create figure with cartopy projection
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    
    # Plot data using contourf with meshgrid coordinates
    mesh = ax.contourf(lon_grid, lat_grid, data_cyclic, levels=20, 
                      transform=ccrs.PlateCarree(), cmap=cmap, extend='both')
    
    # Add colorbar with units if provided
    label = f'{var_name} ({units})' if units else var_name
    plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.05, label=label)
    
    # Title
    plt.title(title, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {output_path}")


def plot_basic(data, var_name, output_path, cmap='viridis', title='', units=''):
    """Plot data with basic matplotlib (no cartopy)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(data, cmap=cmap, aspect='auto', origin='lower')
    
    # Add colorbar with units if provided
    label = f'{var_name} ({units})' if units else var_name
    plt.colorbar(im, ax=ax, label=label, orientation='horizontal', pad=0.05)
    
    plt.title(title, fontsize=12)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {output_path}")


def plot_comparison(ground_truth, prediction, lat, lon, var_name, output_path, 
                   cmap='viridis', diff_cmap='RdBu_r', title='', units='', lead_time_hours=6):
    """
    Create a 3-panel comparison plot: Ground Truth | Prediction | Difference.
    
    Args:
        ground_truth: numpy array (Lat, Lon)
        prediction: numpy array (Lat, Lon)
        lat: latitude array
        lon: longitude array
        var_name: variable name
        output_path: path to save plot
        cmap: colormap for data
        diff_cmap: colormap for difference (diverging)
        title: title prefix
        units: variable units
        lead_time_hours: forecast lead time
    """
    if not HAS_CARTOPY:
        print("Cartopy not available, skipping comparison plot")
        return
    
    # Calculate difference
    difference = prediction - ground_truth
    
    # Add cyclic points
    gt_cyclic, lon_cyclic = add_cyclic_point(ground_truth, coord=lon)
    pred_cyclic, _ = add_cyclic_point(prediction, coord=lon)
    diff_cyclic, _ = add_cyclic_point(difference, coord=lon)
    
    # Create meshgrid
    lon_grid, lat_grid = np.meshgrid(lon_cyclic, lat)
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(18, 5))
    
    # Determine common color scale for GT and Prediction
    vmin = min(np.min(gt_cyclic), np.min(pred_cyclic))
    vmax = max(np.max(gt_cyclic), np.max(pred_cyclic))
    
    # Difference color scale (symmetric around zero)
    diff_max = max(abs(np.min(diff_cyclic)), abs(np.max(diff_cyclic)))
    diff_vmin, diff_vmax = -diff_max, diff_max
    
    # Panel 1: Ground Truth
    ax1 = fig.add_subplot(1, 3, 1, projection=ccrs.PlateCarree())
    ax1.set_global()
    ax1.coastlines(linewidth=0.5)
    ax1.add_feature(cfeature.BORDERS, linewidth=0.5)
    mesh1 = ax1.contourf(lon_grid, lat_grid, gt_cyclic, levels=20, 
                        transform=ccrs.PlateCarree(), cmap=cmap, 
                        vmin=vmin, vmax=vmax, extend='both')
    ax1.set_title(f'Ground Truth', fontsize=11, fontweight='bold')
    cbar1 = plt.colorbar(mesh1, ax=ax1, orientation='horizontal', pad=0.05, shrink=0.8)
    cbar1.ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))

    if units:
        cbar1.set_label(f'{var_name} ({units})', fontsize=9)
    
    # Panel 2: Prediction
    ax2 = fig.add_subplot(1, 3, 2, projection=ccrs.PlateCarree())
    ax2.set_global()
    ax2.coastlines(linewidth=0.5)
    ax2.add_feature(cfeature.BORDERS, linewidth=0.5)
    mesh2 = ax2.contourf(lon_grid, lat_grid, pred_cyclic, levels=20, 
                        transform=ccrs.PlateCarree(), cmap=cmap, 
                        vmin=vmin, vmax=vmax, extend='both')
    ax2.set_title(f'Prediction', fontsize=11, fontweight='bold')
    cbar2 = plt.colorbar(mesh2, ax=ax2, orientation='horizontal', pad=0.05, shrink=0.8)
    cbar2.ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))

    if units:
        cbar2.set_label(f'{var_name} ({units})', fontsize=9)
    
    # Panel 3: Difference (Prediction - Ground Truth)
    ax3 = fig.add_subplot(1, 3, 3, projection=ccrs.PlateCarree())
    ax3.set_global()
    ax3.coastlines(linewidth=0.5)
    ax3.add_feature(cfeature.BORDERS, linewidth=0.5)
    mesh3 = ax3.contourf(lon_grid, lat_grid, diff_cyclic, levels=20, 
                        transform=ccrs.PlateCarree(), cmap=diff_cmap, 
                        vmin=diff_vmin, vmax=diff_vmax, extend='both')
    ax3.set_title(f'Difference (Pred - GT)', fontsize=11, fontweight='bold')
    cbar3 = plt.colorbar(mesh3, ax=ax3, orientation='horizontal', pad=0.05, shrink=0.8)
    cbar3.ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))

    if units:
        cbar3.set_label(f'Δ{var_name} ({units})', fontsize=9)
    
    # Overall title
    fig.suptitle(f'{title}', fontsize=13, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to: {output_path}")


def plot_multi_comparison(ground_truth_list, prediction_list, lat, lon, var_names, 
                         output_path, title='Multi-Variable Comparison', lead_time_hours=6):
    """
    Create a grid of comparison plots for multiple variables.
    Each row shows: Ground Truth | Prediction | Difference for one variable.
    
    Args:
        ground_truth_list: list of numpy arrays (Lat, Lon)
        prediction_list: list of numpy arrays (Lat, Lon)
        lat: latitude array
        lon: longitude array
        var_names: list of variable names
        output_path: path to save plot
        title: overall title
        lead_time_hours: forecast lead time
    """
    if not HAS_CARTOPY:
        print("Cartopy not available, skipping multi-comparison plot")
        return
    
    n_vars = len(var_names)
    fig = plt.figure(figsize=(18, 5 * n_vars))
    
    for idx, (gt, pred, var_name) in enumerate(zip(ground_truth_list, prediction_list, var_names)):
        units = get_variable_units(var_name)
        difference = pred - gt
        
        # Add cyclic points
        gt_cyclic, lon_cyclic = add_cyclic_point(gt, coord=lon)
        pred_cyclic, _ = add_cyclic_point(pred, coord=lon)
        diff_cyclic, _ = add_cyclic_point(difference, coord=lon)
        
        # Create meshgrid
        lon_grid, lat_grid = np.meshgrid(lon_cyclic, lat)
        
        # Determine color scales
        vmin = min(np.min(gt_cyclic), np.min(pred_cyclic))
        vmax = max(np.max(gt_cyclic), np.max(pred_cyclic))
        diff_max = max(abs(np.min(diff_cyclic)), abs(np.max(diff_cyclic)))
        
        # Ground Truth
        ax1 = fig.add_subplot(n_vars, 3, idx*3 + 1, projection=ccrs.PlateCarree())
        ax1.set_global()
        ax1.coastlines(linewidth=0.5)
        ax1.add_feature(cfeature.BORDERS, linewidth=0.5)
        mesh1 = ax1.contourf(lon_grid, lat_grid, gt_cyclic, levels=20,
                            transform=ccrs.PlateCarree(), cmap='viridis',
                            vmin=vmin, vmax=vmax, extend='both')
        ax1.set_title(f'{var_name} - Ground Truth', fontsize=10, fontweight='bold')
        cbar1 = plt.colorbar(mesh1, ax=ax1, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar1.ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

        if units:
            cbar1.set_label(f'({units})', fontsize=8)
        
        # Prediction
        ax2 = fig.add_subplot(n_vars, 3, idx*3 + 2, projection=ccrs.PlateCarree())
        ax2.set_global()
        ax2.coastlines(linewidth=0.5)
        ax2.add_feature(cfeature.BORDERS, linewidth=0.5)
        mesh2 = ax2.contourf(lon_grid, lat_grid, pred_cyclic, levels=20,
                            transform=ccrs.PlateCarree(), cmap='viridis',
                            vmin=vmin, vmax=vmax, extend='both')
        ax2.set_title(f'Prediction (t+{lead_time_hours}h)', fontsize=10, fontweight='bold')
        cbar2 = plt.colorbar(mesh2, ax=ax2, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar2.ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        
        if units:
            cbar2.set_label(f'({units})', fontsize=8)
        
        # Difference
        ax3 = fig.add_subplot(n_vars, 3, idx*3 + 3, projection=ccrs.PlateCarree())
        ax3.set_global()
        ax3.coastlines(linewidth=0.5)
        ax3.add_feature(cfeature.BORDERS, linewidth=0.5)
        mesh3 = ax3.contourf(lon_grid, lat_grid, diff_cyclic, levels=20,
                            transform=ccrs.PlateCarree(), cmap='RdBu_r',
                            vmin=-diff_max, vmax=diff_max, extend='both')
        ax3.set_title(f'Difference (Pred - GT)', fontsize=10, fontweight='bold')
        cbar3 = plt.colorbar(mesh3, ax=ax3, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar3.ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

        if units:
            cbar3.set_label(f'Δ({units})', fontsize=8)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Multi-comparison plot saved to: {output_path}")

def plot_variable_from_zarr(zarr_path: str, output_path: str, var_name: str, cmap: str = 'viridis', time_index: int = 0):
    """Plot a variable from a Zarr dataset as a global map."""
    try:
        ds = xr.open_zarr(zarr_path)
    except Exception as e:
        print(f"Error opening Zarr dataset: {e}")
        sys.exit(1)

    if var_name not in ds.variables:
        print(f"Variable '{var_name}' not found. Available variables:\n{list(ds.variables)}")
        sys.exit(1)

    da = ds[var_name]
    if 'time' in da.dims:
        da = da.isel(time=time_index)

    lats = da["latitude"].values
    lons = da["longitude"].values
    data = da.values

    # Transpose if needed
    if data.shape[0] == len(lons) and data.shape[1] == len(lats):
        data = data.T

    if not os.path.exists(output_path):
        print(f"Output path '{output_path}' does not exist.")
        sys.exit(1)

    output_file = os.path.join(output_path, f"{var_name}_time{time_index}.png")
    units = da.attrs.get("units", "") or get_variable_units(var_name)
    title = f"{var_name} ground truth"
    
    # Use the unified plotting function
    plot_with_cartopy(data, lats, lons, var_name, output_file, cmap=cmap, title=title, units=units)


def plot_testset_visualizations(zarr_path: str, output_path: str, var_name: str, cmap: str = 'viridis', num_samples: int = 5):
    """Plot first few timestamps of test set (year >= 2021) from the Zarr dataset."""
    try:
        ds = xr.open_zarr(zarr_path)
    except Exception as e:
        print(f"Error opening Zarr dataset: {e}")
        sys.exit(1)

    if var_name not in ds.variables:
        print(f"Variable '{var_name}' not found. Available variables:\n{list(ds.variables)}")
        sys.exit(1)

    if 'time' not in ds[var_name].dims:
        print(f"Variable '{var_name}' has no 'time' dimension.")
        sys.exit(1)

    test_times = ds['time'].sel(time=ds['time'].dt.year >= 2021)
    if test_times.size == 0:
        print("No test set timestamps (>=2021) found.")
        sys.exit(1)

    selected_times = test_times[:num_samples]
    print(f"Plotting first {len(selected_times)} timestamps from test set: {selected_times.values}")

    for idx, t in enumerate(selected_times):
        time_idx = int(ds['time'].get_index('time').get_loc(t.values))
        print(f"Plotting time index {time_idx} ({str(t.values)})...")
        plot_variable_from_zarr(zarr_path, output_path, var_name, cmap, time_idx)


def plot_predictions_vs_ground_truth(prediction_dir, zarr_path, output_dir, 
                                     surface_variables, upper_air_variables,
                                     test_years=[2021, 2023], lead_time_hours=24,
                                     pressure_levels=None):
    """
    Load predictions from test_clean.py output and compare with ground truth.
    
    Args:
        prediction_dir: Directory containing .npz prediction files from test_clean.py
        zarr_path: Path to Zarr dataset for ground truth
        output_dir: Directory to save comparison plots
        surface_variables: List of surface variable names
        upper_air_variables: List of upper air variable names
        test_years: Year range for test data
        lead_time_hours: Lead time used in predictions
        pressure_levels: List of pressure levels to plot (None = plot all levels)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Zarr dataset for ground truth
    print(f"Loading ground truth from {zarr_path}...")
    try:
        ds = xr.open_zarr(zarr_path)
    except Exception as e:
        print(f"Error opening Zarr dataset: {e}")
        sys.exit(1)
    
    # Filter test set times
    test_times = ds['time'].sel(time=ds['time'].dt.year >= test_years[0])
    if test_times.size == 0:
        print(f"No test set timestamps (>={test_years[0]}) found.")
        sys.exit(1)
    
    # Get grid dimensions from dataset
    lats = ds['latitude'].values
    lons = ds['longitude'].values
    
    # Find all prediction files
    pred_files = sorted([f for f in os.listdir(prediction_dir) if f.endswith('.npz')])
    
    if len(pred_files) == 0:
        print(f"No prediction files found in {prediction_dir}")
        sys.exit(1)
    
    print(f"Found {len(pred_files)} prediction files")
    
    # Process each prediction file
    for pred_file in pred_files:
        pred_path = os.path.join(prediction_dir, pred_file)
        print(f"\nProcessing {pred_file}...")
        
        # Load prediction
        pred_data = np.load(pred_path)
        surface_pred = pred_data['surface']  # Shape: (C, Lat, Lon)
        upper_air_pred = pred_data['upper_air']  # Shape: (C, Pl, Lat, Lon)
        pred_levels = pred_data['p_levels'] if 'p_levels' in pred_data.files else None
        sample_idx = int(pred_data['sample_idx'])
        
        # Get corresponding ground truth time index
        # Predictions are made for lead_time_hours ahead
        if sample_idx >= len(test_times):
            print(f"Sample index {sample_idx} exceeds test set size, skipping...")
            continue
        
        # Ground truth at the forecast time (t + lead_time_hours)
        # Calculate the target time index
        forecast_steps = lead_time_hours // 6  # Number of 6-hour steps
        target_time_idx = sample_idx + forecast_steps
        
        if target_time_idx >= len(test_times):
            print(f"Target time index {target_time_idx} exceeds test set size, skipping...")
            continue
        
        # Get ground truth time
        target_time = test_times[target_time_idx]
        time_idx_in_full_dataset = int(ds['time'].get_index('time').get_loc(target_time.values))
        
        print(f"Sample {sample_idx}: Forecast time index = {target_time_idx}, Time = {str(target_time.values)}")
        
        # Plot surface variables
        for ch_idx, var in enumerate(surface_variables):
            if var not in ds.variables:
                print(f"Variable '{var}' not found in dataset, skipping...")
                continue
            
            # Get ground truth
            gt_data = ds[var].isel(time=time_idx_in_full_dataset).values
            
            # Transpose if needed (data should be Lat x Lon)
            if gt_data.shape[0] == len(lons) and gt_data.shape[1] == len(lats):
                gt_data = gt_data.T
            
            # Get prediction (already in Lat x Lon format)
            pred_var = surface_pred[ch_idx]
            
            # Create comparison plot
            units = get_variable_units(var)
            title = f'Surface {var}'
            save_path = os.path.join(output_dir, f'comparison_surface_{var}_sample{sample_idx}.png')
            
            plot_comparison(gt_data, pred_var, lats, lons, var, save_path,
                          cmap='viridis', diff_cmap='RdBu_r', title=title, 
                          units=units, lead_time_hours=lead_time_hours)
        
        # Plot upper-air variables at each pressure level
        for ch_idx, var in enumerate(upper_air_variables):
            if var not in ds.variables:
                print(f"Variable '{var}' not found in dataset, skipping...")
                continue
            
            # Get ground truth - need to handle pressure levels
            if 'level' in ds[var].dims or 'pressure_level' in ds[var].dims:
                # Dataset has pressure levels
                level_dim = 'level' if 'level' in ds[var].dims else 'pressure_level'
                gt_data_all_levels = ds[var].isel(time=time_idx_in_full_dataset)  # Has level dimension
                available_pressure_levels = ds[level_dim].values  # e.g., array([50., 100., ...])

                # Get prediction for this variable (has Pl dimension)
                pred_var_all_levels = upper_air_pred[ch_idx]  # Shape: (Pl, Lat, Lon)

                # Determine which pressure levels to plot
                if pressure_levels is not None:
                    # Plot only specified pressure levels
                    levels_to_plot = pressure_levels
                elif pred_levels is not None:
                    # Plot the pressure levels that predictions were produced for
                    levels_to_plot = pred_levels.tolist()
                else:
                    # Plot all available dataset pressure levels
                    levels_to_plot = available_pressure_levels.tolist()

                # Plot each pressure level value by matching GT index and Pred index separately
                for pressure_level in levels_to_plot:
                    # Find dataset level index matching this pressure level (tolerant match)
                    ds_matches = np.where(np.isclose(available_pressure_levels.astype(float), float(pressure_level), atol=0.5))[0]
                    if ds_matches.size == 0:
                        print(f"Dataset does not have level {pressure_level} hPa for {var}, skipping...")
                        continue
                    ds_pl_idx = int(ds_matches[0])

                    # Find prediction level index; if pred_levels not provided, assume same ordering as dataset
                    if pred_levels is not None:
                        pr_matches = np.where(np.isclose(pred_levels.astype(float), float(pressure_level), atol=0.5))[0]
                        if pr_matches.size == 0:
                            print(f"Predictions do not have level {pressure_level} hPa for {var}, skipping...")
                            continue
                        pr_pl_idx = int(pr_matches[0])
                    else:
                        pr_pl_idx = ds_pl_idx

                    if pr_pl_idx >= pred_var_all_levels.shape[0]:
                        print(f"Prediction level index {pr_pl_idx} out of bounds for {var}, skipping level {pressure_level} hPa...")
                        continue

                    # Get ground truth and prediction at this pressure level
                    gt_data = gt_data_all_levels.isel({level_dim: ds_pl_idx}).values
                    # Transpose if needed (data should be Lat x Lon)
                    if gt_data.shape[0] == len(lons) and gt_data.shape[1] == len(lats):
                        gt_data = gt_data.T
                    pred_var = pred_var_all_levels[pr_pl_idx]

                    # Create comparison plot
                    units = get_variable_units(var)
                    pl_int = int(round(float(pressure_level)))
                    title = f'Upper Air {var} @ {pl_int}hPa'
                    save_path = os.path.join(output_dir, f'comparison_upper_{var}_{pl_int}hPa_sample{sample_idx}.png')

                    plot_comparison(gt_data, pred_var, lats, lons, var, save_path,
                                    cmap='viridis', diff_cmap='RdBu_r', title=title,
                                    units=units, lead_time_hours=lead_time_hours)
            else:
                # Variable has no pressure levels
                gt_data = ds[var].isel(time=time_idx_in_full_dataset).values
                
                # Transpose if needed
                if gt_data.shape[0] == len(lons) and gt_data.shape[1] == len(lats):
                    gt_data = gt_data.T
                
                # Get prediction (average over Pl dimension if it exists)
                if upper_air_pred[ch_idx].ndim == 3:
                    pred_var = upper_air_pred[ch_idx].mean(axis=0)
                else:
                    pred_var = upper_air_pred[ch_idx]
                
                # Create comparison plot
                units = get_variable_units(var)
                title = f'Upper Air {var} - Sample {sample_idx} - Lead time: {lead_time_hours}h'
                save_path = os.path.join(output_dir, f'comparison_upper_{var}_sample{sample_idx}.png')
                
                plot_comparison(gt_data, pred_var, lats, lons, var, save_path,
                              cmap='viridis', diff_cmap='RdBu_r', title=title, 
                              units=units, lead_time_hours=lead_time_hours)
    
    print(f"\n✓ Comparison plots saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot variables from Zarr dataset or compare predictions with ground truth.")
    parser.add_argument("--mode", type=str, choices=['single', 'testset', 'compare'], default='single',
                       help="Plotting mode: 'single' (one timestamp), 'testset' (multiple test timestamps), 'compare' (predictions vs ground truth)")
    parser.add_argument("--data_path", type=str,
                       default="/home/bedartha/public/datasets/as_downloaded/weatherbench2/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr",
                       help="Path to Zarr dataset")
    parser.add_argument("--output_dir", type=str, 
                       default="/storage/arpit/Pangu/Pangu_Weather_Prediction_Model/pangu/visualization/",
                       help="Directory to save visualizations")
    parser.add_argument("--variable", type=str, help="Variable name to plot (for single/testset modes)")
    parser.add_argument("--cmap", type=str, default="viridis", help="Colormap to use")
    parser.add_argument("--time_idx", type=int, default=0, help="Time index to plot (for single mode)")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of test timestamps to visualize")
    
    # Arguments for comparison mode
    parser.add_argument("--prediction_dir", type=str, 
                       help="Directory containing .npz prediction files from test_clean.py (for compare mode)")
    parser.add_argument("--surface_variables", type=str, nargs='+', 
                       default=["2m_temperature", "mean_sea_level_pressure", 
                               "10m_u_component_of_wind", "10m_v_component_of_wind"],
                       help="List of surface variable names (for compare mode)")
    parser.add_argument("--upper_air_variables", type=str, nargs='+',
                       default=["geopotential", "specific_humidity", "temperature",
                               "u_component_of_wind", "v_component_of_wind"],
                       help="List of upper-air variable names (for compare mode)")
    parser.add_argument("--test_years", type=int, nargs='+', default=[2021, 2023],
                       help="Year range for test data (for compare mode)")
    parser.add_argument("--lead_time", type=int, default=24, 
                       help="Lead time in hours used in predictions (for compare mode)")
    parser.add_argument("--pressure_levels", type=int, nargs='+',
                       help="Specific pressure levels to plot (e.g., 500 850 1000). If not specified, all levels will be plotted.")

    args = parser.parse_args()

    if args.mode == 'compare':
        # Compare predictions with ground truth
        if not args.prediction_dir:
            print("Error: --prediction_dir is required for compare mode")
            sys.exit(1)
        
        plot_predictions_vs_ground_truth(
            prediction_dir=args.prediction_dir,
            zarr_path=args.data_path,
            output_dir=args.output_dir,
            surface_variables=args.surface_variables,
            upper_air_variables=args.upper_air_variables,
            test_years=args.test_years,
            lead_time_hours=args.lead_time,
            pressure_levels=args.pressure_levels
        )
    elif args.mode == 'testset':
        # Plot test set ground truth
        if not args.variable:
            print("Error: --variable is required for testset mode")
            sys.exit(1)
        plot_testset_visualizations(args.data_path, args.output_dir, args.variable, args.cmap, args.num_samples)
    else:
        # Plot single timestamp ground truth
        if not args.variable:
            print("Error: --variable is required for single mode")
            sys.exit(1)
        plot_variable_from_zarr(args.data_path, args.output_dir, args.variable, args.cmap, args.time_idx)
