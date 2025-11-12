#!/usr/bin/env python3
# File: plot_predictions.py - Visualize Pangu predictions (surface + upper air)
import os
import argparse
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point

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

def _load_metadata(prediction_dir: str):
    """Load metadata.npz if present."""
    meta_path = os.path.join(prediction_dir, "metadata.npz")
    if not os.path.exists(meta_path):
        return None
    m = np.load(meta_path, allow_pickle=True)
    meta = {
        "surface_vars": list(m["surface_vars"].tolist()) if "surface_vars" in m else None,
        "upper_air_vars": list(m["upper_air_vars"].tolist()) if "upper_air_vars" in m else None,
        "pressure_levels": m["pressure_levels"] if "pressure_levels" in m else None,
        "lead_time_hours": int(m["lead_time_hours"]) if "lead_time_hours" in m else None,
    }
    return meta

def _idx_by_name(name: str, name_list):
    """Return index of name in name_list or None."""
    if not name_list:
        return None
    try:
        return name_list.index(name)
    except ValueError:
        return None

def plot_surface(var_data, var_name, lats, lons, cmap, output_dir, sample_idx):
    """Plot 2D surface field with cyclic point."""
    os.makedirs(output_dir, exist_ok=True)
    data_cyclic, lons_cyclic = add_cyclic_point(var_data, coord=lons)
    lon_grid, lat_grid = np.meshgrid(lons_cyclic, lats)

    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(10, 6))
    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)

    mesh = ax.contourf(lon_grid, lat_grid, data_cyclic,
                       transform=ccrs.PlateCarree(), cmap=cmap,
                       levels=20, extend='both')
    plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.05, label=var_name)
    plt.title(f"Surface {var_name} (sample {sample_idx})")
    plt.tight_layout()

    outfile = os.path.join(output_dir, f"{var_name}_sample{sample_idx}.png")
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Saved surface plot: {outfile}")

def plot_upper_air(var_data, var_name, lats, lons, p_levels, requested_levels,
                   cmap, output_dir, sample_idx):
    """Plot 3D upper-air variable for specified pressure levels."""
    os.makedirs(output_dir, exist_ok=True)

    # Choose which levels to plot
    if requested_levels is None:
        levels_to_plot = p_levels
    else:
        # Only keep requested levels that exist in this file
        req = np.array(requested_levels)
        levels_to_plot = [int(p) for p in req if int(p) in set(p_levels.tolist())]

    for p in levels_to_plot:
        # Find index of this pressure level
        matches = np.where(p_levels == p)[0]
        if len(matches) == 0:
            print(f"  ⚠️ Skipping {var_name} @ {p} hPa - not in file levels {p_levels}")
            continue
        p_idx = matches[0]

        field = var_data[p_idx, :, :]  # (Lat, Lon)
        field_cyclic, lons_cyclic = add_cyclic_point(field, coord=lons)
        lon_grid, lat_grid = np.meshgrid(lons_cyclic, lats)

        fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(10, 6))
        ax.set_global()
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)

        mesh = ax.contourf(lon_grid, lat_grid, field_cyclic,
                           transform=ccrs.PlateCarree(), cmap=cmap,
                           levels=20, extend='both')
        plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.05, label=var_name)
        plt.title(f"{var_name} @ {p} hPa (sample {sample_idx})")
        plt.tight_layout()

        outfile = os.path.join(output_dir, f"{var_name}_{p}hPa_sample{sample_idx}.png")
        plt.savefig(outfile, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Saved upper-air plot: {outfile}")

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
    ax2.set_title(f'Prediction (t+{lead_time_hours}h)', fontsize=11, fontweight='bold')
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
    print(f"✅ Comparison plot saved: {output_path}")

def main(args):
    # Load the dataset
    print(f"📂 Loading ground truth from {args.zarr_path}...")
    ds = xr.open_zarr(args.zarr_path, consolidated=True)
    
    # Get dataset coordinates
    lats_ds = ds['latitude'].values
    lons_ds = ds['longitude'].values
    
    # Filter test times
    test_years = args.test_years
    test_times = ds['time'].sel(time=ds['time'].dt.year >= test_years[0])
    if test_times.size == 0:
        print(f"❌ No test set timestamps (>={test_years[0]}) found.")
        return
    print(f"✅ Found {len(test_times)} test timestamps")

    # Load metadata (surface_vars, upper_air_vars, pressure_levels)
    meta = _load_metadata(args.prediction_dir)
    if meta:
        print(f"📄 Loaded metadata.npz from {args.prediction_dir}")
        meta_surface = meta.get("surface_vars")
        meta_upper = meta.get("upper_air_vars")
        meta_plevs = meta.get("pressure_levels")
        lead_time_hours = meta.get("lead_time_hours", 6)
    else:
        print("ℹ️ metadata.npz not found; falling back to CLI lists.")
        meta_surface = None
        meta_upper = None
        meta_plevs = None
        lead_time_hours = args.lead_time_hours

    # Decide which variables to plot
    surface_vars = args.surface_variables if args.surface_variables else meta_surface
    upper_vars = args.upper_air_variables if args.upper_air_variables else meta_upper
    if surface_vars is None:
        surface_vars = []
    if upper_vars is None:
        upper_vars = []

    # Decide which pressure levels to plot (requested set)
    requested_levels = args.pressure_levels if args.pressure_levels else (
        meta_plevs.tolist() if meta_plevs is not None else None
    )

    # Files to plot: skip metadata.npz
    if not os.path.exists(args.prediction_dir):
        raise FileNotFoundError(f"Prediction directory not found: {args.prediction_dir}")

    all_npz = sorted([f for f in os.listdir(args.prediction_dir)
                      if f.endswith(".npz") and f != "metadata.npz"])

    if args.sample_limit:
        all_npz = all_npz[:args.sample_limit]

    if len(all_npz) == 0:
        print(f"⚠️ No prediction .npz files found in {args.prediction_dir}")
        return

    print(f"📂 Found {len(all_npz)} prediction files in {args.prediction_dir}")
    print(f"🧭 Surface vars: {surface_vars}")
    print(f"🧭 Upper-air vars: {upper_vars}")
    print(f"🧭 Lead time: {lead_time_hours}h")
    if requested_levels is None:
        print("🧭 Levels to plot: all available")
    else:
        print(f"🧭 Levels to plot: {requested_levels}")

    for file in all_npz:
        fpath = os.path.join(args.prediction_dir, file)
        try:
            data = np.load(fpath, allow_pickle=True)
        except Exception as e:
            print(f"⚠️ Cannot open {file}: {e}")
            continue

        keys = set(data.files)
        # Required prediction payload keys (any accepted synonym)
        surface_key_opts = [k for k in ("surface", "surface_pred", "surf") if k in keys]
        upper_key_opts   = [k for k in ("upper_air", "upper_air_pred", "upper") if k in keys]
        plevel_key_opts  = [k for k in ("p_levels", "pressure_levels", "plevs") if k in keys]

        if not surface_key_opts or not upper_key_opts or not plevel_key_opts:
            print(f"ℹ️ Skipping {file}: missing required keys. Found keys: {sorted(keys)}")
            continue

        surface_key = surface_key_opts[0]
        upper_key   = upper_key_opts[0]
        plevel_key  = plevel_key_opts[0]

        surface = data[surface_key]        # (C, Lat, Lon)
        upper_air = data[upper_key]        # (C, Pl, Lat, Lon)
        p_levels = data[plevel_key]        # (Pl,)
        sample_idx = int(data["sample_idx"]) if "sample_idx" in keys else -1

        # Grid (assume equiangular)
        n_lat = surface.shape[1]
        n_lon = surface.shape[2]
        lats = np.linspace(90, -90, n_lat)  # North to South
        lons = np.linspace(0, 360, n_lon, endpoint=False)

        print(f"\n{'='*60}")
        print(f"📊 Processing sample {sample_idx} ({file})")
        print(f"  Grid: {n_lat}×{n_lon} | Levels in file: {p_levels.tolist()}")

        # Calculate the target time index for ground truth
        forecast_steps = lead_time_hours // 6  # Number of 6-hour steps
        target_time_idx = sample_idx + forecast_steps
        
        if target_time_idx >= len(test_times):
            print(f"⚠️ Target time index {target_time_idx} exceeds test set size, skipping...")
            continue
        
        # Get ground truth time
        target_time = test_times[target_time_idx]
        time_idx_in_full_dataset = int(ds['time'].get_index('time').get_loc(target_time.values))
        
        print(f"  Sample {sample_idx}: Forecast time index = {target_time_idx}")
        print(f"  Target time: {str(target_time.values)}")
        
        # === SURFACE VARIABLES ===
        print(f"\n📊 Plotting {len(surface_vars)} surface variables...")
        for var_name in surface_vars:
            ch_idx = _idx_by_name(var_name, meta_surface) if meta_surface else None
            if ch_idx is None:
                # Fallback: try positional match if within range
                try_pos = surface_vars.index(var_name)
                if try_pos >= surface.shape[0]:
                    print(f"  ⚠️ Skipping surface '{var_name}': no matching channel")
                    continue
                ch_idx = try_pos
            
            if var_name not in ds.variables:
                print(f"  ⚠️ Variable '{var_name}' not found in dataset, skipping...")
                continue
            
            pred_field = surface[ch_idx, :, :]

            # Get ground truth
            gt_data_xr = ds[var_name].isel(time=time_idx_in_full_dataset)
            
            # Squeeze potential singleton dimensions (e.g., height=1)
            # This ensures the GT data is 2D, just like the prediction.
            level_dims_to_squeeze = [d for d in gt_data_xr.dims if d.lower() in ['level', 'height', 'pressure_level', 'plev']]
            if level_dims_to_squeeze:
                for dim in level_dims_to_squeeze:
                    if gt_data_xr.sizes[dim] == 1:
                        print(f"    Squeezing singleton dimension '{dim}' from GT.")
                        gt_data_xr = gt_data_xr.squeeze(dim)
                        
            # Check the dimensions of the ground truth data
            print(f"    GT dimensions before processing: {gt_data_xr.dims}")
            print(f"    GT shape before processing: {gt_data_xr.shape}")
            
            gt_data = gt_data_xr.values

            # Transpose if needed (data should be Lat x Lon)
            if gt_data.shape[0] == len(lons_ds) and gt_data.shape[1] == len(lats_ds):
                print(f"    Transposing GT from (Lon={gt_data.shape[0]}, Lat={gt_data.shape[1]}) to (Lat, Lon)")
                gt_data = gt_data.T
            elif gt_data.shape[0] == len(lats_ds) and gt_data.shape[1] == len(lons_ds):
                print(f"    GT already in (Lat={gt_data.shape[0]}, Lon={gt_data.shape[1]}) format")
            else:
                print(f"  ⚠️ GT shape {gt_data.shape} doesn't match expected grid ({len(lats_ds)}, {len(lons_ds)}), skipping...")
                continue
            
            print(f"    GT shape after processing: {gt_data.shape}")
            
            
            # Validate shapes
            if gt_data.shape != pred_field.shape:
                print(f"  ⚠️ Shape mismatch for {var_name}: GT={gt_data.shape}, Pred={pred_field.shape}, skipping...")
                continue

            # Print some statistics for debugging
            print(f"    GT range: [{gt_data.min():.2f}, {gt_data.max():.2f}]")
            print(f"    Pred range: [{pred_field.min():.2f}, {pred_field.max():.2f}]")
            
            # Plot comparison
            units = get_variable_units(var_name)
            comparison_output_path = os.path.join(args.output_dir, "surface",
                                                  f"{var_name}_sample{sample_idx}_comparison.png")
            os.makedirs(os.path.dirname(comparison_output_path), exist_ok=True)
            title = f"Surface {var_name} - Sample {sample_idx}"
            
            plot_comparison(gt_data, pred_field, lats, lons, var_name, comparison_output_path,
                            cmap=args.cmap, diff_cmap='RdBu_r',
                            title=title, units=units,
                            lead_time_hours=lead_time_hours)

        # === UPPER-AIR VARIABLES ===
        print(f"\n📊 Plotting {len(upper_vars)} upper-air variables...")
        for var_name in upper_vars:
            ch_idx = _idx_by_name(var_name, meta_upper) if meta_upper else None
            if ch_idx is None:
                try_pos = upper_vars.index(var_name)
                if try_pos >= upper_air.shape[0]:
                    print(f"  ⚠️ Skipping upper-air '{var_name}': no matching channel")
                    continue
                ch_idx = try_pos
            
            if var_name not in ds.variables:
                print(f"  ⚠️ Variable '{var_name}' not found in dataset, skipping...")
                continue
            
            # Determine which pressure levels to plot
            if requested_levels is None:
                levels_to_plot = p_levels.tolist()
            else:
                levels_to_plot = [int(p) for p in requested_levels if int(p) in set(p_levels.tolist())]
            
            print(f"  Variable: {var_name}, Levels to plot: {levels_to_plot}")
            
            # Get ground truth for all levels
            level_dim = 'level' if 'level' in ds[var_name].dims else 'pressure_level'
            gt_data_all_levels = ds[var_name].isel(time=time_idx_in_full_dataset)
            available_levels = ds[level_dim].values
            
            # Plot each pressure level
            for pressure_level in levels_to_plot:
                # Find ground truth level index
                gt_matches = np.where(np.isclose(available_levels.astype(float), 
                                                 float(pressure_level), atol=1.0))[0]
                if gt_matches.size == 0:
                    print(f"    ⚠️ GT doesn't have level {pressure_level} hPa, skipping...")
                    continue
                gt_pl_idx = int(gt_matches[0])
                
                # Find prediction level index
                pred_matches = np.where(np.isclose(p_levels.astype(float), 
                                                   float(pressure_level), atol=1.0))[0]
                if pred_matches.size == 0:
                    print(f"    ⚠️ Predictions don't have level {pressure_level} hPa, skipping...")
                    continue
                pred_pl_idx = int(pred_matches[0])
                
                # Get ground truth at this level
                gt_data = gt_data_all_levels.isel({level_dim: gt_pl_idx}).values
                
                # Transpose if needed
                if gt_data.shape[0] == len(lons_ds) and gt_data.shape[1] == len(lats_ds):
                    gt_data = gt_data.T
                
                # Get prediction at this level
                pred_field = upper_air[ch_idx, pred_pl_idx, :, :]
                
                # Validate shapes
                if gt_data.shape != pred_field.shape:
                    print(f"    ⚠️ Shape mismatch for {var_name} @ {pressure_level}hPa: GT={gt_data.shape}, Pred={pred_field.shape}, skipping...")
                    continue
                
                # Plot comparison
                units = get_variable_units(var_name)
                pl_int = int(round(float(pressure_level)))
                comparison_output_path = os.path.join(args.output_dir, "upper_air", var_name,
                                                      f"{var_name}_{pl_int}hPa_sample{sample_idx}_comparison.png")
                os.makedirs(os.path.dirname(comparison_output_path), exist_ok=True)
                title = f"{var_name} @ {pl_int}hPa - Sample {sample_idx}"
                
                plot_comparison(gt_data, pred_field, lats, lons, var_name, comparison_output_path,
                              cmap=args.cmap, diff_cmap='RdBu_r',
                              title=title, units=units,
                              lead_time_hours=lead_time_hours)

    print(f"\n{'='*60}")
    print(f"✅ All plots saved under: {args.output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Pangu model prediction outputs (.npz files)")
    parser.add_argument("--zarr_path", type=str, required=True,
                       help="Path to ground truth Zarr dataset")
    parser.add_argument("--prediction_dir", type=str, required=True,
                        help="Directory containing prediction .npz files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save plots")
    parser.add_argument("--surface_variables", nargs="+", default=None,
                        help="Surface variable names to plot; default: metadata surface_vars")
    parser.add_argument("--upper_air_variables", nargs="+", default=None,
                        help="Upper-air variable names to plot; default: metadata upper_air_vars")
    parser.add_argument("--cmap", type=str, default="viridis",
                        help="Matplotlib colormap")
    parser.add_argument("--pressure_levels", nargs="+", type=int, default=None,
                        help="Pressure levels to plot (e.g., 500 850 1000). "
                             "Default: metadata pressure_levels (all)")
    parser.add_argument("--sample_limit", type=int, default=None,
                        help="Limit number of samples to plot")
    parser.add_argument("--test_years", type=int, nargs='+', default=[2021, 2023],
                       help="Year range for test data (default: 2021 2023)")
    parser.add_argument("--lead_time_hours", type=int, default=6,
                       help="Lead time in hours (default: 6, used if metadata missing)")
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    main(args)