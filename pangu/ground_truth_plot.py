#!/usr/bin/env python3
# file: plot_all_zarr_variables.py
import argparse
import xarray as xr
import numpy as np
import matplotlib
matplotlib.use("Agg")  # for headless HPC environments
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
import os


def detect_lat_lon(da):
    lat = [n for n in da.dims if "lat" in n.lower()][0]
    lon = [n for n in da.dims if "lon" in n.lower()][0]
    return lat, lon


def plot_field(da, var_name, lats, lons, cmap, level_label, output_file):
    """Plot a single 2D field using contourf with cyclic point."""
    
    # Get data values and ensure 2D
    data = da.values
    
    # Check if data needs transposing
    if data.shape == (len(lons), len(lats)):
        data = data.T  # Transpose from (lon, lat) to (lat, lon)
        print(f"  Transposed {var_name} from (lon, lat) to (lat, lon)")
    
    # Validate shape
    if data.ndim != 2:
        raise ValueError(f"Data must be 2D, got shape {data.shape}")
    
    if data.shape != (len(lats), len(lons)):
        raise ValueError(f"Data shape {data.shape} doesn't match expected ({len(lats)}, {len(lons)})")
    
    # Add cyclic point to avoid white line at longitude wrap
    data_cyclic, lons_cyclic = add_cyclic_point(data, coord=lons)
    print(f"  Added cyclic point: data shape {data.shape} -> {data_cyclic.shape}")
    
    # Create meshgrid with cyclic coordinates
    lon_grid, lat_grid = np.meshgrid(lons_cyclic, lats)
    
    # Create figure
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(10, 6))
    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    
    # Use contourf with cyclic data
    mesh = ax.contourf(lon_grid, lat_grid, data_cyclic, 
                      transform=ccrs.PlateCarree(), 
                      cmap=cmap, levels=20, extend='both')
    
    # Add colorbar
    units = da.attrs.get('units', '')
    plt.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.05, 
                label=f"{var_name} ({units})")
    plt.title(f"{var_name} {level_label}", fontsize=12)
    plt.tight_layout()
    
    # Save
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Saved: {output_file}")


def plot_variable_from_zarr(zarr_path, var_name, cmap, time_index, output_dir, pressure_levels=None):
    """Plot a variable (surface or specific pressure levels if 3D)."""
    ds = xr.open_zarr(zarr_path)
    
    if var_name not in ds.variables:
        raise KeyError(f"Variable '{var_name}' not found. Available: {list(ds.variables)}")
    
    da = ds[var_name]
    
    print(f"\n📊 Processing {var_name}...")
    print(f"  Original shape: {da.shape}, dims: {da.dims}")

    # Select time
    if "time" in da.dims:
        da = da.isel(time=time_index)
        print(f"  After time selection: {da.shape}, dims: {da.dims}")

    # Get lat/lon
    lat_name, lon_name = detect_lat_lon(da)
    lats = da[lat_name].values
    lons = da[lon_name].values
    print(f"  Grid: {len(lats)} lats × {len(lons)} lons")

    # Detect pressure/height dimension
    level_dims = [d for d in da.dims if any(k in d.lower() for k in ["level", "plev", "pressure", "height"])]

    if level_dims:
        # Upper-air variable: plot specific or all levels
        level_dim = level_dims[0]
        available_levels = da[level_dim].values
        num_levels = len(available_levels)
        print(f"  Found {num_levels} levels in '{level_dim}': {available_levels}")
        
        # Filter levels if specific ones requested
        if pressure_levels is not None:
            levels_to_plot = [lvl for lvl in pressure_levels if lvl in available_levels]
            if not levels_to_plot:
                print(f"  ⚠️ Warning: None of the requested levels {pressure_levels} found in available levels")
                return
            print(f"  Plotting {len(levels_to_plot)} requested levels: {levels_to_plot}")
        else:
            levels_to_plot = available_levels
            print(f"  Plotting all {num_levels} levels")
        
        for i, level in enumerate(levels_to_plot):
            da_level = da.sel({level_dim: level})
            print(f"  [{i+1}/{len(levels_to_plot)}] Plotting {level_dim}={float(level):.0f}, shape: {da_level.shape}")
            
            # Verify it's 2D
            if da_level.ndim != 2:
                print(f"    ⚠️ Skipping - not 2D after level selection (shape: {da_level.shape})")
                continue
            
            level_label = f"{level_dim}={float(level):.0f}"
            subdir = os.path.join(output_dir, "upper_air", var_name)
            os.makedirs(subdir, exist_ok=True)
            output_file = os.path.join(subdir, f"{var_name}_{level_dim}_{int(level)}.png")
            plot_field(da_level, var_name, lats, lons, cmap, level_label, output_file)
    else:
        # Surface variable: plot once
        print(f"  Surface variable (no vertical levels)")
        if da.ndim != 2:
            print(f"    ⚠️ Warning: Expected 2D, got {da.ndim}D with shape {da.shape}")
            if da.ndim > 2:
                print(f"    Remaining dims after time selection: {da.dims}")
                raise ValueError(f"Surface variable has unexpected dimensions: {da.dims}")
        
        subdir = os.path.join(output_dir, "surface")
        os.makedirs(subdir, exist_ok=True)
        output_file = os.path.join(subdir, f"{var_name}.png")
        plot_field(da, var_name, lats, lons, cmap, "surface", output_file)


def main(args):
    """Plot specified or all variables in the Zarr dataset."""
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    print(f"📂 Loading dataset: {args.zarr_path}")
    ds = xr.open_zarr(args.zarr_path)
    
    # Determine which variables to plot
    if args.surface_variables or args.upper_air_variables:
        # Use specified variables
        surface_vars = args.surface_variables if args.surface_variables else []
        upper_vars = args.upper_air_variables if args.upper_air_variables else []
        var_list = surface_vars + upper_vars
        print(f"\n📊 Plotting {len(var_list)} specified variables:")
        print(f"  Surface: {surface_vars}")
        print(f"  Upper-air: {upper_vars}")
    else:
        # Plot all variables
        var_list = list(ds.data_vars)
        surface_vars = []
        upper_vars = []
        print(f"\n📊 Plotting all {len(var_list)} variables: {var_list}")

    # Plot each variable
    for var in var_list:
        try:
            # Only pass pressure_levels for upper-air variables
            is_upper = var in upper_vars if upper_vars else None
            plevels = args.pressure_levels if (is_upper or is_upper is None) else None
            
            plot_variable_from_zarr(
                args.zarr_path, 
                var, 
                args.colormap, 
                args.time_index, 
                args.output_dir,
                pressure_levels=plevels
            )
        except Exception as e:
            print(f"⚠️ Error plotting {var}: {e}\n")
            import traceback
            traceback.print_exc()

    print(f"\n✅ All plots saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot variables from a Zarr dataset.")
    parser.add_argument("--zarr_path", type=str, required=True, 
                       help="Path to the Zarr dataset.")
    parser.add_argument("--output_dir", type=str, required=True, 
                       help="Directory to save the plots.")
    parser.add_argument("--surface_variables", nargs='+', default=None,
                       help="Surface variables to plot (e.g., 2m_temperature mean_sea_level_pressure)")
    parser.add_argument("--upper_air_variables", nargs='+', default=None,
                       help="Upper-air variables to plot (e.g., geopotential temperature)")
    parser.add_argument("--pressure_levels", nargs='+', type=int, default=None,
                       help="Pressure levels to plot for upper-air variables (e.g., 500 850 1000)")
    parser.add_argument("--colormap", type=str, default="viridis", 
                       help="Colormap for plotting.")
    parser.add_argument("--time_index", type=int, default=0, 
                       help="Time index to plot.")
    
    args = parser.parse_args()
    main(args)