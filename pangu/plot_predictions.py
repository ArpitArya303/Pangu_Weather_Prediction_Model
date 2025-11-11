#!/usr/bin/env python3
# File: plot_predictions.py - Visualize Pangu predictions (surface + upper air)
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point

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

def main(args):
    # Load metadata (surface_vars, upper_air_vars, pressure_levels)
    meta = _load_metadata(args.prediction_dir)
    if meta:
        print(f"📄 Loaded metadata.npz from {args.prediction_dir}")
        meta_surface = meta.get("surface_vars")
        meta_upper = meta.get("upper_air_vars")
        meta_plevs = meta.get("pressure_levels")
    else:
        print("ℹ️ metadata.npz not found; falling back to CLI lists.")
        meta_surface = None
        meta_upper = None
        meta_plevs = None

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
        lats = np.linspace(-90, 90, n_lat)
        lons = np.linspace(0, 360, n_lon, endpoint=False)

        print(f"\n📊 Plotting sample {sample_idx} ({file})")
        print(f"  Grid: {n_lat}×{n_lon} | Levels in file: {p_levels.tolist()}")

        # Surface: map variable names to channel indices using metadata if available
        for var_name in surface_vars:
            ch_idx = _idx_by_name(var_name, meta_surface) if meta_surface else None
            if ch_idx is None:
                # Fallback: try positional match if within range
                try_pos = surface_vars.index(var_name)
                if try_pos >= surface.shape[0]:
                    print(f"  ⚠️ Skipping surface '{var_name}': no matching channel")
                    continue
                ch_idx = try_pos
            pred_field = surface[ch_idx, :, :]
            plot_surface(pred_field, var_name, lats, lons, args.cmap,
                         os.path.join(args.output_dir, "surface"), sample_idx)

        # Upper-air: map names via metadata if available
        for var_name in upper_vars:
            ch_idx = _idx_by_name(var_name, meta_upper) if meta_upper else None
            if ch_idx is None:
                try_pos = upper_vars.index(var_name)
                if try_pos >= upper_air.shape[0]:
                    print(f"  ⚠️ Skipping upper-air '{var_name}': no matching channel")
                    continue
                ch_idx = try_pos
            plot_upper_air(upper_air[ch_idx, :, :, :], var_name, lats, lons,
                           p_levels, requested_levels, args.cmap,
                           os.path.join(args.output_dir, "upper_air"), sample_idx)

    print(f"\n✅ All plots saved under: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Pangu model prediction outputs (.npz files)")
    parser.add_argument("--prediction_dir", type=str, required=True,
                        help="Directory containing prediction .npz files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save plots")
    # If not provided, script will use metadata.npz lists
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
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    main(args)