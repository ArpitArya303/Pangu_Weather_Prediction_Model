import argparse
import xarray as xr
import numpy as np
import pickle
from tqdm import tqdm

def cal_mean_std(ds, variables, level_dim=None, pressure_levels=None):
    mean, std = {}, {}
    for v in tqdm(variables, desc="Variables"):
        data = ds[v]
        if level_dim and level_dim in data.dims:
            levels = pressure_levels if pressure_levels is not None else data[level_dim].values
            for lv in tqdm(levels, desc=f"{v} levels", leave=False):
                arr = data.sel({level_dim: lv}).values
                mean[(v, lv)] = np.nanmean(arr)
                std[(v, lv)] = np.nanstd(arr)
        else:
            arr = data.values
            mean[v] = np.nanmean(arr)
            std[v] = np.nanstd(arr)
    return mean, std

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate mean and std for weather variables")
    parser.add_argument("--zarr_path", type=str, default="/home/bedartha/public/datasets/as_downloaded/weatherbench2/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr", help="Path to the Zarr dataset")
    parser.add_argument("--surface_variables", nargs='+', default=["2m_temperature","mean_sea_level_pressure","10m_u_component_of_wind","10m_v_component_of_wind"], help="Surface variables")
    parser.add_argument("--upper_air_variables", nargs='+', default=["geopotential","specific_humidity","temperature", "u_component_of_wind", "v_component_of_wind"], help="Upper air variables")
    parser.add_argument("--plevels", nargs='+', type=int, default=[1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100,50], help="Pressure levels for upper air variables")
    print("Parsing arguments...")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save the mean and std pickle files")
    opt = parser.parse_args()

    print("Loading dataset...")
    ds = xr.open_zarr(opt.zarr_path)

    print("Calculating mean and std for surface variables...")
    surface_mean, surface_std = cal_mean_std(ds, opt.surface_variables)
    surface_mean_dir = f"{opt.output_dir}/surface_mean.pkl"
    surface_std_dir = f"{opt.output_dir}/surface_std.pkl"
    with open(surface_mean_dir, "wb") as f: pickle.dump(surface_mean, f)
    with open(surface_std_dir, "wb") as f: pickle.dump(surface_std, f)

    print("Calculating mean and std for upper air variables...")
    upper_air_mean, upper_air_std = cal_mean_std(ds, opt.upper_air_variables, level_dim="level", pressure_levels=opt.plevels)
    upper_air_mean_dir = f"{opt.output_dir}/upper_air_mean.pkl"
    upper_air_std_dir = f"{opt.output_dir}/upper_air_std.pkl"
    with open(upper_air_mean_dir, "wb") as f: pickle.dump(upper_air_mean, f)
    with open(upper_air_std_dir, "wb") as f: pickle.dump(upper_air_std, f)