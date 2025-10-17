import xarray as xr
import numpy as np
import pickle
from tqdm import tqdm

def cal_mean_std(ds, variables, level_dim=None):
    mean, std = {}, {}
    for v in tqdm(variables, desc="Variables"):
        data = ds[v]
        if level_dim and level_dim in data.dims:
            for lv in tqdm(ds[level_dim].values, desc=f"{v} levels", leave=False):
                arr = data.sel({level_dim: lv}).values
                mean[(v, lv)] = np.nanmean(arr)
                std[(v, lv)] = np.nanstd(arr)
        else:
            arr = data.values
            mean[v] = np.nanmean(arr)
            std[v] = np.nanstd(arr)
    return mean, std

if __name__ == "__main__":
    zarr_path = "/path/to/your.zarr"  
    ds = xr.open_zarr(zarr_path, consolidated=True)

    surface_vars = ["u10", "v10", "t2m", "msl"]  # update as needed
    upper_air_vars = ['z', 'q', 't', 'u', 'v']   # update as needed

    surface_mean, surface_std = cal_mean_std(ds, surface_vars)
    with open("surface_mean.pkl", "wb") as f: pickle.dump(surface_mean, f)
    with open("surface_std.pkl", "wb") as f: pickle.dump(surface_std, f)

    upper_air_mean, upper_air_std = cal_mean_std(ds, upper_air_vars, level_dim="level")
    with open("upper_air_mean.pkl", "wb") as f: pickle.dump(upper_air_mean, f)
    with open("upper_air_std.pkl", "wb") as f: pickle.dump(upper_air_std, f)