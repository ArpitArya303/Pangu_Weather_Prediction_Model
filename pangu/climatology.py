#!/usr/bin/env python3
# File: create_hourly_climatology.py
import argparse
import xarray as xr
import numpy as np
import os
from dask.diagnostics import ProgressBar

def main():
    parser = argparse.ArgumentParser(description="Calculate 6-Hourly Climatology (1979-2017).")
    parser.add_argument("--zarr_path", type=str, required=True, 
                        help="Path to the full Zarr dataset.")
    parser.add_argument("--output_path", type=str, required=True, 
                        help="Path to save the Climatology (e.g., clim_6hourly.nc).")
    
    args = parser.parse_args()

    print(f"📂 Loading dataset: {args.zarr_path}")
    ds = xr.open_zarr(args.zarr_path)

    # Pangu Training Period
    print("📅 Selecting training period: 1979-2017...")
    ds_train = ds.sel(time=slice("1979-01-01", "2017-12-31"))

    # Define the 6-hour intervals
    hours = [0, 6, 12, 18]
    clim_list = []

    print(f"⚙️ Calculating climatology for hours: {hours}")
    
    for h in hours:
        print(f"   Processing {h:02d}:00 UTC...")
        # 1. Select only timestamps for this specific hour
        #    (e.g., only 12:00 timestamps from 1979-2017)
        ds_hourly = ds_train.sel(time=ds_train.time.dt.hour == h)
        
        # 2. Group by Day of Year and Average
        #    This gives the average "12:00 on Nov 13" across 39 years
        with ProgressBar():
            clim = ds_hourly.groupby("time.dayofyear").mean("time")
            
        # 3. Add the hour as a dimension so we can combine them later
        clim = clim.expand_dims(hour=[h])
        clim_list.append(clim)

    # Combine into one file with dimensions (hour: 4, dayofyear: 366, lat, lon)
    print("🔗 Combining hourly climatologies...")
    full_clim = xr.concat(clim_list, dim="hour")

    # Save
    print(f"💾 Saving to {args.output_path}...")
    os.makedirs(args.output_path, exist_ok=True)
    output_file = os.path.join(args.output_path, "clim_6hourly.nc")
    full_clim.to_netcdf(output_file)
    print("✅ 6-Hourly Climatology generated!")

if __name__ == "__main__":
    main()