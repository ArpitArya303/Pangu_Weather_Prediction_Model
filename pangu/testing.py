import time
from torch.utils.data import DataLoader
import os
from Pangu.Pangu_Weather_Prediction_Model.pangu.pangu_model import Pangu_lite
from Pangu.Pangu_Weather_Prediction_Model.pangu.data_utils import (
    ZarrWeatherDataset, 
    surface_transform, 
    upper_air_transform, 
)

t0 = time.time()
surface_normalizer, _ = surface_transform(
    os.path.join("/storage/arpit/Pangu/Pangu_Weather_Prediction_Model/pangu/data", "surface_mean.pkl"),
    os.path.join("/storage/arpit/Pangu/Pangu_Weather_Prediction_Model/pangu/data", "surface_std.pkl")
    )

upper_air_normalizer, _, _ = upper_air_transform(
    os.path.join("/storage/arpit/Pangu/Pangu_Weather_Prediction_Model/pangu/data", "upper_air_mean.pkl"),
    os.path.join("/storage/arpit/Pangu/Pangu_Weather_Prediction_Model/pangu/data", "upper_air_std.pkl")
    )
# Train dataset creation
dataset = ZarrWeatherDataset(
        zarr_path="/home/bedartha/public/datasets/as_downloaded/weatherbench2/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr", 
        surface_vars= ["2m_temperature", "mean_sea_level_pressure"],
        upper_air_vars=["geopotential", "temperature"],
        plevels=[100, 200, 300],
        static_vars=["land_sea_mask", "soil_type"],
        surface_transform=surface_normalizer,  
        upper_air_transform=upper_air_normalizer  
    )

dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

it = iter(dataloader)
t1= time.time()
batch = next(it)
t2 = time.time()
print("single batch load (num_workers=6, pin_memory=True):", t2 - t1, "s")
print("Dataset and DataLoader preparation time:", t1 - t0, "s")
