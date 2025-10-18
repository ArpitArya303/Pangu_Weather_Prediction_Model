import pandas as pd

path = "/storage/arpit/Pangu/Pangu_Weather_Prediction_Model/pangu/data"
surface_mean = pd.read_pickle(f"{path}/surface_mean.pkl")
surface_std = pd.read_pickle(f"{path}/surface_std.pkl")
upper_air_mean = pd.read_pickle(f"{path}/upper_air_mean.pkl")
upper_air_std = pd.read_pickle(f"{path}/upper_air_std.pkl")

print("Surface Mean:")
for k, v in surface_mean.items():
    print(f"{k}: {v}")  

print("\nSurface Std:")
for k, v in surface_std.items():
    print(f"{k}: {v}")

print("\nUpper Air Mean:")
for k, v in upper_air_mean.items():
    print(f"{k}: {v}")

print("\nUpper Air Std:")
for k, v in upper_air_std.items():
    print(f"{k}: {v}")

