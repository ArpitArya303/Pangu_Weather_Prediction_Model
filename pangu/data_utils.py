from os import listdir
from os.path import join
import pickle
import pandas as pd
import xarray as xr
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize, Compose

def surface_transform(mean_path, std_path):
    with open(mean_path, "rb") as f:
        surface_mean = pickle.load(f)

    with open(std_path, "rb") as f:
        surface_std = pickle.load(f)

    mean_seq, std_seq, channel_seq = [], [], []
    variables = sorted(list(surface_mean.keys()))
    for v in variables:
        channel_seq.append(v)
        mean_seq.append(surface_mean[v])
        std_seq.append(surface_std[v])

    return Normalize(mean_seq, std_seq), channel_seq


def upper_air_transform(mean_path, std_path):
    with open(mean_path, "rb") as f:
        upper_air_mean = pickle.load(f)

    with open(std_path, "rb") as f:
        upper_air_std = pickle.load(f)

    variables = set()
    pLevels = set()
    
    # Extract variables and levels from tuple keys
    for (var, level) in upper_air_mean.keys():
        variables.add(var)
        pLevels.add(level)
    
    variables = sorted(list(variables))
    pLevels = sorted(list(pLevels))

    normalize = {}

    for pl in pLevels:
        mean_seq = [upper_air_mean[(v, pl)] for v in variables]
        std_seq = [upper_air_std[(v, pl)] for v in variables]
        normalize[pl] = Normalize(mean_seq, std_seq)

    return normalize, variables, pLevels

def surface_inv_transform(mean_path, std_path):
    with open(mean_path, "rb") as f:
        surface_mean = pickle.load(f)

    with open(std_path, "rb") as f:
        surface_std = pickle.load(f)

    mean_seq, std_seq, channel_seq = [], [], []
    variables = sorted(list(surface_mean.keys()))
    for v in variables:
        channel_seq.append(v)
        mean_seq.append(surface_mean[v])
        std_seq.append(surface_std[v])

    invTrans = Compose([
        Normalize([0.] * len(mean_seq), [1 / x for x in std_seq]), 
        Normalize([-x for x in mean_seq], [1.] * len(std_seq))
    ])
    return invTrans, channel_seq


def upper_air_inv_transform(mean_path, std_path):
    with open(mean_path, "rb") as f:
        upper_air_mean = pickle.load(f)

    with open(std_path, "rb") as f:
        upper_air_std = pickle.load(f)

    pLevels = sorted(list(upper_air_mean.keys()))
    variables = sorted(list(list(upper_air_mean.values())[0].keys()))
    normalize = {}
    for pl in pLevels:
        mean_seq, std_seq = [], []
        for v in variables:
            mean_seq.append(upper_air_mean[pl][v])
            std_seq.append(upper_air_std[pl][v])

        invTrans = Compose([
            Normalize([0.] * len(mean_seq), [1 / x for x in std_seq]), 
            Normalize([-x for x in mean_seq], [1.] * len(std_seq))
        ])
        normalize[pl] = invTrans

    return normalize, variables, pLevels


# class DatasetFromFolder(Dataset):
#     def __init__(self, dataset_dir, flag: Literal["train", "test", "valid"]):
#         super().__init__()
#         self.dataset_dir = dataset_dir
#         self.flag = flag
#         self.surface_filenames = [join(dataset_dir, flag, x) for x in listdir(join(dataset_dir, flag)) if x.startswith("surface")]
#         self.upper_air_filenames = [join(dataset_dir, flag, f"upper_air_{y}_{str(m).zfill(2)}.nc") 
#                                     for y, m in [x.split(".")[-2].split("_")[-2:] for x in self.surface_filenames]]
#         self.surface_transform, self.surface_variables = surface_transform(join(dataset_dir, "surface_mean.pkl"), 
#                                                                            join(dataset_dir, "surface_std.pkl"))
#         self.upper_air_transform, self.upper_air_variables, self.upper_air_pLevels = upper_air_transform(join(dataset_dir, "upper_air_mean.pkl"), 
#                                                                                                          join(dataset_dir, "upper_air_std.pkl"))
        
#         times = [datetime.strptime(x.split(".")[-2][-7:], "%Y_%m") for x in self.surface_filenames]
#         st = min(times)  # include
#         et = max(times) + relativedelta(months=+1)  # exclude

#         self.date = np.arange(f"{st.year}-{str(st.month).zfill(2)}", f"{et.year}-{str(et.month).zfill(2)}", dtype='datetime64[D]')

#         self.land_mask, self.soil_type, self.topography = self._load_constant_mask()

#     def __getitem__(self, index):
#         surface_t, upper_air_t = self._get_data(index)
#         surface_t_1, upper_air_t_1 = self._get_data(index + 1)
#         if self.flag == "train":
#             return surface_t, upper_air_t, surface_t_1, upper_air_t_1
#         return surface_t, upper_air_t, surface_t_1, upper_air_t_1, torch.tensor([
#             self.date[index].astype(int), self.date[index + 1].astype(int)
#         ])

#     def _get_data(self, index):
#         date = self.date[index]
#         year = date.astype("datetime64[Y]").astype(int) + 1970
#         month = date.astype("datetime64[M]").astype(int) % 12 + 1
#         day = (date - date.astype("datetime64[M]")).astype(int) + 1

#         surface_file = join(self.dataset_dir, self.flag, f"surface_{year}_{str(month).zfill(2)}.nc")
#         upper_air_file = join(self.dataset_dir, self.flag, f"upper_air_{year}_{str(month).zfill(2)}.nc")

#         surface_ds = xr.open_dataset(surface_file).sel(time=f"{year}-{str(month).zfill(2)}-{str(day).zfill(2)}")
#         upper_air_ds = xr.open_dataset(upper_air_file).sel(time=f"{year}-{str(month).zfill(2)}-{str(day).zfill(2)}")

#         surface_data = np.stack([surface_ds[x].data for x in self.surface_variables], axis=0)  # C Lat Lon
#         surface_data = torch.from_numpy(surface_data.astype(np.float32))
#         surface_data = self.surface_transform(surface_data)

#         upper_air_data = torch.stack([self.upper_air_transform[pl](
#             torch.from_numpy(np.stack([upper_air_ds.sel(level=pl)[x].data for x in self.upper_air_variables], axis=0).astype(np.float32))
#         ) for pl in self.upper_air_pLevels], dim=1)  # C Pl Lat Lon
#         return surface_data, upper_air_data

#     def __len__(self):
#         return len(self.date) - 1

#     def _load_constant_mask(self):
#         mask_dir = "constant_mask"
#         land_mask = torch.from_numpy(np.load(join(self.dataset_dir, mask_dir, "land_mask.npy")).astype(np.float32))
#         soil_type = torch.from_numpy(np.load(join(self.dataset_dir, mask_dir, "soil_type.npy")).astype(np.float32))
#         topography = torch.from_numpy(np.load(join(self.dataset_dir, mask_dir, "topography.npy")).astype(np.float32))

#         return land_mask, soil_type, topography
    
#     def get_constant_mask(self):
#         return self.land_mask, self.soil_type, self.topography

#     def get_lat_lon(self):
#         example = self.surface_filenames[0]
#         ds = xr.open_dataset(example)
#         return ds["latitude"].data, ds["longitude"].data

def get_year_month_day(dt):
    year = dt.astype("datetime64[Y]").astype(int) + 1970
    month = dt.astype("datetime64[M]").astype(int) % 12 + 1
    day = (dt.astype("datetime64[D]") - dt.astype("datetime64[M]")).astype(int) + 1
    return year, month, day

class ZarrWeatherDataset(Dataset):
    def __init__(self, zarr_path, surface_vars, upper_air_vars, plevels, static_vars=None, 
                 year_range=None, split_type='train', surface_transform=None, upper_air_transform=None, chunk_size=1):
        """Initialize the dataset with optimized data loading.
        Args:
            zarr_path (str): Path to the Zarr dataset.
            surface_vars (list): List of surface variable names.
            upper_air_vars (list): List of upper air variable names.
            plevels (list): List of pressure levels for upper air variables.
            static_vars (list, optional): List of static variable names. Defaults to None.
            year_range (tuple, optional): Tuple of (start_year, end_year) to use. Defaults to None (all years).
            split_type (str, optional): One of 'train', 'val', or 'test'. Determines year range if year_range not provided.
            surface_transform (callable, optional): Transform for surface variables. Defaults to None.
            upper_air_transform (dict, optional): Dict of transforms for upper air variables by pressure level. Defaults to None.
            chunk_size (int, optional): Number of samples to load at once. Defaults to 1.
        """
        self.ds = xr.open_zarr(zarr_path, chunks={'time': chunk_size})
        self.surface_vars = surface_vars
        self.upper_air_vars = upper_air_vars
        self.plevels = sorted(plevels)
        self.static_vars = static_vars if static_vars is not None else []
        self.surface_transform = surface_transform
        self.upper_air_transform = upper_air_transform
        self.chunk_size = chunk_size

        # Cache dataset dimensions
        self.dims = dict(self.ds.dims)
        
        # Pre-compute pressure level indices
        if 'level' in self.ds.dims:
            self.plevel_indices = [self.ds.level.values.tolist().index(pl) for pl in self.plevels]
        
        # Cache static variables
        if self.static_vars:
            self.static_data = torch.from_numpy(
                np.stack([self.ds[var].values for var in self.static_vars], axis=0)
            ).float()
        else:
            self.static_data = None

        # Set up time indices based on years
        times = self.ds.time.values
        years = pd.DatetimeIndex(times).year

        if year_range is None:
            # Default splits based on split_type
            if split_type == 'train':
                year_range = (1959, 2017)
            elif split_type == 'val':
                year_range = (2018, 2020)
            elif split_type == 'test':
                year_range = (2021, 2023)
            else:
                raise ValueError(f"Invalid split_type: {split_type}")

        # Create mask for the specified years
        year_mask = (years >= year_range[0]) & (years <= year_range[1])
        self.indices = np.where(year_mask)[0][:-1]  # Exclude last timestep for each year

        # Initialize data cache
        self._cache = {}
        self._cache_idx = None

    def _load_chunk(self, t):
        """Load a chunk of data into cache."""
        # Surface variables: (time, lon, lat)
        surface_data = self.ds[self.surface_vars].isel(time=slice(t, t + 2)).to_array().values
        surface = surface_data[:, 0, ...]  # Current timestep (var, lat, lon)
        surface_target = surface_data[:, 1, ...]  # Next timestep (var, lat, lon)
        
        # Initialize lists to store upper air variables for each pressure level
        upper_air_vars = []
        upper_air_target_vars = []
        
        # Load data for each variable and pressure level
        for var in self.upper_air_vars:
            var_data = []
            var_target_data = []
            for pl in self.plevels:
                # Select data for current variable and pressure level
                level_data = self.ds[var].sel(level=pl).isel(time=slice(t, t + 2)).values
                var_data.append(level_data[0])  # Current timestep
                var_target_data.append(level_data[1])  # Next timestep
            upper_air_vars.append(np.stack(var_data))  # (level, lat, lon)
            upper_air_target_vars.append(np.stack(var_target_data))  # (level, lat, lon)
        
        # Stack all variables
        upper_air = np.stack(upper_air_vars)  # (var, level, lat, lon)
        upper_air_target = np.stack(upper_air_target_vars)  # (var, level, lat, lon)

        # Convert to torch tensors
        surface = torch.from_numpy(surface.astype(np.float32))
        surface_target = torch.from_numpy(surface_target.astype(np.float32))
        upper_air = torch.from_numpy(upper_air.astype(np.float32))
        upper_air_target = torch.from_numpy(upper_air_target.astype(np.float32))

        # Apply transforms
        if self.surface_transform is not None:
            surface = self.surface_transform(surface)
            surface_target = self.surface_transform(surface_target)
            
        if self.upper_air_transform is not None:
            # Apply transforms for each pressure level
            for level_idx, pl in enumerate(self.plevels):
                # Transform shape: (var, lat, lon)
                level_data = upper_air[:, level_idx]
                level_target = upper_air_target[:, level_idx]
                
                # Apply transform
                upper_air[:, level_idx] = self.upper_air_transform[pl](level_data)
                upper_air_target[:, level_idx] = self.upper_air_transform[pl](level_target)

        return {
            'surface': surface,
            'upper_air': upper_air,
            'static': self.static_data,
            'surface_target': surface_target,
            'upper_air_target': upper_air_target
        }

    def __getitem__(self, idx):
        """Get input and target data with caching.
        Args:
            idx (int): Index of the data point.
        Returns:
            dict: Dictionary containing input and target tensors.
        """
        t = self.indices[idx]
        chunk_idx = t // self.chunk_size

        # Check if we need to load a new chunk
        if self._cache_idx != chunk_idx:
            self._cache = self._load_chunk(t)
            self._cache_idx = chunk_idx

        return self._cache
        
    def __len__(self):
        return len(self.indices)

    def get_constant_mask(self):
        land_mask = torch.from_numpy(self.ds['land_sea_mask'].values.astype(np.float32))
        soil_type = torch.from_numpy(self.ds['soil_type'].values.astype(np.float32))
        topography = torch.from_numpy(self.ds['topography'].values.astype(np.float32))
        return land_mask, soil_type, topography
    
    def get_lat_lon(self):
        lat = self.ds['latitude'].values
        lon = self.ds['longitude'].values
        return lat, lon
    
