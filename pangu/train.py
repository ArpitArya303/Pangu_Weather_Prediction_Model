import argparse
import os
import numpy as np
import pandas as pd
import xarray as xr

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from pangu_model import Pangu 
from data_utils import ZarrWeatherDataset, surface_inv_transform, upper_air_inv_transform

parser = argparse.ArgumentParser(description="Train Pangu")
parser.add_argument("--num_epochs", default=50, type=int, help="train epoch number")
parser.add_argument("--zarr_path", required=True, help="Path to your .zarr file")
parser.add_argument("--surface_variables", nargs='+', default=["u10", "v10", "t2m", "msl"], help="Surface variables4")
parser.add_argument("--upper_air_variables", nargs='+', default=["u", "v", "t", "r"], help="Upper air variables")
parser.add_argument("--upper_air_pLevels", nargs='+', default=[1000, 850, 700, 500, 300, 200, 100], type=int, help="Pressure levels for upper air variables")
parser.add_argument("--static_variables", nargs='+', default=["land_sea_mask", "soil_type", "topography"], help="Static variables")
parser.add_argument("--log_dir", default="runs/trial", help="Directory to save logs")

if __name__ == "__main__":
    # parse args
    opt = parser.parse_args()
    NUM_EPOCHS = opt.num_epochs
    zarr_path = opt.zarr_path
    surface_vars = opt.surface_variables
    upper_air_vars = opt.upper_air_variables
    upper_air_pLevels = opt.upper_air_pLevels
    static_vars = opt.static_variables
    log_dir = opt.log_dir

    # prepare data
    ds = xr.open_zarr(zarr_path)
    n = ds.dims['time'] - 1
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(0.8 * n)
    train_idx, val_idx = idx[:split], idx[split:]

    train_set = ZarrWeatherDataset(opt.zarr_path, surface_vars, upper_air_vars, static_vars, indices=train_idx)
    val_set = ZarrWeatherDataset(opt.zarr_path, surface_vars, upper_air_vars, static_vars, indices=val_idx)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    
    # Get static mask (same for all samples)
    land_mask, soil_type, topography = train_set.get_constant_mask()
    surface_mask = torch.stack([land_mask, soil_type, topography], dim=0)  # C Lat Lon
    lat, lon = train_set.get_lat_lon()

    pangu = Pangu()
    print("# parameters: ", sum(param.numel() for param in pangu.parameters()))

    surface_criterion = nn.L1Loss()
    upper_air_criterion = nn.L1Loss()

    if torch.cuda.is_available():
        pangu.cuda()
        surface_criterion.cuda()
        upper_air_criterion.cuda()

        surface_mask = surface_mask.cuda()

    surface_invTrans, surface_variables = surface_inv_transform("data/surface_mean.pkl", "data/surface_std.pkl")
    upper_air_invTrans, upper_air_variables, upper_air_pLevels = upper_air_inv_transform("data/upper_air_mean.pkl", "data/upper_air_std.pkl")

    # The learning rate is 5e-4 as in the paper, while the weight decay is 3e-6
    optimizer = torch.optim.Adam(pangu.parameters(), lr=5e-4, weight_decay=3e-6)

    results = {'loss': [], 'surface_mse': [], 'upper_air_mse': []}

    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {"batch_sizes": 0, "loss": 0}

        pangu.train()
        for batch in train_bar:
            input_surface = batch['surface']
            input_upper_air = batch['upper_air']
            target_surface = batch['surface_target']
            target_upper_air = batch['upper_air_target']
            batch_size = input_surface.size(0)
            if torch.cuda.is_available():
                input_surface = input_surface.cuda()
                input_upper_air = input_upper_air.cuda()
                target_surface = target_surface.cuda()
                target_upper_air = target_upper_air.cuda()

            output_surface, output_upper_air = pangu(input_surface, surface_mask, input_upper_air)

            optimizer.zero_grad()
            surface_loss = surface_criterion(output_surface, target_surface)
            upper_air_loss = upper_air_criterion(output_upper_air, target_upper_air)
            loss = upper_air_loss + surface_loss * 0.25
            loss.backward()
            optimizer.step()

            running_results["loss"] += loss.item() * batch_size
            running_results["batch_sizes"] += batch_size

            train_bar.set_description(desc="[%d/%d] Loss: %.4f" % 
                                      (epoch, NUM_EPOCHS, running_results["loss"] / running_results["batch_sizes"]))

        with torch.no_grad():
            val_bar = tqdm(val_loader)
            valing_results = {"batch_sizes": 0, "surface_mse": 0, "upper_air_mse": 0} 
            for batch in val_bar:
                val_input_surface = batch['surface']
                val_input_upper_air = batch['upper_air']
                val_target_surface = batch['surface_target']
                val_target_upper_air = batch['upper_air_target']
                batch_size = val_input_surface.size(0)
                if torch.cuda.is_available():
                    val_input_surface = val_input_surface.cuda()
                    val_input_upper_air = val_input_upper_air.cuda()
                    val_target_surface = val_target_surface.cuda()
                    val_target_upper_air = val_target_upper_air.cuda()

                val_output_surface, val_output_upper_air = pangu(val_input_surface, surface_mask, val_input_upper_air)

                val_output_surface = val_output_surface.squeeze(0)  # C Lat Lon
                val_output_upper_air = val_output_upper_air.squeeze(0)  # C Pl Lat Lon

                val_target_surface = val_target_surface.squeeze(0)
                val_target_upper_air = val_target_upper_air.squeeze(0)

                valing_results["batch_sizes"] += batch_size

                surface_mse = ((val_output_surface - val_target_surface) ** 2).data.mean().cpu().item()
                upper_air_mse = ((val_output_upper_air - val_target_upper_air) ** 2).data.mean().cpu().item()

                valing_results["surface_mse"] += surface_mse * batch_size
                valing_results["upper_air_mse"] += upper_air_mse * batch_size

                val_bar.set_description(desc="[validating] Surface MSE: %.4f Upper Air MSE: %.4f" % 
                                        (valing_results["surface_mse"] / valing_results["batch_sizes"], valing_results["upper_air_mse"] / valing_results["batch_sizes"]))
        
        os.makedirs("epochs", exist_ok=True)
        torch.save(pangu.state_dict(), "epochs/pangu_epoch_%d.pth" % (epoch))

        avg_train_loss = running_results["loss"] / running_results["batch_sizes"]
        avg_surface_mse = valing_results["surface_mse"] / valing_results["batch_sizes"]
        avg_upper_air_mse = valing_results["upper_air_mse"] / valing_results["batch_sizes"]

        # Log to TensorBoard
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('MSE/val_surface', avg_surface_mse, epoch)
        writer.add_scalar('MSE/val_upper_air', avg_upper_air_mse, epoch)

        # results["loss"].append(running_results["loss"] / running_results["batch_sizes"])
        # results["surface_mse"].append(valing_results["surface_mse"] / valing_results["batch_sizes"])
        # results["upper_air_mse"].append(valing_results["upper_air_mse"] / valing_results["batch_sizes"])

        data_frame = pd.DataFrame(
            data=results, 
            index=range(1, epoch + 1)
        )
        save_root = "train_logs"
        if not os.path.exists(save_root):
            os.makedirs(save_root)

        data_frame.to_csv(os.path.join(save_root, "logs.csv"), index_label="Epoch")
