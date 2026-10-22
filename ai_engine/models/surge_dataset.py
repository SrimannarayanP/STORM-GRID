# surge_dataset.py


from pathlib import Path
from torch.utils.data import DataLoader, Dataset

import numpy as np
import xarray as xr

import logging, sys, torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from config import TRAINING_TENSORS_DIR


logging.basicConfig(level = logging.INFO, format = "%(asctime)s - [%(levelname)s] - %(message)s")


class HydroDataset(Dataset):

    """Feeds sequential 1-hr transitions into Autoregressive PINN."""

    def __init__(self, tensor_dir = TRAINING_TENSORS_DIR):
        self.tensor_paths = list(Path(tensor_dir).glob('structured*.nc'))

        if not self.tensor_paths:
            raise FileNotFoundError(f"No training tensors found in {tensor_dir}.")

        self.samples = []

        for file_path in self.tensor_paths:
            with xr.open_dataset(file_path, engine = 'h5netcdf') as ds:
                if 'time' not in ds.dims:
                    logging.warning(f"Skipping {file_path}: No temporal dim found.")

                    continue

                time_steps = len(ds['time'])

                for t_idx in range(time_steps - 1):
                    self.samples.append((file_path, t_idx))

        logging.info(f"Init-ed Dataset: {len(self.tensor_paths)} storm yielding {len(self.samples)} valid hourly transitions.")

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):
        file_path, t0 = self.samples[idx]

        t1 = t0 + 1

        # Load the struct-ed 2D grid
        with xr.open_dataset(file_path, engine = 'h5netcdf') as ds:
            # We use np.nan_to_num to convert any missing data pts over dry land to 0.0
            topo = np.nan_to_num(ds['topo'].values, nan = 0.0)
            # Fluid state at T0 (Initial cond for autoregressive loop)
            h_t0 = np.nan_to_num(ds['surge_h'].isel(time = t0).values, nan = 0.0)
            u_t0 = np.nan_to_num(ds['surge_u'].isel(time = t0).values, nan = 0.0)
            v_t0 = np.nan_to_num(ds['surge_v'].isel(time = t0).values, nan = 0.0)
            # Atm forces at T0 & T1 (for linear interpolation during micro-steps)
            wind_u_t0 = np.nan_to_num(ds['wind_u'].isel(time = t0).values, nan = 0.0)
            wind_v_t0 = np.nan_to_num(ds['wind_v'].isel(time = t0).values, nan = 0.0)
            wind_u_t1 = np.nan_to_num(ds['wind_u'].isel(time = t1).values, nan = 0.0)
            wind_v_t1 = np.nan_to_num(ds['wind_v'].isel(time = t1).values, nan = 0.0)
            # Target fluid state at T1 (Ground truth for data MSE loss)
            h_t1 = np.nan_to_num(ds['surge_h'].isel(time = t1).values, nan = 0.0)
            u_t1 = np.nan_to_num(ds['surge_u'].isel(time = t1).values, nan = 0.0)
            v_t1 = np.nan_to_num(ds['surge_v'].isel(time = t1).values, nan = 0.0)

        static_bathy = torch.from_numpy(np.expand_dims(topo, axis = 0)).float()
        fluid_t0 = torch.from_numpy(np.stack([h_t0, u_t0, v_t0], axis = 0)).float()
        wind_t0 = torch.from_numpy(np.stack([wind_u_t0, wind_v_t0], axis = 0)).float()
        wind_t1 = torch.from_numpy(np.stack([wind_u_t1, wind_v_t1], axis = 0)).float()
        target_fluid_t1 = torch.from_numpy(np.stack([h_t1, u_t1, v_t1], axis = 0)).float()

        return static_bathy, wind_t0, wind_t1, fluid_t0, target_fluid_t1


if __name__ == '__main__':
    try:
        dataset = HydroDataset(TRAINING_TENSORS_DIR)
        dataloader = DataLoader(dataset, batch_size = 2, shuffle = True) # DataLoader handles batching, shuffling & multi-threaded CPU loading.

        static_bathy, wind_t0, wind_t1, fluid_t0, target_h_t1 = next(iter(dataloader))
        
        logging.info(f"Static Bathy: {static_bathy.shape}")
        logging.info(f"Wind T0: {wind_t0.shape} | Wind T1: {wind_t1.shape}")
        logging.info(f"Initial Fluid (T0): {fluid_t0.shape}")
        logging.info(f"Target Depth (T1): {target_h_t1.shape}")
    except Exception as e:
        logging.error(e)