# surge_dataset.py


from pathlib import Path
from torch.utils.data import DataLoader, Dataset

import numpy as np
import xarray as xr

import logging, torch


logging.basicConfig(level = logging.INFO, format = "%(asctime)s - [%(levelname)s] - %(message)s")


class HydroDataset(Dataset):

    """Ingest struct-ed 256x256 NetCDF tensors & serves them as normalized PyTorch batches for U-Net training."""

    def __init__(self, tensor_dir):
        self.tensor_paths = list(Path(tensor_dir).glob('structured*.nc'))

        if not self.tensor_paths:
            raise FileNotFoundError(f"No training tensors found in {tensor_dir}.")

        logging.info(f"Init-ed PyTorch Dataset with {len(self.tensor_paths)} storm tensors.")

        # Max physical vals for min-max scaling (normalizing b/w 0 & 1). NNs suffer from gradient explosion if unscaled physical vals (e.g. wind = 150) are passed.
        self.max_surge = 10.0 # Max expected surge in m

    def __len__(self):

        return len(self.tensor_paths)

    def __getitem__(self, idx):
        file_path = self.tensor_paths[idx]

        # Load the struct-ed 2D grid
        with xr.open_dataset(file_path) as ds:
            # We use np.nan_to_num to convert any missing data pts over dry land to 0.0
            surge_grid = np.nan_to_num(ds['surge_label'].values, nan = 0.0)

        # NNs require the channel dim 1st: [Channels, Height, Width]. Since surge is the only var rn, we expand dims to make it [1, 256, 256]
        surge_tensor = np.expand_dims(surge_grid, axis = 0)
        surge_tensor = surge_tensor/self.max_surge # Normalize

        # Rn, we're only extracting the label (Y). Normally, we would extract wind & topography (X) here. For now, we're passing a dummy X to validate the arch pipeline.
        y_label = torch.from_numpy(surge_tensor).float() # Convert std np arrs into PyTorch computational tensors

        x_features = torch.zeros((3, 256, 256)) # 3 channels: topo, wind_u, wind_v

        return x_features, y_label


dataset = HydroDataset('../data_pipeline/data/training_tensors')
dataloader = DataLoader(dataset, batch_size = 1, shuffle = True) # DataLoader handles batching, shuffling & multi-threaded CPU loading.

for batch_idx, (X, Y) in enumerate(dataloader):
    logging.info(f"Batch {batch_idx} gen-d.")
    logging.info(f"Input feature tensor (X) shape: {X.shape}")
    logging.info(f"Target surge tensor (Y) shape: {Y.shape}")

    break