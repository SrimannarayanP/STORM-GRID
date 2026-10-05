# mesh_interpolator.py


from scipy.interpolate import griddata
from pathlib import Path

import numpy as np
import xarray as xr

import logging, sys


logging.basicConfig(level = logging.INFO, format = "%(asctime)s - [%(levelname)s] - %(message)s")


class TensorInterpolator():
    
    """Converts unstruct-ed ADCIRC finite-element meshes into uniform Cartesian grids required for CNN/U-Net training."""

    def __init__(self, raw_tensor_path):
        self.raw_path = Path(raw_tensor_path)

        if not self.raw_path.exists():
            raise FileNotFoundError(f"Missing raw tensor at {self.raw_path}")

    def grid_mesh(self, output_name, resolution = 256):
        logging.info(f"Loading unstruct-ed mesh from {self.raw_path}...")

        try:
            with xr.open_dataset(self.raw_path) as ds:
                # Extract 1D point clouds. x & y rep spatial coords, zeta_max is the max surge height.
                raw_x = ds['x'].values
                raw_y = ds['y'].values
                raw_surge = ds['zeta_max'].values

                logging.info(f"Extracted {len(raw_surge)} global nodes. Applying geographic filter...")

                # Def the target bounding box. A small buffer is added around the 256x256 grid to ensure edge interpolation is accurate.
                lon_min, lon_max = 84.0, 88.0
                lat_min, lat_max = 18.0, 22.0

                mask = (raw_x >= lon_min) & (raw_x <= lon_max) & (raw_y >= lat_min) & (raw_y <= lat_max)

                pts_x = raw_x[mask]
                pts_y = raw_y[mask]
                surge_vals = raw_surge[mask]

                pts = np.column_stack((pts_x, pts_y))

                logging.info(f"Filter complete. Reduced to {len(surge_vals)} regional nodes for triangulation.")

                if (len(surge_vals) < 100):
                    logging.error("Too few nodes in bounding box. Check coords.")

                    sys.exit(1)
        except KeyError as e:
            logging.error(f"Missing expected vars in NetCDF: {e}")

            sys.exit(1)

        # Def the PyTorch target grid. Create a 256x256 bounding box based on the min/max of the mesh
        logging.info(f"Gen-ing {resolution}x{resolution} Cartesian bounding box...")

        grid_x, grid_y = np.mgrid[pts_x.min():pts_x.max():complex(0, resolution), pts_y.min():pts_y.max():complex(0, resolution)]

        # Spatial interpolation: Project the unstruct-ed pts onto the grid. 'linear' interpolates the triangles. fill-value = 0 sets dry land to 0.
        logging.info("Executing Delaunay triangluation & grid projection...")

        grid_surge = griddata(pts, surge_vals, (grid_x, grid_y), method = 'linear', fill_value = 0.0)

        # Package as a PyTorch-ready NetCDF. Now the data is a 2D matrix (x, y) instead of 1D list of pts.
        processed_ds = xr.Dataset({'surge_label': (['x', 'y'], grid_surge)}, coords = {'x': (['x'], grid_x[:, 0]), 'y': (['y'], grid_y[0, :])})

        out_path = Path('data/training_tensors')/f'{output_name}.nc'

        logging.info(f"Writing struct-ed CNN tensor to {out_path}...")

        with processed_ds:
            processed_ds.to_netcdf(out_path)

        logging.info("Interpolation complete. Tensor is ready for PyTorch.")


interpolator = TensorInterpolator('data/training_tensors/latest_surge_tensor.nc')
interpolator.grid_mesh(output_name = 'structured_surge_tensor_256', resolution = 256)