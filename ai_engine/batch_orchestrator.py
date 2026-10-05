# batch_orchestrator.py


from scipy.interpolate import griddata
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import concurrent.futures, logging, s3fs, sys, warnings


warnings.filterwarnings('ignore')

logging.basicConfig(level = logging.INFO, format = "%(asctime)s - [%(processName)s] - %(message)s")


def process_storm_tensor(target_date, output_dir):
    """Isolated worker func. Streams, masks, interpolates & saves a single tensor."""

    output_path = output_dir/f"structured_surge_{target_date}.nc"

    # Checkpoint: Don't re-process a file that already exists. If the VM dies, the orchestrator skips to where it left off.
    if output_path.exists():

        return f"Skipped {target_date} (Already exists)"

    try:
        # Conn to the AWS Open Data Registry without billing creds
        fs = s3fs.S3FileSystem(anon = True)

        # For data older than 30 days, this path would point to NOAA's NCEI cold-storage archives. It follows the operational GESTOFS arch we made.
        bucket_prefix = f'noaa-gestofs-pds/estofs.{target_date}'

        try:
            files = fs.ls(bucket_prefix)
        except FileNotFoundError:

            return f"Failed {target_date}: Dir not found in S3."

        nc_files = [f for f in files if f.endswith('.nc')]

        if not nc_files:

            return f"Failed {target_date}: No NetCDF found in S3 path."

        target_file = nc_files[0]

        with fs.open(target_file, 'rb') as f:
            with xr.open_dataset(f, engine = 'h5netcdf', drop_variables = ['nvel', 'nope', 'nbou']) as ds:
                raw_x = ds['x'].values
                raw_y = ds['y'].values
                raw_surge = ds['zeta_max'].values

                lon_min, lon_max = 84.0, 88.0
                lat_min, lat_max = 18.0, 22.0

                mask = (raw_x >= lon_min) & (raw_x <= lon_max) & (raw_y >= lat_min) & (raw_y <= lat_max)

                pts_x = raw_x[mask]
                pts_y = raw_y[mask]
                surge_vals = raw_surge[mask]

        if len(surge_vals) < 100:

            return f"Failed {target_date}: Insufficient regional nodes ({len(surge_vals)} found)"

        grid_x, grid_y = np.mgrid[lon_min:lon_max:256j, lat_min:lat_max:256j]
        
        pts = np.column_stack((pts_x, pts_y))

        grid_surge = griddata(pts, surge_vals, (grid_x, grid_y), method = 'linear', fill_value = 0.0)

        # Package as a PyTorch-ready NetCDF. Now the data is a 2D matrix (x, y) instead of 1D list of pts.
        processed_ds = xr.Dataset({'surge_label': (['x', 'y'], grid_surge)}, coords = {'x': (['x'], grid_x[:, 0]), 'y': (['y'], grid_y[0, :])})

        with processed_ds:
            processed_ds.to_netcdf(output_path)

        return f"Success {target_date}"
    except Exception as e:
        
        return f"Failed {target_date}: {str(e)}"


output_dir = Path('data/training_tensors')
output_dir.mkdir(parents = True, exist_ok = True)

logging.info("Gen-ing 5000 target dates...")

date_range = pd.date_range(end = pd.Timestamp.today(), periods = 5000)
target_dates = date_range.strftime(r'%Y%m%d').tolist()

logging.info(f"Init-ing headless multi-core orchestration for {len(target_dates)} tensors...")

success_count = fail_count = 0

# max_workers = 12 leaves cores open for the OS while saturating the transoceanic network link.
with concurrent.futures.ProcessPoolExecutor(max_workers = 12) as executor:
    # Submit all 5000 tasks to the queue.
    futures = {executor.submit(process_storm_tensor, date, output_dir): date for date in target_dates}

    for future in concurrent.futures.as_completed(futures):
        try:
            result = future.result()

            logging.info(result)

            if 'Success' in result or 'Skipped' in result:
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            logging.error(f"Worker crashed completely: {e}")

            fail_count += 1

logging.info(f"Batch extraction complete. Success: {success_count} | Failed: {fail_count}")