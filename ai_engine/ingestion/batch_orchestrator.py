# batch_orchestrator.py


from pathlib import Path

import numpy as np
import xarray as xr

import concurrent.futures, gc, logging, multiprocessing, s3fs, sys, warnings


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from config import BLACKLIST_PATH, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, UNSTRUCTURED_DIR


warnings.filterwarnings('ignore')

logging.basicConfig(level = logging.INFO, format = "%(asctime)s - [%(processName)s] - %(message)s")


BUCKET_NAME = 'noaa-gestofs-pds'

S3_CONFIG = {'connect_timeout': 30, 'read_timeout': 300, 'retries': {'max_attempts': 10, 'mode': 'adaptive'}}


def discover_available_s3_dates(bucket_name, blacklisted_dates):
    """Queries the S3 bucket root once in the main process."""

    fs = s3fs.S3FileSystem(anon = True, config_kwargs = S3_CONFIG)

    logging.info(f"Scanning s3://{bucket_name} for existing date dirs...")

    dirs = fs.ls(bucket_name)

    available_map = {}

    for d in dirs:
        folder = d.split('/')[-1]

        if folder.startswith('gestofs.') or folder.startswith('estofs.'):
            date_str = folder.split('.')[-1]

            if date_str not in blacklisted_dates:
                available_map[date_str] = d

    return available_map

def process_storm_tensor(target_date, s3_dir_path, output_dir):
    """Streams, masks at the HDF5 index level & saves a regional unstruct-ed temporal tensor."""

    output_path = output_dir/f'unstructured_surge_{target_date}.nc'

    # Checkpoint: Don't re-process a file that already exists. If the process dies, the orchestrator skips to where it left off.
    if output_path.exists():

        return 'SKIPPED', target_date, f"Skipped {target_date} (Already exists)"

    try:
        # Conn to the AWS Open Data Registry without billing creds
        fs = s3fs.S3FileSystem(anon = True, default_block_size = 8*1024*1024, config_kwargs = S3_CONFIG)

        try:
            files = fs.find(s3_dir_path)
        except FileNotFoundError:

            return 'FAILED', target_date, f"Failed {target_date}: Dir not found in S3."
        except Exception as e:

            return 'FAILED', target_date, f"Failed {target_date}: AWS Network Timeout/Err - {str(e)}"

        nc_files = [f for f in files if f.endswith('.nc') and 'cwl' in f and 'max' not in f] # 'cwl' - Cont. H2O level. It contains the hourly progression of the H2O
                                                                          # level across the grid. Has the time-series 'zeta' tensor.

        if not nc_files:

            return 'FAILED', target_date, f"Failed {target_date}: No time-series (cwl) NetCDF found in {s3_dir_path}."

        target_file = sorted(nc_files)[0]

        logging.info(f"[{target_date}] Streaming HDF5 metadata & regional slice from s3://{target_file}...")

        with fs.open(target_file, 'rb') as f:
            with xr.open_dataset(f, engine = 'h5netcdf', drop_variables = ['nvel', 'nope', 'nbou', 'element']) as ds:
                if 'time' not in ds.dims:

                    return 'FAILED', target_date, f"Failed {target_date}: Dataset lacks time dim."

                if 'zeta' not in ds.data_vars:

                    return 'FAILED', target_date, f"Failed {target_date}: File lacks hourly 'zeta' time-series. Found: {list(ds.data_vars.keys())}"

                raw_x = ds['x'].values
                raw_y = ds['y'].values
                mask = (raw_x >= LON_MIN) & (raw_x <= LON_MAX) & (raw_y >= LAT_MIN) & (raw_y <= LAT_MAX)
                node_indices = np.flatnonzero(mask)

                if len(node_indices) < 100:

                    return 'FAILED', target_date, f"Failed {target_date}: Insufficient regional nodes ({len(node_indices)} found)"

                pts_x = raw_x[node_indices]
                pts_y = raw_y[node_indices]

                del raw_x, raw_y, mask

                node_dim = ds['x'].dims[0]
                # Extract time series seqs instead of a static max
                time_series = ds['time'].values
                # Slice only the regional node indices over S3-B range reqs
                masked_zeta = ds['zeta'].isel({node_dim: node_indices}).values

                if 'u-vel' in ds.data_vars:
                    masked_u = ds['u-vel'].isel({node_dim: node_indices}).values
                else:
                    masked_u = np.zeros_like(masked_zeta)

                if 'v-vel' in ds.data_vars:
                    masked_v = ds['v-vel'].isel({node_dim: node_indices}).values
                else:
                    masked_v = np.zeros_like(masked_zeta)

                processed_ds = xr.Dataset(
                    {
                        'zeta': (['time', 'node'], masked_zeta),
                        'u-vel': (['time', 'node'], masked_u),
                        'v-vel': (['time', 'node'], masked_v),
                        'x': (['node'], pts_x),
                        'y': (['node'], pts_y),
                    },
                    coords = {'time': time_series}
                )

                processed_ds.to_netcdf(output_path, engine = 'h5netcdf')

        gc.collect()

        return 'SUCCESS', target_date, f"Success {target_date} ({len(node_indices)} regional nodes)"
    except Exception as e:
        
        return 'FAILED', target_date, f"Failed {target_date}: {str(e)}"
    

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force = True)

    output_dir = UNSTRUCTURED_DIR
    blacklist_path = BLACKLIST_PATH

    if blacklist_path.exists():
        with open(blacklist_path, 'r') as f:
            blacklisted_dates = set(f.read().splitlines())
    else:
        blacklisted_dates = set()

    available_s3_dates = discover_available_s3_dates(BUCKET_NAME, blacklisted_dates)

    logging.info(f"Valid S3 dates discovered {len(available_s3_dates)} | Blacklisted: {len(blacklisted_dates)}")

    if not available_s3_dates:
        logging.warning("No matching un-blacklisted dates found in S3 bucket. Exiting.")

        raise SystemExit(0)

    target_items = list(sorted(available_s3_dates.items(), reverse = True))[:1]

    logging.info(f"Init-ing headless multi-core orchestration for {len(target_items)} tensor(s)...")

    success_count = fail_count = 0

    new_fails = []

    # max_workers = 4 leaves cores open for the OS while saturating the transoceanic network link.
    with concurrent.futures.ProcessPoolExecutor(max_workers = 2, max_tasks_per_child = 5) as executor:
        # Submit all 5000 tasks to the queue.
        futures = {executor.submit(process_storm_tensor, date, s3_path, output_dir): date for date, s3_path in target_items}

        for future in concurrent.futures.as_completed(futures):
            date = futures[future]

            try:
                status, _, msg = future.result()

                if status in ('SUCCESS', 'SKIPPED'):
                    logging.info(msg)

                    success_count += 1
                else:
                    logging.warning(msg)
                    
                    fail_count += 1

                    new_fails.append(date)
            except Exception as e:
                logging.error(f"Worker crashed on {date}: {e}")

                fail_count += 1

                new_fails.append(date)

    if new_fails:
        with open(blacklist_path, 'a') as f:
            for fd in new_fails:
                f.write(f'{fd}\n')

        logging.info(f"Appended {len(new_fails)} failed dates to blacklist.")

    logging.info(f"Batch extraction complete. Success: {success_count} | Failed: {fail_count}")