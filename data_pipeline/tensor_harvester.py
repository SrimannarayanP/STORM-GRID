# tensor_harvester.py


from pathlib import Path

import xarray as xr

import logging, s3fs, sys


logging.basicConfig(level = logging.INFO, format = "%(asctime)s - [%(levelname)s] - %(message)s")


class AWSTensorHarvester:

    """
        Connects to NOAA's (Natl Oceanic & Atmospheric Admin) AWS Open Data Registry S3 buckets to stream historical hydrodynamic NetCDF files directly into
        PyTorch-ready sensors.
    """

    def __init__(self, bucket_name):
        logging.info(f"Mounting AWS S3 bucket: {bucket_name}")

        try:
            self.fs = s3fs.S3FileSystem(anon = True) # anon = True allows us to access the open registry without AWS billing creds.
            self.bucket = bucket_name
        except Exception as e:
            logging.error("Failed to conn to AWS S3.")

            raise e

    def extract_latest_tensor(self, output_name):
        logging.info(f"Scanning root of s3://{self.bucket} for available dates...")

        try:
            dirs = self.fs.ls(self.bucket)
            date_dirs = [d for d in dirs if 'estofs.' in d or 'gestofs.' in d]

            if not date_dirs:
                logging.error("No valid date dirs found in the bucket.")

                sys.exit(1)

            latest_dir = sorted(date_dirs)[-1]

            logging.info(f"Latest operational data dir found: {latest_dir}")

            files = self.fs.ls(latest_dir)
            nc_files = [f for f in files if f.endswith('.nc')]

            if not nc_files:
                logging.error(f"No NetCDF files found in {latest_dir}.")

                sys.exit(1)

            target_file = nc_files[0]

            logging.info(f"Streaming {target_file} into memory...")

            with self.fs.open(target_file, 'rb') as f:
                ds = xr.open_dataset(f, engine = 'h5netcdf', drop_variables = ['nvel'])

                logging.info(f"Vars mapped successfully: {list(ds.data_vars.keys())}")
                logging.info(f"Dataset coords: {list(ds.coords.keys())}")

                out_path = Path('data/training_tensors')/f'{output_name}.nc'
                out_path.parent.mkdir(parents = True, exist_ok = True)

                logging.info(f"Writing raw tensor to {out_path}...")

                ds.to_netcdf(out_path)

                logging.info("Extraction complete.")
        except Exception as e:
            logging.error(f"Extraction failed: {e}")

            sys.exit(1)


harvester = AWSTensorHarvester('noaa-gestofs-pds') # Pointing to the Global Extratropical Surge & Tide Operational Forecast Sys on AWS
harvester.extract_latest_tensor(output_name = 'latest_surge_tensor')

