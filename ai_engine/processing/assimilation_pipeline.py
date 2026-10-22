# assimilation_engine.py


from pathlib import Path

import numpy as np
import xarray as xr

import argparse, logging, sys


logging.basicConfig(level = logging.INFO, format = "%(asctime)s - [%(levelname)s] - %(message)s")


class GribProcessor:

    """
        Ingests raw GRIB2 meteorological fields & calculates hydrodynamic forcing vectors. We encapsulate this in a class to maintain state if we process sequential
        time-steps.
    """

    def __init__(self, grib_path):
        self.grib_path = Path(grib_path)
        self.air_density = 1.225 # kg/m^3 std atm density

        if not self.grib_path.exists():
            raise FileNotFoundError(f"GRIB data not found at {self.grib_path}")

    def load_dataset(self):
        """
            Loads the GRIB2 file into an xarray dataset. Why xarray? Because metorological data is often multi-dimensional (time, latitude, longitude, etc.), & xarray
            provides a convenient way to handle such data. Using pandas/np would destroy the spatial relationships.
        """

        logging.info(f"Loading GRIB2 dataset from {self.grib_path}")

        try:

            return xr.open_dataset(self.grib_path, engine = 'cfgrib') # engine = 'cgfrib' explicitly calls the eccodes C-lib to decode the World Meterological
                                                                      # Organization (WMO) GRIB2 format. This is the most robust way to handle GRIB2 files in Py.
        except Exception as e:
            logging.error("Failed to decode GRIB. Ensure eccodes is installed at the OS level.")

            raise e

    def calculate_wind_stress(self, ds):
        """Translates raw u & v velocity vectors into physical surface stress."""

        logging.info("Calculating wind stress from u & v velocity vectors...")

        # (GRIB uses std short names: 'u10' & 'v10' for 10-meter wind components)
        u_wind = ds['u10'] # 10-meter u-component of wind (m/s)
        v_wind = ds['v10'] # 10-meter v-component of wind (m/s)

        wind_speed = np.sqrt(u_wind**2 + v_wind**2) # Magnitude of wind vector. xarray handles the numpy vectorization automatically across the entire spatial grid.

        cd = xr.where(wind_speed < 11.0, 0.0012, 0.00049 + (0.000065*wind_speed)) # Drag coefficient (Cd) is a func of wind speed. It uses a simplified Large & Pond
                                                                                  # (1981) piecewise func. If speed < 11 m/s, Cd is ~0.0012, else it increases linearly
                                                                                  # with speed.

        wind_stress = self.air_density*cd*(wind_speed**2) # Wind stress (Pa) = air density * drag coefficient * wind speed squared. This is the physical force exerted by
                                                          # the wind on the ocean surface.
                                                        
        ds['wind_speed'] = wind_speed
        ds['wind_speed'].attrs = {'units': 'm s**-1', 'long_name': "10m wind speed magnitude"}
        ds['wind_stress'] = wind_stress
        ds['wind_stress'].attrs = {'units': 'N m**-2', 'long_name': "Surface wind stress"}

        return ds

    def export_forcing_data(self, ds, output_filename):
        """Exports the processed grid. Why NetCDF? Industry-std for storing multi-dimensional arrs. Stores the arrs, the coords (lat/lon) & metadata in 1 compiled bin."""

        out_path = Path('data/processed') / output_filename
        out_path.parent.mkdir(parents = True, exist_ok = True) # Ensure the output dir exists

        logging.info(f"Writing forcing data to NetCDF: {out_path}")

        ds_clean = ds[['wind_speed', 'wind_stress', 'msl']] # msl = Mean Sea Level Pressure. We only want to export the relevant forcing fields, not the entire GRIB dataset.
        ds_clean.to_netcdf(out_path) # Write to NetCDF4 format. This is a widely supported format for scientific data.

        logging.info("Export complete.")


parser = argparse.ArgumentParser(description = "Ingests raw GRIB2 & calculates hydrodynamic forcing grids.")
parser.add_argument('--input', required = True, help = "Path to the raw GRIB2 forecast file.")
parser.add_argument('--output', required = True, help = "Path to the output NetCDF (.nc) file.")

args = parser.parse_args()

try:
    processor = GribProcessor(args.input) # Instantiate the processor with the path to the GRIB2 file
    raw_ds = processor.load_dataset() # Load the raw GRIB2 dataset
    forced_ds = processor.calculate_wind_stress(raw_ds) # Calculate wind stress and add it to the dataset
    processor.export_forcing_data(forced_ds, args.output) # Export the processed
except Exception as e:
    logging.critical(f"Pipeline failed: {e}")

    sys.exit(1)