# gee_ingestion.py


from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import ee, io, logging, requests, rioxarray, sys, zipfile


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from config import GRID_RES, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, PROCESSED_DIR, RAW_DIR, UNSTRUCTURED_DIR


logging.basicConfig(level = logging.INFO, format = "%(asctime)s - [%(levelname)s] - %(message)s")


GCP_PROJECT_ID = 'cyclone-forecaster-509919'


class GEEDataFetcher:

    """Automates the extraction of static GeoTIFFs & dynamic ERA5 hourly wind cubes from Google Earth Engine (GEE)."""

    def __init__(self, project_id):
        try:
            # Inits Earth Engine API using local Google creds.
            ee.Initialize(project = project_id)

            logging.info("Earth Engine API auth-d & conn-ed.")
        except Exception as e:
            logging.error(f"Failed to connect to GEE. You must run 'earthengine authenticate' in your terminal 1st.")
            
            raise e

    def fetch_geotiff(self, image, bbox, scale, output_path):
        """Sends a spatial query to Google's servers & downloads the clipped result."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents = True, exist_ok = True)

        # Define the spatial boundary: [West, South, East, North]
        region = ee.Geometry.BBox(*bbox)

        logging.info(f"Requesting server-side crop for {output_path.name} (scale = {scale}m)...")

        # Req the computation. getDownloadURL() forces Google to process the crop at the req-d 'scale' (resolution in m per pixel) & package the resulting matrix in
        # GeoTIFF format.
        url = image.getDownloadURL({'region': region, 'scale': scale, 'format': 'GEO_TIFF', 'crs': 'EPSG:4326'}) # EPSG:4326 forces std lat/lon projection for
                                                                                                                 # consistency across datasets.

        logging.info(f"Streaming bin data for {output_path.name}...")

        # Stream the res into memory. Google returns a ZIP file. We extract the internal .tif directly into our raw data folder without saving the ZIP to disk.
        res = requests.get(url, timeout = 120)

        if res.status_code != 200:
            raise Exception(f"Google API err {res.status_code}: {res.text}")

        try:
            with zipfile.ZipFile(io.BytesIO(res.content)) as z:
                for file_name in z.namelist():
                    if file_name.endswith('.tif'):
                        extracted_data = z.read(file_name)

                        with open(output_path, 'wb') as f:
                            f.write(extracted_data)
        except zipfile.BadZipFile:
            with open(output_path, 'wb') as f:
                f.write(res.content)

        logging.info(f"Successfully secured: {output_path} ({output_path.stat().st_size/1024:.1f} KB)")

    def fetch_storm_winds_for_nc(self, unstructured_nc_path, bbox):
        """
            Reads the exact timestamps from an unstruct-ed storm NetCDF file, downloads the matching hourly 10m wind vectors from ECMWF/ERA5/HOURLY & saves a 256x256
            wind_{date_tag}.nc file.
        """

        date_tag = unstructured_nc_path.stem.replace('unstructured_surge_', '')
        out_wind_nc = PROCESSED_DIR/f'wind_{date_tag}.nc'

        if out_wind_nc.exists():
            logging.info(f"Wind forcing {out_wind_nc.name} already exists. Skipping.")

            return out_wind_nc

        with xr.open_dataset(unstructured_nc_path, engine = 'h5netcdf') as ds:
            raw_times = pd.to_datetime(ds['time'].values)
            num_t = len(raw_times)

        t_start = raw_times[0].strftime(r'%Y-%m-%dT%H:%M:%S')
        # Add 1 hr buffer to end date because GEE filterDate end bound is exclusive.
        t_end = (raw_times[-1] + pd.Timedelta(hours = 1)).strftime(r'%Y-%m-%dT%H:%M:%S')

        logging.info(f"[{date_tag}] Querying ECMWF/ERA5/HOURLY from {t_start} to {t_end} ({num_t} steps)...")

        era5_col = (ee.ImageCollection('ECMWF/ERA5/HOURLY').filterDate(t_start, t_end).select(['u_component_of_wind_10m', 'v_component_of_wind_10m']))
        multiband_wind_img = era5_col.toBands()

        tmp_tif = PROCESSED_DIR/f'tmp_era5_{date_tag}.tif'

        try:
            self.fetch_geotiff(image = multiband_wind_img, bbox = bbox, scale = 10000, output_path = tmp_tif)

            target_lat = np.linspace(LAT_MIN, LAT_MAX, GRID_RES, dtype = np.float32)
            target_lon = np.linspace(LON_MIN, LON_MAX, GRID_RES, dtype = np.float32)

            with rioxarray.open_rasterio(tmp_tif, masked = True) as da:
                if da.rio.crs and da.rio.crs.to_epsg() != 4326:
                    da = da.rio.reproject('EPSG:4326')

                sampled = da.interp(x = target_lon, y = target_lat, method = 'linear')
                data = np.nan_to_num(sampled.values, nan = 0.0).astype(np.float32)

            # toBands() alternates [t0_u, t0_v, t1_u, t1_v,...] along band dim 0
            u_bands = data[0::2, :, :]
            v_bands = data[1::2, :, :]

            if u_bands.shape[0] < num_t:
                pad_len = num_t - u_bands.shape[0]

                u_bands = np.pad(u_bands, ((0, pad_len), (0, 0), (0, 0)), mode = 'edge')
                v_bands = np.pad(v_bands, ((0, pad_len), (0, 0), (0, 0)), mode = 'edge')
            else:
                u_bands = u_bands[:num_t]
                v_bands = v_bands[:num_t]

            wind_ds = xr.Dataset(
                {'wind_u': (['time', 'y', 'x'], u_bands), 'wind_v': (['time', 'y', 'x'], v_bands)},
                coords = {'time': raw_times.values, 'y': (['y'], target_lat), 'x': (['x'], target_lon)}
            )
            wind_ds.to_netcdf(out_wind_nc, engine = 'h5netcdf')

            logging.info(f"[{date_tag}] Saved 256x256 ERA5 wind cube: {out_wind_nc.name}")

            return out_wind_nc
        finally:
            if tmp_tif.exists():
                tmp_tif.unlink()


if __name__ == '__main__':
    # Standardized [West, South, East, North] bounding box from config.py: [84.0, 18.0, 88.0, 22.0]
    domain_bbox = [LON_MIN - 0.05, LAT_MIN - 0.05, LON_MAX + 0.05, LAT_MAX + 0.05] # 0.05 deg buffer around [84.0, 18.0, 88.0, 22.0] so linear interpolation at the
                                                                                   # 256x256 boundary doesn't hit NaN.

    fetcher = GEEDataFetcher(project_id = GCP_PROJECT_ID)

    try:
        srtm_path = RAW_DIR/'srtm_puri.tif'

        if not srtm_path.exists():
            # Terrestrial topography (SRTM).
            # Mask out <= 0 ocean vals so open H2O is exported as NaN (NoData). Otherwise SRTM's 0m ocean will overwrite ETOPO1's -ve ocean depths in mesh_interpolator.py
            srtm_raw = ee.Image('USGS/SRTMGL1_003') # 30m terrestrial topography for Puri
            srtm_land_only = srtm_raw.updateMask(srtm_raw.gt(0))
            fetcher.fetch_geotiff(image = srtm_land_only, bbox = domain_bbox, scale = 250, output_path = RAW_DIR/'srtm_puri.tif')

        bathy_path = RAW_DIR/'bathymetry_bay_of_bengal.tif'

        if not bathy_path.exists():
            # Ground floor bathymetry
            bathymetry = ee.Image('NOAA/NGDC/ETOPO1').select('bedrock') # Selecting the 'bedrock' band of the ETOPO1 dataset, which represents the ocean floor topography.
            fetcher.fetch_geotiff(image = bathymetry, bbox = domain_bbox, scale = 1800, output_path = RAW_DIR/'bathymetry_bay_of_bengal.tif') # 1800m bathymetry for Bay of
                                                                                                        # Bengal

        unstructured_files = sorted(UNSTRUCTURED_DIR.glob('unstructured_surge_*.nc'))

        if not unstructured_files:
            logging.warning(f"No storm files found in {UNSTRUCTURED_DIR} to pull winds for.")
        else:
            for raw_nc in unstructured_files:
                fetcher.fetch_storm_winds_for_nc(raw_nc, domain_bbox)
    except Exception as e:
        logging.critical(f"Ingestion pipeline halted: {e}")

        sys.exit(1)