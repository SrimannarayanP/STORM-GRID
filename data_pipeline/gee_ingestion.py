# gee_ingestion.py


from pathlib import Path

import ee, io, logging, requests, sys, zipfile


logging.basicConfig(level = logging.INFO, format = "%(asctime)s - [%(levelname)s] - %(message)s")


class GEEDataFetcher:

    """Automates the extraction of localized GeoTIFFs from Google Earth Engine (GEE)."""

    def __init__(self, project_id):
        try:
            # Inits Earth Engine API using local Google creds.
            ee.Initialize(project = project_id)

            logging.info("Earth Engine API authenticated & connected.")
        except Exception as e:
            logging.error(f"Failed to connect to GEE. You must run 'earthengine authenticate' in your terminal 1st.")
            
            raise e

    def fetch_geotiff(self, image, bbox, scale, output_path):
        """Sends a spatial query to Google's servers & downloads the clipped result."""
        # Define the spatial boundary: [West, South, East, North]
        region = ee.Geometry.BBox(*bbox)

        logging.info(f"Requesting server-side crop for {output_path}...")

        # Req the computation. getDownloadURL() forces Google to process the crop at the req-d 'scale' (resolution in m per pixel) & package the resulting matrix in
        # GeoTIFF format.
        url = image.getDownloadURL({'region': region, 'scale': scale, 'format': 'GEO_TIFF', 'crs': 'EPSG:4326'}) # EPSG:4326 forces std lat/lon projection for
                                                                                                                 # consistency across datasets.

        logging.info(f"Streaming bin data...")

        # Stream the res into memory. Google returns a ZIP file. We extract the internal .tif directly into our raw data folder without saving the ZIP to disk.
        res = requests.get(url)

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

        logging.info(f"Successfully secured: {output_path}")


GCP_PROJECT_ID = 'cyclone-forecaster-509919'

Path('data/raw').mkdir(parents = True, exist_ok = True)

fetcher = GEEDataFetcher(project_id = GCP_PROJECT_ID)

try:
    srtm = ee.Image('USGS/SRTMGL1_003') # 30m terrestrial topography for Puri
    fetcher.fetch_geotiff(image = srtm, bbox = [85.0, 19.0, 86.0, 20.0], scale = 30, output_path = 'data/raw/srtm_puri.tif')

    bathymetry = ee.Image('NOAA/NGDC/ETOPO1').select('bedrock') # Selecting the 'bedrock' band of the ETOPO1 dataset, which represents the ocean floor topography.
    fetcher.fetch_geotiff(image = bathymetry, bbox = [84.5, 17.5, 88.0, 21.0], scale = 1800, output_path = 'data/raw/bathymetry_bay_of_bengal.tif') # 1800m bathymetry
                                                                                                # for Bay of Bengal
except Exception as e:
    logging.critical(f"Ingestion pipeline halted: {e}")

    sys.exit(1)