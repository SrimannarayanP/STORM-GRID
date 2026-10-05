# mesh_builder.py


from rioxarray.merge import merge_arrays
from pathlib import Path

import xarray as xr

import argparse, logging, rioxarray, sys


logging.basicConfig(level = logging.INFO, format = "%(asctime)s - [%(levelname)s] - %(message)s")


class MeshBuilder:
    
    """Fuses bathymetric (ocean) & topographic (land) datasets into a cont hydrodynamic computational mesh."""

    def __init__(self, srtm_path, bathymetry_path):
        self.srtm_path = Path(srtm_path)
        self.bathymetry_path = Path(bathymetry_path)

        if not self.srtm_path.exists() or not self.bathymetry_path.exists():
            raise FileNotFoundError(f"Either SRTM or bathymetry data not found at {self.srtm_path} or {self.bathymetry_path}")

    def build_unified_mesh(self):
        """Loads, resamples, & merges SRTM & bathymetry datasets into a single xarray dataset. This is the computational mesh for hydrodynamic modeling."""

        logging.info(f"Loading topography (SRTM) & bathymetry (bathymetry) datasets from {self.srtm_path} & {self.bathymetry_path}...")

        land = rioxarray.open_rasterio(self.srtm_path) # SRTM topography (land)
        ocean = rioxarray.open_rasterio(self.bathymetry_path) # Bathymetry

        # CRS (Coord Ref Sys) alignment. Both must be in the same projection (e.g. EPSG:4326 for std lat/lon)
        if land.rio.crs != ocean.rio.crs:
            logging.info("CRS mismatch detected. Reprojecting bathymetry to match SRTM...")
            
            ocean = ocean.rio.reproject(land.rio.crs)

        # Spatial resampling. Bathymetry is coarse (~450m). SRTM is finer (~30m). We use bilinear interpolation to mathematically stretch the ocean floor to match the
        # exact pixel grid of the land model.
        logging.info("Resampling bathymetry to match SRTM resolution...")
        
        ocean_resampled = ocean.rio.reproject_match(land, resampling = 5)

        logging.info("Merging SRTM & bathymetry into a unified mesh...")

        unified_mesh = merge_arrays([ocean_resampled, land]) # Merge the 2 datasets. Wherever land exists, we use SRTM. Where it doesn't (NaN), the bathymetry ocean
                                                             # floor shows.

        return unified_mesh

    def export_mesh(self, mesh, output_filename):
        out_path = Path('data/processed')/output_filename

        logging.info(f"Writing cont computational mesh to {out_path}...")

        mesh.rio.to_raster(out_path, tiled = True, compress = 'lzw') # Export as a Cloud-optimized GeoTIFF (COG) for fast, chunked read-access later. This is a std
                                                                     # raster format that preserves geospatial info (CRS, pixel size, etc.)

        logging.info("Mesh gen complete.")


parser = argparse.ArgumentParser(description = "Builds a unified computational mesh from SRTM & bathymetry datasets (topo-bathymetric) mesh.")
parser.add_argument('--land', required = True, help = "Path to SRTM topography GeoTIFF.")
parser.add_argument('--ocean', required = True, help = "Path to bathymetry GeoTIFF.")
parser.add_argument('--output', required = True, help = "Output mesh file name.")

args = parser.parse_args()

try:
    builder = MeshBuilder(args.land, args.ocean)
    mesh = builder.build_unified_mesh()
    builder.export_mesh(mesh, args.output)
except Exception as e:
    logging.critical(f"Mesh build failed: {e}")

    sys.exit(1)