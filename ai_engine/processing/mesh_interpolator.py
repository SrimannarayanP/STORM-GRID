# mesh_interpolator.py


from pathlib import Path
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

import numpy as np
import xarray as xr

import logging, rioxarray, sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from config import GRID_RES, LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, PROCESSED_DIR, RAW_DIR, TRAINING_TENSORS_DIR, UNSTRUCTURED_DIR


logging.basicConfig(level = logging.INFO, format = "%(asctime)s - [%(levelname)s] - %(message)s")


class TensorInterpolator():
    
    """Converts 1D unstruct-ed ADCIRC node dumps into 256x256 Cartesian training tensors/grids fused with static topo-bathymetry & wind forcing."""

    def __init__(self, resolution = GRID_RES):
        self.resolution = resolution
        self.lon_min, self.lon_max = LON_MIN, LON_MAX
        self.lat_min, self.lat_max = LAT_MIN, LAT_MAX

        # Standardize grid ordering to (Y: Lat, X: Lon) to match train.py (Batch, C, Y, X)
        self.lat_1d = np.linspace(self.lat_min, self.lat_max, resolution, dtype = np.float32)
        self.lon_1d = np.linspace(self.lon_min, self.lon_max, resolution, dtype = np.float32)
        self.grid_lon, self.grid_lat = np.meshgrid(self.lon_1d, self.lat_1d) # Both shapes: (256, 256) -> (Y, X)

        self.topo_grid = self._build_unified_topo_grid()

    def _sample_geotiff_to_grid(self, tif_path):
        """Samples a GeoTIFF directly onto the 256x256 (y, x) target coords."""

        da = rioxarray.open_rasterio(tif_path, masked = True).squeeze()

        if da.rio.crs and da.rio.crs.to_epsg() != 4326:
            da = da.rio.reproject('EPSG:4326')

        sampled = da.interp(x = self.lon_1d, y = self.lat_1d, method = 'linear')

        return sampled.values.astype(np.float32)

    def _build_unified_topo_grid(self):
        """Fuses ETOPO1 ocean bathymetry & SRTM land topography diretly on the 256x256 domain without cropping the ocean to the 1x1 deg land box."""

        bathy_path = RAW_DIR/'bathymetry_bay_of_bengal.tif'
        srtm_path = RAW_DIR/'srtm_puri.tif'
        premerged_files = sorted(PROCESSED_DIR.glob('*.tif'))

        if bathy_path.exists():
            logging.info(f"Sampling ocean bathymetry from {bathy_path.name}...")

            topo = np.nan_to_num(self._sample_geotiff_to_grid(bathy_path), nan = 0.0)

            if srtm_path.exists():
                logging.info(f"Overlaying terrestrial SRTM topography from {srtm_path.name}...")

                land = self._sample_geotiff_to_grid(srtm_path)
                valid_land = ~np.isnan(land)
                topo[valid_land] = land[valid_land]

            return topo.astype(np.float32)

        if premerged_files:
            logging.info(f"Loading pre-merged topo mesh from {premerged_files[0].name}...")

            return np.nan_to_num(self._sample_geotiff_to_grid(premerged_files[0]), nan = 0.0).astype(np.float32)

        logging.warning("No GeoTIFFs found in data/raw or data/processed. Using synthetic Bay of Bengal continental shelf slope (from -2000m to 10m).")

        # Slopes from -2000m in SE to 10m along the NW coast.
        norm_lat = (self.grid_lat - self.lat_min)/(self.lat_max - self.lat_min)

        return (-2000.0*(1.0 - norm_lat) + 10.0*norm_lat).astype(np.float32)

    def _load_wind_forcing(self, date_tag, num_timesteps):
        """Loads processed GRIB2 wind grids if available, else returns 0 tensors."""

        candidates = [PROCESSED_DIR/f'wind_{date_tag}.nc', PROCESSED_DIR/'wind_forcing.nc']

        for wind_path in candidates:
            if wind_path.exists():
                with xr.open_dataset(wind_path, engine = 'h5netcdf') as wds:
                    if 'wind_u' in wds.data_vars and 'wind_v' in wds.data_vars:
                        wu = wds['wind_u'].values.astype(np.float32)
                        wv = wds['wind_v'].values.astype(np.float32)

                        if wu.shape[0] >= num_timesteps:
                            logging.info(f"Fused wind forcing from {wind_path.name}")

                            return wu[:num_timesteps], wv[:num_timesteps]

        shape = (num_timesteps, self.resolution, self.resolution)

        return np.zeros(shape, dtype = np.float32), np.zeros(shape, dtype = np.float32)

    def interpolate_file(self, raw_path, out_path, date_tag):
        logging.info(f"Interpolating {raw_path.name} -> {out_path.name}...")

        with xr.open_dataset(raw_path, engine = 'h5netcdf') as ds:
            if 'time' not in ds.dims:
                raise KeyError(f"{raw_path} lacks a 'time' dim.")

            raw_x = ds['x'].values
            raw_y = ds['y'].values
            time_steps = ds['time'].values
            num_t = len(time_steps)

            mask = ((raw_x >= self.lon_min) & (raw_x <= self.lon_max) & (raw_y >= self.lat_min) & (raw_y <= self.lat_max))

            pts_x = raw_x[mask]
            pts_y = raw_y[mask]

            if len(pts_x) < 100:
                raise ValueError(f"Too few regional nodes ({len(pts_x)}) in bounding box.")

            # Pull (Time, Nodes) arrs
            zeta_all = ds['zeta'].values[:, mask]
            u_all = ds['u-vel'].values[:, mask] if 'u-vel' in ds.data_vars else np.zeros_like(zeta_all)
            v_all = ds['v-vel'].values[:, mask] if 'v-vel' in ds.data_vars else np.zeros_like(zeta_all)

            # Build Delaunay triangulation once & interpolate all (3, Time) slices in a single C call.
            pts = np.column_stack((pts_x, pts_y))
            tri = Delaunay(pts)

            # Stack as (Nodes, 3, Time) so LinearNDInterpolator computes simplex weights only once.
            stacked_node_vals = np.stack([zeta_all.T, u_all.T, v_all.T], axis = 1)
            interpolator = LinearNDInterpolator(tri, stacked_node_vals, fill_value = 0.0)
            # Output shape from interpolator: (Y: 256, X: 256, Var: 3, Time: T)
            interp_grid = np.nan_to_num(interpolator(self.grid_lon, self.grid_lat), nan = 0.0).astype(np.float32)
            # Transpose to (Var: 3, Time: T, Y: 256, X: 256) & unpack
            grid_h_arr, grid_u_arr, grid_v_arr = np.transpose(interp_grid, (2, 3, 0, 1))
            
            wind_u_arr, wind_v_arr = self._load_wind_forcing(date_tag, num_t)

            processed_ds = xr.Dataset(
                {
                    'topo': (['y', 'x'], self.topo_grid),
                    'surge_h': (['time', 'y', 'x'], grid_h_arr),
                    'surge_u': (['time', 'y', 'x'], grid_u_arr),
                    'surge_v': (['time', 'y', 'x'], grid_v_arr),
                    'wind_u': (['time', 'y', 'x'], wind_u_arr),
                    'wind_v': (['time', 'y', 'x'], wind_v_arr)
                },
                coords = {'time': time_steps, 'y': (['y'], self.lat_1d), 'x': (['x'], self.lon_1d)}
            )

            out_path.parent.mkdir(parents = True, exist_ok = True)

            processed_ds.to_netcdf(out_path, engine = 'h5netcdf')

            logging.info(f"Saved complete 6-var training tensor ({num_t} steps): {out_path.name}")


if __name__ == '__main__':
    raw_files = sorted(UNSTRUCTURED_DIR.glob('unstructured_surge_*.nc'))

    if not raw_files:
        logging.error(f"No unstruct-ed files found in {UNSTRUCTURED_DIR}. Run batch_orchestrator.py 1st.")

        sys.exit(1)

    interpolator = TensorInterpolator(resolution = GRID_RES)

    for raw_file in raw_files:
        date_tag = raw_file.stem.replace('unstructured_surge_', '')
        out_file = TRAINING_TENSORS_DIR/f'structured_surge_{date_tag}.nc'
        
        try:
            interpolator.interpolate_file(raw_file, out_file, date_tag)
        except Exception as e:
            logging.error(f"Failed to interpolate {raw_file.name}: {e}")