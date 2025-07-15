"""
Rainfall Ensemble Downloader and Processor

This script downloads real-time ECMWF ensemble precipitation forecast data (GRIB2 format),
saves the raw data, subsets it to the Nicaragua region, and saves the processed data as NetCDF.
All paths and parameters are loaded from config/rainfall_config.yaml.
"""
import os
import yaml
import xarray as xr
from ecmwf.opendata import Client
from src.utils.path_utils import ensure_directory


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def download_rainfall_ensemble(config):
    """
    Download ECMWF ensemble precipitation forecast (GRIB2) and save to raw folder.
    Returns the path to the downloaded file.
    """
    client = Client(source="ecmwf")
    forecast_hour = config['forecast_hour']
    run_time = config['run_time']
    raw_dir = config['raw_data_dir']
    ensure_directory(raw_dir)
    filename = f"precip_ens_{forecast_hour}h.grib2"
    raw_path = os.path.join(raw_dir, filename)
    result = client.retrieve(
        time=run_time,
        stream="enfo",
        type="pf",
        step=forecast_hour,
        param="tp",
        target=raw_path
    )
    return raw_path

def subset_and_save_processed(raw_path, config):
    """
    Subset the GRIB2 file to the region and save as NetCDF in processed folder.
    """
    ds = xr.open_dataset(raw_path, engine='cfgrib')
    precip_data = ds['tp'] * 1000  # meters to mm
    lat_min = config['region']['lat_min']
    lat_max = config['region']['lat_max']
    lon_min = config['region']['lon_min']
    lon_max = config['region']['lon_max']
    precip_region = precip_data.sel(
        latitude=slice(lat_max, lat_min),
        longitude=slice(lon_min, lon_max)
    )
    processed_dir = config['processed_data_dir']
    ensure_directory(processed_dir)
    processed_path = os.path.join(processed_dir, 'rainfall_ensemble.nc')
    precip_region.to_netcdf(processed_path)
    return processed_path

def main(config_path='config/rainfall_config.yaml'):
    config = load_config(config_path)
    raw_path = download_rainfall_ensemble(config)
    processed_path = subset_and_save_processed(raw_path, config)
    print(f"Raw rainfall data saved to: {raw_path}")
    print(f"Processed rainfall data saved to: {processed_path}")

if __name__ == "__main__":
    main() 