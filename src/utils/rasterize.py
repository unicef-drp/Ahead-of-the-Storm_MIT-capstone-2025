import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
import yaml
import os

from src.utils.path_utils import get_project_root


def read_grid_resolution_from_config(config_path=None):
    """
    Reads the grid resolution (in degrees) from the impact_analysis_config.yaml file.
    """
    if config_path is None:
        root = get_project_root()
        config_path = os.path.join(root, 'config', 'impact_analysis_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    # Look for grid resolution in impact_analysis section
    try:
        return config['impact_analysis']['grid']['resolution_degrees']
    except KeyError:
        raise KeyError("Could not find grid resolution in config. Expected at impact_analysis['grid']['resolution_degrees'].")


def rasterize_points_to_tiff(
    geojson_path=None,
    output_tiff_path=None,
    grid_resolution_deg=None,
    config_path=None,
    config_key=None
):
    """
    Rasterizes point data from a GeoJSON file to a grid and saves as a GeoTIFF.

    Args:
        geojson_path (str, optional): Path to the input GeoJSON file with point data. If config_key is provided, this is read from config.
        output_tiff_path (str, optional): Path to save the output GeoTIFF file. If config_key is provided, this is read from config.
        grid_resolution_deg (float, optional): Grid resolution in degrees. If None, read from config.
        config_path (str, optional): Path to config YAML. If None, use default location.
        config_key (str, optional): Key in config YAML to read input/output paths from.
    """
    # Read grid resolution from config if not provided
    if grid_resolution_deg is None:
        grid_resolution_deg = read_grid_resolution_from_config(config_path)

    # If config_key is provided, read paths from config
    if config_key is not None:
        if config_path is None:
            root = get_project_root()
            config_path = os.path.join(root, 'config', 'impact_analysis_config.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if config_key not in config:
            raise KeyError(f"Config key '{config_key}' not found in config file.")
        geojson_path = config[config_key]['input_geojson']
        output_tiff_path = config[config_key]['output_tiff']

    if geojson_path is None or output_tiff_path is None:
        raise ValueError("geojson_path and output_tiff_path must be provided, either as arguments or in config.")

    # Read point data
    gdf = gpd.read_file(geojson_path)
    if gdf.empty:
        raise ValueError(f"No point data found in {geojson_path}")
    if gdf.geometry.geom_type.unique().tolist() != ['Point']:
        raise ValueError("GeoJSON must contain only Point geometries.")

    # Filter out invalid points (e.g., with -1.79769313e+308 coordinates)
    valid_mask = (
        (gdf.geometry.x > -1e6) & (gdf.geometry.x < 1e6) &
        (gdf.geometry.y > -1e6) & (gdf.geometry.y < 1e6)
    )
    num_invalid = (~valid_mask).sum()
    if num_invalid > 0:
        print(f"Warning: {num_invalid} invalid points dropped due to extreme coordinates.")
    gdf = gdf[valid_mask]
    if gdf.empty:
        raise ValueError("All points were invalid after filtering. Check your input data.")

    # Get bounds
    minx, miny, maxx, maxy = gdf.total_bounds

    # Calculate grid size
    x_bins = int(np.ceil((maxx - minx) / grid_resolution_deg))
    y_bins = int(np.ceil((maxy - miny) / grid_resolution_deg))

    # Assign each point to a grid cell
    x_indices = ((gdf.geometry.x - minx) / grid_resolution_deg).astype(int)
    y_indices = ((maxy - gdf.geometry.y) / grid_resolution_deg).astype(int)  # y reversed for raster

    # Create raster grid
    raster = np.zeros((y_bins, x_bins), dtype=np.uint16)
    for x, y in zip(x_indices, y_indices):
        if 0 <= x < x_bins and 0 <= y < y_bins:
            raster[y, x] += 1

    # Define transform (top-left corner)
    transform = from_origin(minx, maxy, grid_resolution_deg, grid_resolution_deg)

    # Write to GeoTIFF
    with rasterio.open(
        output_tiff_path,
        'w',
        driver='GTiff',
        height=raster.shape[0],
        width=raster.shape[1],
        count=1,
        dtype=raster.dtype,
        crs=gdf.crs.to_string() if gdf.crs else 'EPSG:4326',
        transform=transform
    ) as dst:
        dst.write(raster, 1)

    print(f"Rasterized output saved to {output_tiff_path}") 