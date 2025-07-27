import sys
import os
import geopandas as gpd
import numpy as np
from src.data_prep.flood_exposure import (
    load_elevation_data,
    calculate_slope,
    calculate_flow_direction_d8,
    calculate_flow_accumulation,
    calculate_topographic_wetness_index,
    find_optimal_drainage_threshold,
    calculate_hand_efficient,
    create_river_network_raster,
    flood_fill_algorithm,
    load_and_reproject_all_flood_maps,
)
from src.utils.config_utils import load_config
import yaml


def main(config_path="config/flood_config.yaml"):
    print("[Flood] Loading configuration...")
    config = load_config(config_path)
    flood_cfg = config["flood"]
    model_cfg = config["model"]
    dem_path = flood_cfg["dem_file"]
    discharge_path = flood_cfg["discharge_file"]
    output_dir = flood_cfg["output_directory"]
    riskmap_dir = flood_cfg["riskmap_dir"]
    riskmap_files = flood_cfg["riskmap_files"]
    # Build flood maps dict (RP:int -> path)
    flood_maps_dict = {}
    for k, v in riskmap_files.items():
        rp = int(k.replace("h", ""))
        flood_maps_dict[rp] = os.path.join(riskmap_dir, v)
    # Model params
    DOWNSAMPLE_FACTOR = model_cfg["downsample_factor"]
    SLOPE_WEIGHT = model_cfg["slope_weight"]
    TWI_WEIGHT = model_cfg["twi_weight"]
    DEPTH_TO_INUNDATION_COEFF = model_cfg["depth_to_inundation_coeff"]
    print(f"[Flood] DEM: {dem_path}")
    print(f"[Flood] Discharge: {discharge_path}")
    print(f"[Flood] Output dir: {output_dir}")
    print(f"[Flood] Flood maps: {list(flood_maps_dict.values())}")
    # Check files
    if not os.path.exists(dem_path):
        print(f"[Flood][ERROR] DEM file not found: {dem_path}")
        sys.exit(1)
    if not os.path.exists(discharge_path):
        print(f"[Flood][ERROR] Discharge file not found: {discharge_path}")
        sys.exit(1)
    for k, v in flood_maps_dict.items():
        if not os.path.exists(v):
            print(f"[Flood][ERROR] Flood map not found: {v}")
            sys.exit(1)
    # Terrain analysis
    elevation, transform, crs, bounds = load_elevation_data(DOWNSAMPLE_FACTOR, dem_path)
    slope = calculate_slope(elevation)
    flow_dir = calculate_flow_direction_d8(elevation)
    flow_acc = calculate_flow_accumulation(flow_dir, elevation)
    twi = calculate_topographic_wetness_index(elevation, flow_acc, slope)
    threshold = find_optimal_drainage_threshold(flow_acc)
    hand, drainage_mask = calculate_hand_efficient(elevation, flow_acc, threshold)
    # Load flood maps
    flood_maps = load_and_reproject_all_flood_maps(
        elevation.shape, transform, crs, flood_maps_dict
    )
    # Load discharge data
    discharge_gdf = gpd.read_file(discharge_path)
    # Example: process one return period (25)
    rp = 25
    river_network = create_river_network_raster(
        discharge_gdf, elevation.shape, transform, crs
    )
    flood_map = flood_maps[rp]
    flood_extent = flood_fill_algorithm(
        elevation,
        hand,
        slope,
        twi,
        flood_map,
        river_network,
        rp,
        SLOPE_WEIGHT=SLOPE_WEIGHT,
        TWI_WEIGHT=TWI_WEIGHT,
        DEPTH_TO_INUNDATION_COEFF=DEPTH_TO_INUNDATION_COEFF,
    )
    # Save result
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "test_flood_extent.tif")
    print(f"[Flood] Saving flood extent raster to {out_path} ...")
    import rasterio

    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=flood_extent.shape[0],
        width=flood_extent.shape[1],
        count=1,
        dtype=flood_extent.dtype,
        crs=crs,
        transform=transform,
        compress="lzw",
    ) as dst:
        dst.write(flood_extent, 1)
    print(f"[Flood] Done! Flood extent saved to {out_path}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/flood_config.yaml"
    main(config_path)
