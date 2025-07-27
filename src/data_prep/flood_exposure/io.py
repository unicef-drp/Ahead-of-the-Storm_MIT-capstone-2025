import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

# --- FLOOD MAP LOADING AND REPROJECTION FOR FLOOD MODELING ---


def load_and_reproject_all_flood_maps(
    elevation_shape, elevation_transform, elevation_crs, flood_map_filenames
):
    print("[Flood] Loading and reprojecting flood risk maps...")
    flood_maps = {}
    for rp, filename in flood_map_filenames.items():
        try:
            print(f"[Flood] Loading flood map for RP={rp}: {filename}")
            with rasterio.open(filename) as src:
                flood_data = src.read(1)
                src_crs = src.crs if src.crs is not None else "EPSG:4326"
                # Clean up invalid values
                flood_data = np.where(flood_data < 0, 0, flood_data)
                flood_data = np.where(flood_data > 2000, 0, flood_data)
                reprojected = np.zeros(elevation_shape, dtype=np.float32)
                reproject(
                    source=flood_data,
                    destination=reprojected,
                    src_transform=src.transform,
                    src_crs=src_crs,
                    dst_transform=elevation_transform,
                    dst_crs=elevation_crs,
                    resampling=Resampling.bilinear,
                )
                flood_maps[rp] = reprojected
                print(
                    f"[Flood] Flood map for RP={rp} reprojected. Nonzero cells: {np.sum(reprojected > 0)}"
                )
        except Exception:
            print(f"[Flood][ERROR] Failed to load or reproject flood map: {filename}")
            continue
    print("[Flood] All flood risk maps loaded and reprojected.")
    return flood_maps
