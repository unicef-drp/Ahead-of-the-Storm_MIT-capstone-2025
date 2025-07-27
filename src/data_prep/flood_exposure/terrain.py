import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt

# --- TERRAIN ANALYSIS HELPERS FOR FLOOD MODELING ---


def load_elevation_data(downsample_factor, dem_path):
    print("[Flood] Loading DEM from:", dem_path)
    with rasterio.open(dem_path) as src:
        if downsample_factor == 1:
            elevation = src.read(1)
            transform = src.transform
        else:
            elevation = src.read(1)[::downsample_factor, ::downsample_factor]
            transform = rasterio.Affine(
                src.transform[0] * downsample_factor,
                src.transform[1],
                src.transform[2],
                src.transform[3],
                src.transform[4] * downsample_factor,
                src.transform[5],
            )
        crs = src.crs
        bounds = src.bounds
        # Clean up nodata and invalid values
        elevation = np.where(elevation == src.nodata, np.nan, elevation)
        elevation = np.where(elevation == -32768, np.nan, elevation)
        elevation = np.where(elevation < -100, np.nan, elevation)
    print(f"[Flood] DEM loaded. Shape: {elevation.shape}, CRS: {crs}")
    return elevation, transform, crs, bounds


def calculate_slope(elevation, resolution=90):
    print("[Flood] Calculating slope...")
    grad_y, grad_x = np.gradient(elevation)
    slope_radians = np.arctan(np.sqrt(grad_x**2 + grad_y**2) / resolution)
    slope_degrees = np.degrees(slope_radians)
    slope_degrees = np.where(np.isnan(elevation), np.nan, slope_degrees)
    print("[Flood] Slope calculation complete.")
    return slope_degrees


def calculate_topographic_wetness_index(
    elevation, flow_acc, slope_degrees, resolution=90
):
    print("[Flood] Calculating Topographic Wetness Index (TWI)...")
    slope_radians = np.radians(slope_degrees + 0.001)
    specific_catchment_area = flow_acc * resolution
    twi = np.log(specific_catchment_area / np.tan(slope_radians))
    twi = np.where(np.isnan(elevation), np.nan, twi)
    twi = np.where(np.isinf(twi), np.nan, twi)
    print("[Flood] TWI calculation complete.")
    return twi


def calculate_flow_direction_d8(elevation):
    print("[Flood] Calculating D8 flow direction...")
    rows, cols = elevation.shape
    flow_dir = np.zeros_like(elevation, dtype=np.uint8)
    directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (1, 1)]
    direction_codes = [1, 2, 4, 8, 16, 32, 64, 128]
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if np.isnan(elevation[i, j]):
                continue
            center_elev = elevation[i, j]
            max_slope = -999
            best_direction = 0
            for k, (di, dj) in enumerate(directions):
                ni, nj = i + di, j + dj
                if not np.isnan(elevation[ni, nj]):
                    slope = center_elev - elevation[ni, nj]
                    if slope > max_slope:
                        max_slope = slope
                        best_direction = direction_codes[k]
            flow_dir[i, j] = best_direction
    print("[Flood] D8 flow direction calculation complete.")
    return flow_dir


def calculate_flow_accumulation(flow_dir, elevation):
    print("[Flood] Calculating flow accumulation...")
    rows, cols = flow_dir.shape
    flow_acc = np.ones_like(flow_dir, dtype=np.float32)
    direction_map = {
        1: (0, 1),
        2: (1, 1),
        4: (1, 0),
        8: (1, -1),
        16: (0, -1),
        32: (-1, -1),
        64: (-1, 0),
        128: (-1, 1),
    }
    for iteration in range(5):
        new_flow_acc = flow_acc.copy()
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if flow_dir[i, j] == 0 or np.isnan(elevation[i, j]):
                    continue
                total_inflow = 1
                for source_dir, (di, dj) in direction_map.items():
                    si, sj = i - di, j - dj
                    if (
                        0 <= si < rows
                        and 0 <= sj < cols
                        and flow_dir[si, sj] == source_dir
                        and not np.isnan(elevation[si, sj])
                    ):
                        total_inflow += flow_acc[si, sj]
                new_flow_acc[i, j] = total_inflow
        flow_acc = new_flow_acc
    print("[Flood] Flow accumulation calculation complete.")
    return flow_acc


def find_optimal_drainage_threshold(flow_acc):
    print("[Flood] Finding optimal drainage threshold...")
    flow_acc_stats = flow_acc[~np.isnan(flow_acc)]
    test_percentiles = [85, 90, 92, 95, 97, 98, 99]
    best_threshold = None
    for percentile in test_percentiles:
        threshold = np.percentile(flow_acc_stats, percentile)
        drainage_pixels = np.sum(flow_acc > threshold)
        percentage = drainage_pixels / flow_acc.size * 100
        if 0.5 <= percentage <= 5.0 and best_threshold is None:
            best_threshold = threshold
    if best_threshold is None:
        best_threshold = np.percentile(flow_acc_stats, 90)
    print(f"[Flood] Optimal drainage threshold: {best_threshold}")
    return best_threshold


def calculate_hand_efficient(elevation, flow_acc, threshold):
    print("[Flood] Calculating HAND (Height Above Nearest Drainage)...")
    drainage_mask = flow_acc > threshold
    distance_to_drainage = distance_transform_edt(~drainage_mask)
    hand = np.full_like(elevation, np.nan)
    hand[drainage_mask] = 0
    non_drainage_mask = ~drainage_mask & ~np.isnan(elevation)
    non_drainage_indices = np.where(non_drainage_mask)
    rows, cols = elevation.shape
    batch_size = 20000
    total_batches = (len(non_drainage_indices[0]) + batch_size - 1) // batch_size
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(non_drainage_indices[0]))
        for idx in range(start_idx, end_idx):
            i, j = non_drainage_indices[0][idx], non_drainage_indices[1][idx]
            dist = int(distance_to_drainage[i, j])
            search_radius = min(dist + 2, 10)
            min_drainage_elev = np.inf
            for di in range(-search_radius, search_radius + 1):
                for dj in range(-search_radius, search_radius + 1):
                    ni, nj = i + di, j + dj
                    if (
                        0 <= ni < rows
                        and 0 <= nj < cols
                        and drainage_mask[ni, nj]
                        and not np.isnan(elevation[ni, nj])
                    ):
                        min_drainage_elev = min(min_drainage_elev, elevation[ni, nj])
            if min_drainage_elev != np.inf:
                hand[i, j] = max(0, elevation[i, j] - min_drainage_elev)
    print("[Flood] HAND calculation complete.")
    return hand, drainage_mask
