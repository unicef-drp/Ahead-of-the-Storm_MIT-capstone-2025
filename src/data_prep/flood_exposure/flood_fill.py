import numpy as np
from scipy.ndimage import distance_transform_edt, label

try:
    from skimage.segmentation import watershed
except ImportError:
    watershed = None

# --- FLOOD FILL AND CONNECTIVITY FOR FLOOD MODELING ---


def flood_fill_algorithm(
    elevation,
    hand,
    slope,
    twi,
    flood_depth,
    river_network,
    exceeds_rp,
    SLOPE_WEIGHT=0.3,
    TWI_WEIGHT=0.4,
    DEPTH_TO_INUNDATION_COEFF=1.5,
):
    """
    Advanced flood fill algorithm that combines HAND, slope, TWI, and distance to river.
    Returns a flood extent raster (depth in each cell).
    """
    print(f"[Flood] Running flood fill algorithm for RP={exceeds_rp}...")
    flood_extent = np.zeros_like(elevation, dtype=np.float32)
    base_hand_threshold = 3 + (exceeds_rp * 0.8)
    max_distance = 40 + (exceeds_rp * 10)
    slope_threshold = 5 + (exceeds_rp * 2)
    distance_from_rivers = distance_transform_edt(~river_network)
    suitability = np.zeros_like(elevation, dtype=np.float32)
    valid_base = (~np.isnan(elevation)) & (flood_depth > 0)
    # Combine multiple suitability factors
    hand_factor = np.where(
        valid_base & (~np.isnan(hand)),
        np.maximum(0, 1 - (hand / base_hand_threshold)),
        0,
    )
    distance_factor = np.where(
        valid_base, np.maximum(0, 1 - (distance_from_rivers / max_distance)), 0
    )
    slope_factor = np.where(
        valid_base & (~np.isnan(slope)), np.maximum(0, 1 - (slope / slope_threshold)), 0
    )
    if twi is not None and not np.all(np.isnan(twi)):
        twi_normalized = np.where(
            ~np.isnan(twi),
            (twi - np.nanmin(twi)) / (np.nanmax(twi) - np.nanmin(twi)),
            0,
        )
        twi_factor = np.where(valid_base, twi_normalized, 0)
    else:
        twi_factor = np.ones_like(elevation) * 0.5
    suitability = (
        hand_factor * 0.3
        + distance_factor * 0.25
        + slope_factor * SLOPE_WEIGHT
        + twi_factor * TWI_WEIGHT
    )
    min_suitability = 0.3
    flood_suitable = (suitability >= min_suitability) & valid_base
    base_flood_depth = flood_depth * DEPTH_TO_INUNDATION_COEFF
    flood_extent = np.where(flood_suitable, base_flood_depth * suitability, 0)
    # Enhance connectivity using watershed
    flood_extent = flood_connectivity(flood_extent, river_network, elevation)
    print(f"[Flood] Flood fill complete. Flooded cells: {np.sum(flood_extent > 0)}")
    return flood_extent


def flood_connectivity(flood_extent, river_network, elevation):
    """
    Enhance flood connectivity using a watershed-based approach seeded from river network.
    Returns an improved flood extent raster.
    """
    print("[Flood] Enhancing flood connectivity with watershed...")
    river_flood_seeds = river_network & (flood_extent > 0)
    if not np.any(river_flood_seeds):
        print("[Flood] No river flood seeds found. Skipping connectivity enhancement.")
        return flood_extent
    labeled_seeds, num_labels = label(river_flood_seeds)
    if num_labels == 0 or watershed is None:
        print("[Flood] No labeled seeds or watershed unavailable. Skipping.")
        return flood_extent
    elevation_for_watershed = np.where(
        np.isnan(elevation), np.nanmax(elevation) + 1000, elevation
    )
    watershed_landscape = -elevation_for_watershed
    try:
        watershed_result = watershed(
            watershed_landscape, labeled_seeds, mask=(flood_extent > 0)
        )
        enhanced_flood = np.where(
            watershed_result > 0,
            np.maximum(flood_extent, flood_extent.mean() * 0.3),
            flood_extent,
        )
        print("[Flood] Watershed connectivity enhancement complete.")
        return enhanced_flood
    except Exception:
        print(
            "[Flood][ERROR] Watershed enhancement failed. Returning original flood extent."
        )
        return flood_extent
