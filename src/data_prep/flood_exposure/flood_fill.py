import numpy as np
from scipy.ndimage import distance_transform_edt, label

try:
    from skimage.segmentation import watershed
except ImportError:
    watershed = None

# --- FLOOD FILL AND CONNECTIVITY FOR FLOOD MODELING ---


def flood_fill_algorithm(elevation, hand, slope, twi, flood_depth,
                        river_network, exceeds_rp, resolution=90):
    """
    1. HAND thresholds based on Nobre et al. (2011)
    2. Distance limits based on Özay & Orhan (2023)
    3. Slope limits based on Özay & Orhan (2023)
    4. Science-based factor weighting based on Özay & Orhan (2023)
    """
    print(f"[Flood] Running flood fill algorithm for RP={exceeds_rp}...")

    # Initialize flood extent
    flood_extent = np.zeros_like(elevation, dtype=np.float32)

    # 1. HAND Threshold - Nobre et al. (2011)
    base_hand_threshold = _calculate_hand_threshold(exceeds_rp)

    # 2. Distance Threshold - Özay & Orhan (2023)
    max_distance_m = _calculate_distance_threshold(exceeds_rp)

    # 3. Slope Threshold - Özay & Orhan (2023)
    slope_threshold = _calculate_slope_threshold(exceeds_rp)

    print(f"      HAND ≤ {base_hand_threshold:.1f}m (Nobre et al. 2011)")
    print(f"      Distance ≤ {max_distance_m/1000:.1f}km (Özay & Orhan 2023)")
    print(f"      Slope ≤ {slope_threshold:.1f}° (Özay & Orhan 2023)")

    # Calculate distance from river network (IN METERS)
    distance_from_rivers_m = distance_transform_edt(~river_network) * resolution

    # Create suitability map (Özay & Orhan 2023)
    suitability = _calculate_suitability(
        elevation, hand, slope, twi, flood_depth, distance_from_rivers_m,
        base_hand_threshold, max_distance_m, slope_threshold
    )

    if suitability is None:
        print("      No valid areas for suitability calculation")
        return flood_extent

    # Apply minimum thresholds
    min_suitability = 0.3
    flood_suitable = suitability >= min_suitability

    if not np.any(flood_suitable):
        print("      No suitable flood areas found")
        return flood_extent

    flood_extent = np.where(flood_suitable, flood_depth * suitability, 0)

    # Ensure connectivity
    flood_extent = flood_connectivity(flood_extent, river_network, elevation)

    # Ensure hydraulic connectivity to river network
    flood_pixels = np.sum(flood_extent > 0)
    flood_area_km2 = flood_pixels * (resolution/1000)**2
    
    # Debug: Show flood depth statistics
    if flood_pixels > 0:
        flood_depths = flood_extent[flood_extent > 0]
        print(f"      Flood depth stats: min={flood_depths.min():.3f}m, max={flood_depths.max():.3f}m, mean={flood_depths.mean():.3f}m")
        print(f"      Flood depths > 0.1m: {np.sum(flood_depths > 0.1)} pixels")
        print(f"      Flood depths > 0.5m: {np.sum(flood_depths > 0.5)} pixels")
        print(f"      Flood depths > 1.0m: {np.sum(flood_depths > 1.0)} pixels")
    
    print(f"      Flood area: {flood_area_km2:.1f} km² ({flood_pixels:,} pixels)")

    return flood_extent


def _calculate_hand_threshold(exceeds_rp):
    """
    Calculate HAND threshold based on flood zone classification.

    HAND-based flood zone classification:
    - HAND < 5m: Regularly flooded areas (high flood susceptibility)
    - HAND 5-15m: Intermediate zones with occasional flooding
    - HAND > 15m: Dry uplands, generally not flooded (low flood susceptibility)

    The threshold increases logarithmically with return period to account for
    more extreme floods reaching higher elevations.

    References:
    Nobre et al. (2011). Height Above the Nearest Drainage – a hydrologically relevant
    new terrain model. Journal of Hydrology, Vol. 404, Issues 1–2, Pages 13-29.
    """

    # Base threshold: transition from regularly flooded to intermediate zone
    base_threshold = 5.0  # meters

    # Maximum threshold: transition to dry uplands (even extreme floods rarely exceed this)
    max_threshold = 15.0  # meters

    # Scale logarithmically with return period
    if exceeds_rp <= 2:
        # Small floods stay in regularly flooded zone
        return base_threshold
    elif exceeds_rp >= 500:
        return max_threshold
    else:
        log_factor = np.log10(exceeds_rp) / np.log10(500)  # Scale to 500-year max
        hand_threshold = base_threshold + (max_threshold - base_threshold) * log_factor
        return np.clip(hand_threshold, base_threshold, max_threshold)


def _calculate_distance_threshold(exceeds_rp):
    """
    Calculate distance-to-river threshold based on flood susceptibility research.

    Distance classifications derived from GIS-based multi-criteria decision analysis
    using best-worst and logistic regression methods (Özay & Orhan, 2023).

    Classifications:
    - 0-200m: Highest flood susceptibility
    - 200-500m: High susceptibility
    - 500-1000m: Medium susceptibility
    - 1000-1500m: Low susceptibility
    - >1500m: Very low susceptibility

    References:
    Özay, B., & Orhan, O. (2023). Flood susceptibility mapping by best–worst and
    logistic regression methods in Mersin, Turkey. Environmental Science and
    Pollution Research, 30(15), 45151-45170.
    """
    base_distance = 200.0   # meters
    max_distance = 1500.0   # meters

    if exceeds_rp <= 2:
        return base_distance
    else:
        log_factor = np.log10(exceeds_rp) / np.log10(1000)
        distance_threshold = base_distance + (max_distance - base_distance) * log_factor
        return np.clip(distance_threshold, base_distance, max_distance)


def _calculate_slope_threshold(exceeds_rp):
    """
    Calculate slope threshold based on topographic factor analysis in flood modeling.

    Slope classifications derived from geomorphological flood susceptibility analysis
    using analytical hierarchy process (Özay & Orhan, 2023).

    Classifications (degrees):
    - 0-5°: Highest flood susceptibility
    - 5-10°: High susceptibility
    - 10-20°: Medium susceptibility
    - 20-35°: Low susceptibility
    - >35°: Very low susceptibility

    References:
    Özay, B., & Orhan, O. (2023). Flood susceptibility mapping by best–worst and
    logistic regression methods in Mersin, Turkey. Environmental Science and
    Pollution Research, 30(15), 45151-45170.
    """
    base_slope = 5.0    # degrees
    max_slope = 35.0    # degrees

    if exceeds_rp <= 2:
        return base_slope
    else:
        linear_factor = min(exceeds_rp / 500.0, 1.0)
        slope_threshold = base_slope + (max_slope - base_slope) * linear_factor
        return np.clip(slope_threshold, base_slope, max_slope)


def _calculate_suitability(elevation, hand, slope, twi, flood_depth,
                          distance_from_rivers_m, hand_threshold,
                          distance_threshold, slope_threshold):
    """
    Calculate flood suitability using multi-criteria decision analysis weights.

    Factor weights derived from best-worst method and analytical hierarchy process
    applied to flood susceptibility mapping (Özay & Orhan, 2023). Weights normalized
    for available parameters:

    - Elevation: 48.9% (most important factor)
    - Slope: 28.5% (second most important)
    - Distance to river: 20.4% (third most important)
    - Topographic wetness index: 2.3% (least important)

    The weighting scheme reflects the relative importance of each factor in determining
    flood susceptibility based on empirical analysis of historical flood events.

    References:
    Özay, B., & Orhan, O. (2023). Flood susceptibility mapping by best–worst and
    logistic regression methods in Mersin, Turkey. Environmental Science and
    Pollution Research, 30(15), 45151-45170.
    """

    # Valid areas: not NaN, has potential flood depth
    valid_base = (~np.isnan(elevation)) & (flood_depth > 0)

    if not np.any(valid_base):
        return None

    # Factor 1: HAND suitability (proxy for elevation effects)
    hand_factor = np.where(valid_base & (~np.isnan(hand)),
                          np.maximum(0, 1 - (hand / hand_threshold)), 0)

    # Factor 2: Distance suitability
    distance_factor = np.where(valid_base,
                              np.maximum(0, 1 - (distance_from_rivers_m / distance_threshold)), 0)

    # Factor 3: Slope suitability
    slope_factor = np.where(valid_base & (~np.isnan(slope)),
                           np.maximum(0, 1 - (slope / slope_threshold)), 0)

    # Factor 4: TWI suitability
    if twi is not None and not np.all(np.isnan(twi)):
        twi_valid = twi[~np.isnan(twi)]
        if len(twi_valid) > 0:
            twi_min, twi_max = np.percentile(twi_valid, [5, 95])
            twi_factor = np.where(valid_base & (~np.isnan(twi)),
                                 np.clip((twi - twi_min) / (twi_max - twi_min), 0, 1), 0)
        else:
            twi_factor = np.ones_like(elevation) * 0.5
    else:
        twi_factor = np.ones_like(elevation) * 0.5

    # Multi-criteria decision analysis weights (Özay & Orhan, 2023)
    # Normalized for available parameters
    ELEVATION_WEIGHT = 0.489    # 48.9% - Most important factor
    SLOPE_WEIGHT = 0.285        # 28.5% - Second most important
    DISTANCE_WEIGHT = 0.204     # 20.4% - Third most important
    TWI_WEIGHT = 0.023          # 2.3% - Least important

    # Verify weights sum to 1.0
    total_weight = ELEVATION_WEIGHT + SLOPE_WEIGHT + DISTANCE_WEIGHT + TWI_WEIGHT
    if abs(total_weight - 1.0) > 0.001:
        print(f"Warning: Weights sum to {total_weight}, normalizing...")
        ELEVATION_WEIGHT /= total_weight
        SLOPE_WEIGHT /= total_weight
        DISTANCE_WEIGHT /= total_weight
        TWI_WEIGHT /= total_weight

    # Combine factors with empirically-derived weights
    suitability = (
        hand_factor * ELEVATION_WEIGHT +     # 48.9% - Most important
        slope_factor * SLOPE_WEIGHT +        # 28.5% - Second most important
        distance_factor * DISTANCE_WEIGHT +  # 20.4% - Third most important
        twi_factor * TWI_WEIGHT             # 2.3% - Least important
    )

    return suitability


def flood_connectivity(flood_extent, river_network, elevation):
    """Flood connectivity using watershed-based approach"""

    # Create seeds from river network
    river_flood_seeds = river_network & (flood_extent > 0)

    if not np.any(river_flood_seeds):
        return flood_extent

    # Label connected components
    labeled_seeds, num_labels = label(river_flood_seeds)

    if num_labels == 0:
        return flood_extent

    # Create watershed landscape
    elevation_for_watershed = np.where(np.isnan(elevation),
                                      np.nanmax(elevation) + 1000, elevation)
    watershed_landscape = -elevation_for_watershed

    # Apply watershed connectivity
    try:
        if watershed is None:
            print("Warning: Watershed module not available, skipping connectivity enhancement")
            return flood_extent
            
        watershed_result = watershed(watershed_landscape, labeled_seeds,
                                   mask=(flood_extent > 0))

        # Maintain connectivity
        flood = np.where(watershed_result > 0,
                        np.maximum(flood_extent, flood_extent.mean() * 0.3),
                        flood_extent)
        return flood

    except Exception as e:
        print(f"Warning: Watershed connectivity failed: {e}")
        return flood_extent
