import math
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid
from scipy.interpolate import splprep, splev
import requests
from shapely.geometry import shape

__all__ = [
    "wind_quadrant_polygon",
    "bspline_smooth",
    "roundcorner_smooth",
    "superspline_smooth",
    "compute_smoothed_wind_region",
]


def wind_quadrant_polygon(lat, lon, r_ne, r_se, r_sw, r_nw):
    """
    Create a rectangular wind region polygon from quadrant radii (in km).
    Returns a Shapely Polygon in lon/lat degrees.

    Parameters:
        lat, lon: Center point of the storm (degrees)
        r_ne, r_se, r_sw, r_nw: Radii in km for each quadrant (NE, SE, SW, NW)
    """

    def km_to_deg_lat(km):
        # Convert kilometers to degrees latitude
        return km / 110.574

    def km_to_deg_lon(km, lat):
        # Convert kilometers to degrees longitude at a given latitude
        return km / (111.320 * np.cos(np.radians(lat)))

    radii = [r_ne, r_se, r_sw, r_nw]
    # If all radii are zero, NaN, or infinite, return None
    if all(r <= 0 or math.isnan(r) or math.isinf(r) for r in radii):
        return None

    def radius_to_deg(r_km, lat_center):
        # Convert a radius in km to (delta_lat, delta_lon) in degrees
        if r_km > 0 and not math.isnan(r_km) and not math.isinf(r_km):
            return km_to_deg_lat(r_km), km_to_deg_lon(r_km, lat_center)
        else:
            return 0.0, 0.0

    # Convert each quadrant's radius to degree offsets
    r_ne_dlat, r_ne_dlon = radius_to_deg(r_ne, lat)
    r_se_dlat, r_se_dlon = radius_to_deg(r_se, lat)
    r_sw_dlat, r_sw_dlon = radius_to_deg(r_sw, lat)
    r_nw_dlat, r_nw_dlon = radius_to_deg(r_nw, lat)
    # Compute the corner points of the polygon
    ne_lat, ne_lon = lat + r_ne_dlat, lon + r_ne_dlon
    se_lat, se_lon = lat - r_se_dlat, lon + r_se_dlon
    sw_lat, sw_lon = lat - r_sw_dlat, lon - r_sw_dlon
    nw_lat, nw_lon = lat + r_nw_dlat, lon - r_nw_dlon
    coords = [
        (ne_lon, ne_lat),
        (se_lon, se_lat),
        (sw_lon, sw_lat),
        (nw_lon, nw_lat),
        (ne_lon, ne_lat),  # Close the polygon
    ]
    # Filter out any invalid coordinates (NaN or inf)
    valid_coords = [
        c
        for c in coords
        if not (
            math.isnan(c[0]) or math.isnan(c[1]) or math.isinf(c[0]) or math.isinf(c[1])
        )
    ]
    # Need at least 4 valid points to form a polygon
    if len(valid_coords) < 4:
        return None
    # Ensure the polygon is closed
    if valid_coords[0] != valid_coords[-1]:
        valid_coords.append(valid_coords[0])
    try:
        return Polygon(valid_coords)
    except Exception as e:
        print(f"Warning: Could not create wind polygon: {e}")
        return None


def bspline_smooth(coords, smoothing_factor=0, num_points=200):
    """
    Smooth a closed polygon using B-spline interpolation.
    Returns a list of (x, y) tuples for the smoothed polygon.

    Parameters:
        coords: List of (x, y) tuples (should be closed)
        smoothing_factor: Spline smoothing parameter (0 = interpolate through all points)
        num_points: Number of points in the output smoothed polygon
    """
    coords = np.array(coords)
    # Remove duplicate closing point for periodic spline
    if np.allclose(coords[0], coords[-1]):
        coords = coords[:-1]
    
    # Check if we have enough points for spline fitting
    if len(coords) < 3:
        # Return original coordinates if too few points
        return list(coords)
    
    x, y = coords[:, 0], coords[:, 1]
    
    try:
        # Fit a periodic B-spline to the coordinates
        tck, u = splprep([x, y], s=smoothing_factor, per=True)
        u_new = np.linspace(0, 1, num_points)
        x_new, y_new = splev(u_new, tck)
        smoothed = list(zip(x_new, y_new))
        # Ensure the output is closed
        if not np.allclose(smoothed[0], smoothed[-1]):
            smoothed.append(smoothed[0])
        return smoothed
    except (ValueError, TypeError) as e:
        # If spline fitting fails, return original coordinates
        print(f"Warning: B-spline smoothing failed, using original coordinates: {e}")
        return list(coords)


def roundcorner_smooth(coords, puff_deg=0.05, shrink_ratio=0.6, resolution=16):
    """
    Smooth a closed lon/lat ring with rounded corners while ensuring
    the result still contains the entire original polygon.
    Returns exterior coordinates of the smoothed polygon (closed).

    Parameters:
        coords: List of (lon, lat) tuples (should be closed)
        puff_deg: Buffer distance (degrees) for initial smoothing
        shrink_ratio: How much to shrink the buffer to avoid excessive growth
        resolution: Number of segments per quarter circle for buffer
    """
    ring = np.asarray(coords)
    # Remove duplicate closing point
    if np.allclose(ring[0], ring[-1]):
        ring = ring[:-1]
    poly = Polygon(ring)
    # Repair invalid polygons if needed
    if not poly.is_valid:
        poly = make_valid(poly).buffer(0)
    # "Puff" out the polygon to round corners
    puffed = poly.buffer(puff_deg, resolution=resolution, join_style="round")
    # Shrink back to avoid excessive area gain
    shrunk = puffed.buffer(
        -puff_deg * shrink_ratio, resolution=resolution, join_style="round"
    )
    # Union with original to ensure containment
    safe_smooth = shrunk.union(poly)
    out = list(safe_smooth.exterior.coords)
    # Ensure closed output
    if not np.allclose(out[0], out[-1]):
        out.append(out[0])
    return out


def superspline_smooth(
    coords,
    smoothing_factor=0,
    num_points=400,
    min_outset=0.002,
    max_outset=0.1,
    growth_factor=1.5,
):
    """
    1. B-spline smooth (very curved).
    2. Repair topology.
    3. Incrementally buffer outward until the smoothed polygon fully
       contains the original -> containment guarantee with minimal area gain.
    Returns a CLOSED list of lon/lat tuples.

    Parameters:
        coords: List of (lon, lat) tuples (should be closed)
        smoothing_factor: Spline smoothing parameter
        num_points: Number of points in the output smoothed polygon
        min_outset: Initial buffer distance (degrees) for containment
        max_outset: Maximum buffer distance to try
        growth_factor: Multiplicative step for buffer growth
    """
    ring = np.asarray(coords)
    # Remove duplicate closing point
    if np.allclose(ring[0], ring[-1]):
        ring = ring[:-1]
    orig = Polygon(ring)
    # Repair invalid polygons if needed
    if not orig.is_valid:
        orig = make_valid(orig).buffer(0)
    x, y = ring[:, 0], ring[:, 1]
    # Fit a periodic B-spline to the coordinates
    tck, _ = splprep([x, y], s=smoothing_factor, per=True)
    u_new = np.linspace(0, 1, num_points)
    x_new, y_new = splev(u_new, tck)
    smooth_poly = Polygon(list(zip(x_new, y_new)))
    # Repair invalid polygons if needed
    if not smooth_poly.is_valid:
        smooth_poly = make_valid(smooth_poly).buffer(0)
    # If the smoothed polygon already contains the original, return it
    if smooth_poly.contains(orig):
        out = list(smooth_poly.exterior.coords)
        if not np.allclose(out[0], out[-1]):
            out.append(out[0])
        return out
    # Otherwise, buffer outward until containment is achieved or max_outset is reached
    d = min_outset
    while d <= max_outset:
        grown = smooth_poly.buffer(d, join_style="round", resolution=16)
        if grown.contains(orig):
            out = list(grown.exterior.coords)
            if not np.allclose(out[0], out[-1]):
                out.append(out[0])
            return out
        d *= growth_factor
    # Fallback: union of original and smoothed polygons
    fallback = unary_union([smooth_poly, orig])
    out = list(fallback.exterior.coords)
    if not np.allclose(out[0], out[-1]):
        out.append(out[0])
    return out


def compute_smoothed_wind_region(wind_polys, smoothing="bspline", **kwargs):
    """
    Given a list of wind polygons, compute the union of convex hulls between consecutive polygons,
    then smooth the resulting shape using the specified method.

    Parameters:
        wind_polys: List of Shapely Polygon objects (wind regions)
        smoothing: 'bspline', 'roundcorner', 'superspline', or None
        kwargs: parameters for the smoothing function

    Returns:
        A Shapely Polygon (smoothed wind region) or None if input is empty.
    """
    convex_hulls = []
    # Compute convex hulls between each pair of consecutive wind polygons
    for i in range(len(wind_polys) - 1):
        poly1 = wind_polys[i]
        poly2 = wind_polys[i + 1]
        if poly1 is not None and poly2 is not None:
            union = unary_union([poly1, poly2])
            hull = union.convex_hull
            if hull is not None and hull.is_valid and not hull.is_empty:
                convex_hulls.append(hull)
    if not convex_hulls:
        return None
    # Union all convex hulls to get a continuous wind region shape
    continuous_shape = unary_union(convex_hulls)
    # If the result is a single Polygon, apply smoothing if requested
    if continuous_shape.geom_type == "Polygon":
        coords = list(continuous_shape.exterior.coords)
        if smoothing == "bspline":
            smoothed_coords = bspline_smooth(coords, **kwargs)
            continuous_shape = Polygon(smoothed_coords)
        elif smoothing == "roundcorner":
            smoothed_coords = roundcorner_smooth(coords, **kwargs)
            continuous_shape = Polygon(smoothed_coords)
        elif smoothing == "superspline":
            smoothed_coords = superspline_smooth(coords, **kwargs)
            continuous_shape = Polygon(smoothed_coords)
    # Repair any invalid geometry that may have resulted from smoothing
    continuous_shape = make_valid(continuous_shape).buffer(0)
    return continuous_shape


def get_nicaragua_polygon():
    """
    Download and return the Nicaragua polygon from a public GeoJSON source.
    Returns:
        shapely.geometry.Polygon: Nicaragua boundary polygon
    """
    countries_url = "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"
    countries = requests.get(countries_url, timeout=10).json()
    nic_poly = None
    for feat in countries["features"]:
        props = feat.get("properties", {})
        if any(
            isinstance(v, str) and v.strip().lower() == "nicaragua"
            for v in props.values()
        ):
            nic_poly = shape(feat["geometry"])
            break
    if nic_poly is None:
        raise RuntimeError("Could not find Nicaragua in GeoJSON.")
    return nic_poly


def get_nicaragua_boundary():
    """
    Return a GeoDataFrame with the Nicaragua boundary polygon.
    Returns:
        geopandas.GeoDataFrame: Nicaragua boundary
    """
    import geopandas as gpd

    try:
        nic_poly = get_nicaragua_polygon()
        nicaragua_gdf = gpd.GeoDataFrame(geometry=[nic_poly], crs="EPSG:4326")
        return nicaragua_gdf
    except Exception as e:
        print(f"Error getting Nicaragua boundary: {e}")
        return None


__all__ += ["get_nicaragua_polygon", "get_nicaragua_boundary"]
