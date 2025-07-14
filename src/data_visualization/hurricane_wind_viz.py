import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, shape
import geopandas as gpd
import requests
from datetime import datetime
from src.utils.hurricane_geom import (
    wind_quadrant_polygon,
    bspline_smooth,
    roundcorner_smooth,
    superspline_smooth,
    compute_smoothed_wind_region,
)


def get_nicaragua_polygon():
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
    try:
        nic_poly = get_nicaragua_polygon()
        nicaragua_gdf = gpd.GeoDataFrame(geometry=[nic_poly], crs="EPSG:4326")
        return nicaragua_gdf
    except Exception as e:
        print(f"Error getting Nicaragua boundary: {e}")
        return None


def find_latest_synthetic_csv():
    folder = "data/preprocessed/weatherlab/synthetic/"
    files = glob.glob(os.path.join(folder, "*.csv"))
    if not files:
        raise FileNotFoundError("No synthetic hurricane CSV found in " + folder)
    latest = max(files, key=os.path.getmtime)
    return latest


def main():
    # 1. Load latest synthetic hurricane CSV
    csv_path = find_latest_synthetic_csv()
    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    # 2. Filter for ensemble_member == 0.0
    df = df[df["ensemble_member"] == 0.0].copy()
    # 3. Sort by valid_time if present
    if "valid_time" in df.columns:
        try:
            df["valid_time"] = df["valid_time"].astype(str)
            df = df.sort_values(by=["valid_time"])
        except Exception as e:
            print(f"Warning: Could not sort by valid_time: {e}")
    # 4. Prepare wind polygons
    wind_polys = []
    for _, row in df.iterrows():
        lat = row["latitude"]
        lon = row["longitude"]
        r_ne = row.get("radius_50_knot_winds_ne_km", 0) or 0
        r_se = row.get("radius_50_knot_winds_se_km", 0) or 0
        r_sw = row.get("radius_50_knot_winds_sw_km", 0) or 0
        r_nw = row.get("radius_50_knot_winds_nw_km", 0) or 0
        # Only plot if at least one radius is positive
        if any([r_ne, r_se, r_sw, r_nw]):
            poly = wind_quadrant_polygon(lat, lon, r_ne, r_se, r_sw, r_nw)
            if poly is not None and poly.is_valid and not poly.is_empty:
                wind_polys.append(poly)
    wind_gdf = gpd.GeoDataFrame(geometry=wind_polys, crs="EPSG:4326")

    # --- New: Compute convex hulls between consecutive polygons ---
    from shapely.ops import unary_union

    convex_hulls = []
    for i in range(len(wind_polys) - 1):
        poly1 = wind_polys[i]
        poly2 = wind_polys[i + 1]
        if poly1 is not None and poly2 is not None:
            union = unary_union([poly1, poly2])
            hull = union.convex_hull
            if hull is not None and hull.is_valid and not hull.is_empty:
                convex_hulls.append(hull)
    # Union all convex hulls into a single shape
    if convex_hulls:
        # Use the centralized helper for smoothing
        continuous_shape = compute_smoothed_wind_region(
            wind_polys, smoothing="bspline", smoothing_factor=0, num_points=200
        )
    else:
        continuous_shape = None

    # 5. Plot (discrete version)
    fig, ax = plt.subplots(figsize=(10, 10))
    if isinstance(ax, np.ndarray):
        ax = ax.flat[0]
    nicaragua = get_nicaragua_boundary()
    if nicaragua is not None:
        nicaragua.plot(ax=ax, color="none", edgecolor="black", linewidth=2, alpha=0.7)
    if not wind_gdf.empty:
        wind_gdf.plot(ax=ax, color="orange", alpha=0.3, edgecolor="red", linewidth=0.5)
    if not df.empty:
        ax.plot(
            df["longitude"],
            df["latitude"],
            color="blue",
            marker="o",
            label="Track",
            linewidth=2,
        )
    ax.set_title("Synthetic Hurricane Track and 50kt Wind Regions (Ensemble Member 0)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()
    plt.tight_layout()
    out_path = "data/results/wind_spread/synthetic_hurricane_50kt_wind.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved wind spread plot to: {out_path}")
    plt.close()

    # 6. Plot with continuous convex hull overlay
    fig2, ax2 = plt.subplots(figsize=(10, 10))
    if isinstance(ax2, np.ndarray):
        ax2 = ax2.flat[0]
    if nicaragua is not None:
        nicaragua.plot(ax=ax2, color="none", edgecolor="black", linewidth=2, alpha=0.7)
    if not wind_gdf.empty:
        wind_gdf.plot(ax=ax2, color="orange", alpha=0.2, edgecolor="red", linewidth=0.5)
    if not df.empty:
        ax2.plot(
            df["longitude"],
            df["latitude"],
            color="blue",
            marker="o",
            label="Track",
            linewidth=2,
        )
    # Plot the continuous convex hull shape
    if continuous_shape is not None and not continuous_shape.is_empty:
        hull_geom = continuous_shape
        # Optionally smooth the hull
        if hull_geom.geom_type == "Polygon":
            coords = list(hull_geom.exterior.coords)
            smoothed_coords = bspline_smooth(coords, smoothing_factor=0, num_points=200)
            from shapely.geometry import Polygon

            hull_geom = Polygon(smoothed_coords)
        hull_gdf = gpd.GeoDataFrame(geometry=[hull_geom], crs="EPSG:4326")
        hull_gdf.plot(
            ax=ax2,
            color="deepskyblue",
            alpha=0.4,
            edgecolor="black",
            linewidth=2,
            label="Continuous Hull",
        )
    ax2.set_title(
        "Synthetic Hurricane Track and Continuous 50kt Wind Region (Convex Hull)"
    )
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    ax2.legend()
    plt.tight_layout()
    out_path2 = "data/results/wind_spread/synthetic_hurricane_50kt_wind_continuous.png"
    plt.savefig(out_path2, dpi=300)
    print(f"Saved continuous wind spread plot to: {out_path2}")
    plt.close()


if __name__ == "__main__":
    main()
