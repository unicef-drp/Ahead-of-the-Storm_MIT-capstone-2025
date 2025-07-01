"""
Hurricane Grid Heatmap
This module contains logic to create a grid heatmap of hurricane track intersections.
"""

import os
import geopandas as gpd
import requests
from shapely.geometry import shape, box, LineString
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


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
        print("✅ Successfully loaded Nicaragua boundary from GeoJSON")
        return nicaragua_gdf
    except Exception as e:
        print(f"⚠️  Error getting Nicaragua boundary from GeoJSON: {e}")
        print("⚠️  Using bounding box as fallback")
        nicaragua_bbox = box(-87.7, 10.7, -82.7, 15.0)
        nicaragua_gdf = gpd.GeoDataFrame(
            [{"geometry": nicaragua_bbox}], crs="EPSG:4326"
        )
        return nicaragua_gdf


def create_hurricane_heatmap(df, chosen_forecast, output_dir, grid_res=0.1):
    print(
        "\n🎯 Generating grid heatmap of ensemble track intersections over Nicaragua..."
    )
    nicaragua_gdf = get_nicaragua_boundary()
    if nicaragua_gdf is not None:
        minx, miny, maxx, maxy = nicaragua_gdf.total_bounds
        print(
            f"Nicaragua bounds: lon {minx:.6f} to {maxx:.6f}, lat {miny:.6f} to {maxy:.6f}"
        )
    else:
        minx, maxx = -88, -82
        miny, maxy = 10, 16
        print(f"Using default bounds: lon {minx} to {maxx}, lat {miny} to {maxy}")
    grid_cells = []
    x_coords = np.arange(minx, maxx, grid_res)
    y_coords = np.arange(miny, maxy, grid_res)
    for x in x_coords:
        for y in y_coords:
            grid_cells.append(box(x, y, x + grid_res, y + grid_res))
    grid_gdf = gpd.GeoDataFrame(grid_cells, columns=["geometry"], crs="EPSG:4326")
    df_ens = df[df["forecast_time"] == chosen_forecast].copy()
    member_lines = []
    for member in df_ens["ensemble_member"].unique():
        member_data = df_ens[df_ens["ensemble_member"] == member].sort_values(
            "valid_time"
        )
        if len(member_data) > 1:
            coords = list(zip(member_data["longitude"], member_data["latitude"]))
            line = LineString(coords)
            member_lines.append(line)
    print(f"Total ensemble member lines: {len(member_lines)}")
    # Count, for each grid cell, how many unique ensemble tracks (LineStrings) intersect it
    counts = []
    for cell in grid_gdf.geometry:
        count = sum(line.intersects(cell) for line in member_lines)
        counts.append(count)
    grid_gdf["track_count"] = counts
    fig, ax = plt.subplots(figsize=(12, 10))
    grid_gdf.plot(
        ax=ax,
        column="track_count",
        cmap="YlOrRd",
        linewidth=0.1,
        edgecolor="grey",
        alpha=0.7,
        legend=True,
        legend_kwds={"label": "# of Ensemble Tracks per Cell"},
    )
    nicaragua_gdf.plot(ax=ax, color="none", edgecolor="black", linewidth=3, alpha=1.0)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_xlabel("Longitude (°E)", fontsize=12)
    ax.set_ylabel("Latitude (°N)", fontsize=12)
    ax.set_title(
        f"Ensemble Track Intersection Heatmap\nGrid Resolution: {grid_res}°",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    heatmap_path = os.path.join(
        output_dir,
        f"ensemble_grid_heatmap_{chosen_forecast.strftime('%Y%m%d_%H%M')}.png",
    )
    print(f"[DEBUG] Saving grid heatmap to: {heatmap_path}")
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    print(f"\n✅ Grid heatmap saved:\n   {heatmap_path}")
    plt.close()
    return grid_gdf
