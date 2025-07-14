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
from shapely import make_valid
from src.utils.config_utils import get_config_value
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


def create_hurricane_heatmap(
    df, chosen_forecast, output_dir, grid_res=0.1, config=None
):
    print(
        "\n🎯 Generating grid heatmap of ensemble track intersections over Nicaragua..."
    )

    # Get configuration values with defaults
    if config is None:
        config = {}

    # Visualization settings
    figure_size = get_config_value(
        config, "impact_analysis.heatmaps.figure_size", [12, 10]
    )
    if isinstance(figure_size, list) and len(figure_size) == 2:
        fig_size = tuple(figure_size)
    else:
        fig_size = (12, 10)
    dpi = get_config_value(config, "impact_analysis.heatmaps.dpi", 300)
    color_map = get_config_value(
        config, "impact_analysis.heatmaps.color_maps.hurricane", "YlOrRd"
    )
    alpha = get_config_value(config, "impact_analysis.heatmaps.alpha", 0.7)
    line_width = get_config_value(config, "impact_analysis.heatmaps.line_width", 0.1)
    edge_color = get_config_value(config, "impact_analysis.heatmaps.edge_color", "grey")

    # Grid bounds
    bounds_config = get_config_value(config, "impact_analysis.grid.bounds", {})
    default_bounds = {
        "lon_min": -87.7,
        "lon_max": -82.7,
        "lat_min": 10.7,
        "lat_max": 15.1,
    }
    bounds = {**default_bounds, **bounds_config}

    nicaragua_gdf = get_nicaragua_boundary()
    if nicaragua_gdf is not None:
        minx, miny, maxx, maxy = nicaragua_gdf.total_bounds
        print(
            f"Nicaragua bounds: lon {minx:.6f} to {maxx:.6f}, lat {miny:.6f} to {maxy:.6f}"
        )
    else:
        minx, maxx = bounds["lon_min"], bounds["lon_max"]
        miny, maxy = bounds["lat_min"], bounds["lat_max"]
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

    print("\n🎯 Generating grid heatmap of wind region intersections over Nicaragua...")
    # Read smoothing method and params from config
    smoothing_method = get_config_value(
        config, "impact_analysis.wind_region.smoothing_method", "bspline"
    )
    smoothing_params = get_config_value(
        config,
        "impact_analysis.wind_region.smoothing_params",
        {"smoothing_factor": 0, "num_points": 200},
    )
    # For each ensemble member, compute the smoothed wind region polygon
    df_ens = df[df["forecast_time"] == chosen_forecast].copy()
    member_regions = []
    for member in df_ens["ensemble_member"].unique():
        member_data = df_ens[df_ens["ensemble_member"] == member].sort_values(
            "valid_time"
        )
        wind_polys = []
        for _, row in member_data.iterrows():
            lat = row["latitude"]
            lon = row["longitude"]
            r_ne = row.get("radius_50_knot_winds_ne_km", 0) or 0
            r_se = row.get("radius_50_knot_winds_se_km", 0) or 0
            r_sw = row.get("radius_50_knot_winds_sw_km", 0) or 0
            r_nw = row.get("radius_50_knot_winds_nw_km", 0) or 0
            if any([r_ne, r_se, r_sw, r_nw]):
                poly = wind_quadrant_polygon(lat, lon, r_ne, r_se, r_sw, r_nw)
                if poly is not None and poly.is_valid and not poly.is_empty:
                    wind_polys.append(poly)
        # Use the centralized helper for smoothing
        continuous_shape = compute_smoothed_wind_region(
            wind_polys, smoothing=smoothing_method, **smoothing_params
        )
        if continuous_shape is not None:
            member_regions.append(continuous_shape)
    # For each grid cell, count how many member regions intersect it
    region_counts = []
    for cell in grid_gdf.geometry:
        count = sum(
            region is not None
            and region.is_valid
            and not region.is_empty
            and region.intersects(cell)
            for region in member_regions
        )
        region_counts.append(count)
    grid_gdf["wind_region_count"] = region_counts

    fig, ax = plt.subplots(figsize=fig_size)
    grid_gdf.plot(
        ax=ax,
        column="track_count",
        cmap=color_map,
        linewidth=line_width,
        edgecolor=edge_color,
        alpha=alpha,
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
    plt.savefig(heatmap_path, dpi=dpi, bbox_inches="tight")
    print(f"\n✅ Grid heatmap saved:\n   {heatmap_path}")
    plt.close()

    # Plot the new heatmap
    fig2, ax2 = plt.subplots(figsize=fig_size)
    grid_gdf.plot(
        ax=ax2,
        column="wind_region_count",
        cmap=color_map,
        linewidth=line_width,
        edgecolor=edge_color,
        alpha=alpha,
        legend=True,
        legend_kwds={"label": "# of Wind Regions per Cell"},
    )
    nicaragua_gdf.plot(ax=ax2, color="none", edgecolor="black", linewidth=3, alpha=1.0)
    ax2.set_xlim(minx, maxx)
    ax2.set_ylim(miny, maxy)
    ax2.set_xlabel("Longitude (°E)", fontsize=12)
    ax2.set_ylabel("Latitude (°N)", fontsize=12)
    ax2.set_title(
        f"Wind Region Intersection Heatmap\nGrid Resolution: {grid_res}°",
        fontsize=14,
        fontweight="bold",
    )
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    heatmap_path2 = os.path.join(
        output_dir,
        f"ensemble_grid_heatmap_windregion_{chosen_forecast.strftime('%Y%m%d_%H%M')}.png",
    )
    print(f"[DEBUG] Saving wind region grid heatmap to: {heatmap_path2}")
    plt.savefig(heatmap_path2, dpi=dpi, bbox_inches="tight")
    print(f"\n✅ Wind region grid heatmap saved:\n   {heatmap_path2}")
    plt.close()
    return grid_gdf
