"""
School Grid Heatmap
This module contains logic to create a grid heatmap of school concentrations.
"""

import os
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from src.impact_analysis.hurricane_grid_heatmap import get_nicaragua_boundary
from src.utils.config_utils import get_config_value
from src.utils.path_utils import get_data_path
from shapely.geometry import box


def create_school_heatmap(output_dir, grid_res=0.1, config=None):
    print("\n🎯 Generating school concentration heatmap over Nicaragua...")

    # Get configuration values with defaults
    if config is None:
        config = {}

    # Get school data file path from config
    school_data_path = get_config_value(
        config, "impact_analysis.input.school_data", "data/raw/osm/schools.geojson"
    )
    schools_file = get_data_path(school_data_path)

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
        config, "impact_analysis.heatmaps.color_maps.school", "Blues"
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
    else:
        minx, maxx = bounds["lon_min"], bounds["lon_max"]
        miny, maxy = bounds["lat_min"], bounds["lat_max"]

    grid_cells = []
    x_coords = np.arange(minx, maxx, grid_res)
    y_coords = np.arange(miny, maxy, grid_res)
    for x in x_coords:
        for y in y_coords:
            grid_cells.append(box(x, y, x + grid_res, y + grid_res))
    grid_gdf = gpd.GeoDataFrame(grid_cells, columns=["geometry"], crs="EPSG:4326")

    if schools_file.exists():
        print(f"Loading schools data: {schools_file}")
        schools_gdf = gpd.read_file(schools_file)
        print(f"Total schools found: {len(schools_gdf)}")
        school_counts = []
        for cell in grid_gdf.geometry:
            count = schools_gdf.within(cell).sum()
            school_counts.append(count)
        grid_gdf["school_count"] = school_counts
        print(
            f"Grid cells with schools: {sum(1 for count in school_counts if count > 0)}"
        )
        print(f"Max schools per cell: {max(school_counts)}")
        non_zero_counts = [count for count in school_counts if count > 0]
        if non_zero_counts:
            print(
                f"Mean schools per cell (non-zero): {sum(non_zero_counts)/len(non_zero_counts):.1f}"
            )
            print(
                f"Median schools per cell (non-zero): {sorted(non_zero_counts)[len(non_zero_counts)//2]}"
            )
            print(
                f"95th percentile: {sorted(non_zero_counts)[int(len(non_zero_counts)*0.95)]}"
            )

        fig, ax = plt.subplots(figsize=fig_size)
        log_counts = [np.log10(count + 1) for count in school_counts]
        grid_gdf["log_school_count"] = log_counts
        grid_gdf.plot(
            ax=ax,
            column="log_school_count",
            cmap=color_map,
            linewidth=line_width,
            edgecolor=edge_color,
            alpha=alpha,
            legend=True,
            legend_kwds={"label": "Log10(Schools + 1) per Cell"},
        )
        nicaragua_gdf.plot(
            ax=ax, color="none", edgecolor="black", linewidth=3, alpha=1.0
        )
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_xlabel("Longitude (°E)", fontsize=12)
        ax.set_ylabel("Latitude (°N)", fontsize=12)
        ax.set_title(
            f"School Concentration Heatmap (Log Scale)\nGrid Resolution: {grid_res}°",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        school_heatmap_path = os.path.join(
            output_dir,
            "school_concentration_heatmap.png",
        )
        print(f"[DEBUG] Saving school heatmap to: {school_heatmap_path}")
        plt.savefig(school_heatmap_path, dpi=dpi, bbox_inches="tight")
        print(f"\n✅ School concentration heatmap saved:\n   {school_heatmap_path}")
        plt.close()
        return grid_gdf
    else:
        print(f"❌ Schools data file not found: {schools_file}")
        return None
