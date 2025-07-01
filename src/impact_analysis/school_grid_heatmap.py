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
from shapely.geometry import box


def create_school_heatmap(output_dir, grid_res=0.1):
    print("\n🎯 Generating school concentration heatmap over Nicaragua...")
    nicaragua_gdf = get_nicaragua_boundary()
    if nicaragua_gdf is not None:
        minx, miny, maxx, maxy = nicaragua_gdf.total_bounds
    else:
        minx, maxx = -88, -82
        miny, maxy = 10, 16
    grid_cells = []
    x_coords = np.arange(minx, maxx, grid_res)
    y_coords = np.arange(miny, maxy, grid_res)
    for x in x_coords:
        for y in y_coords:
            grid_cells.append(box(x, y, x + grid_res, y + grid_res))
    grid_gdf = gpd.GeoDataFrame(grid_cells, columns=["geometry"], crs="EPSG:4326")
    schools_file = "data/raw/osm/schools.geojson"
    if os.path.exists(schools_file):
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
        fig, ax = plt.subplots(figsize=(12, 10))
        log_counts = [np.log10(count + 1) for count in school_counts]
        grid_gdf["log_school_count"] = log_counts
        grid_gdf.plot(
            ax=ax,
            column="log_school_count",
            cmap="Blues",
            linewidth=0.1,
            edgecolor="grey",
            alpha=0.7,
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
        plt.savefig(school_heatmap_path, dpi=300, bbox_inches="tight")
        print(f"\n✅ School concentration heatmap saved:\n   {school_heatmap_path}")
        plt.close()
        return grid_gdf
    else:
        print(f"❌ Schools data file not found: {schools_file}")
        return None
