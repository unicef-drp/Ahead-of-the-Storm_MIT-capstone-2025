"""
Script to plot the rasterized schools GeoTIFF as a heatmap with Nicaragua boundary overlay, matching the style of school_grid_heatmap.py.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import yaml
from src.utils.path_utils import get_project_root
from src.utils.hurricane_geom import get_nicaragua_boundary
from src.utils.config_utils import get_config_value


def read_config(config_path=None):
    if config_path is None:
        root = get_project_root()
        config_path = os.path.join(root, 'config', 'impact_analysis_config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    try:
        # Read config
        config = read_config()
        school_raster_path = config['school_rasterization']['output_tiff']
        output_plot_path = os.path.join(
            get_project_root(),
            'data', 'results', 'impact_analysis', 'schools_raster_heatmap.png'
        )

        # Visualization settings (match school_grid_heatmap.py)
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
        # line_width and edge_color are not used for raster, but keep for consistency
        line_width = get_config_value(config, "impact_analysis.heatmaps.line_width", 0.1)
        edge_color = get_config_value(config, "impact_analysis.heatmaps.edge_color", "grey")

        # Get grid bounds
        bounds_config = get_config_value(config, "impact_analysis.grid.bounds", {})
        default_bounds = {
            "lon_min": -87.7,
            "lon_max": -82.7,
            "lat_min": 10.7,
            "lat_max": 15.1,
        }
        bounds = {**default_bounds, **bounds_config}

        # Get Nicaragua boundary
        nicaragua_gdf = get_nicaragua_boundary()
        if nicaragua_gdf is not None:
            minx, miny, maxx, maxy = nicaragua_gdf.total_bounds
        else:
            minx, maxx = bounds["lon_min"], bounds["lon_max"]
            miny, maxy = bounds["lat_min"], bounds["lat_max"]

        # Read raster
        with rasterio.open(school_raster_path) as src:
            raster = src.read(1)
            raster[raster < 0] = 0  # Remove any negative values
            extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

        # Plot
        fig, ax = plt.subplots(figsize=fig_size)
        img = ax.imshow(
            np.log10(raster + 1),
            extent=extent,
            origin='upper',
            cmap=color_map,
            interpolation='nearest',
            vmin=0,
            alpha=alpha,
            aspect='auto'
        )
        if nicaragua_gdf is not None:
            nicaragua_gdf.plot(ax=ax, color="none", edgecolor="black", linewidth=3, alpha=1.0)
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_xlabel("Longitude (°E)", fontsize=12)
        ax.set_ylabel("Latitude (°N)", fontsize=12)
        ax.set_title(
            "School Concentration Heatmap (Log Scale)\nGrid Resolution: 0.1°",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(img, ax=ax, label='Log10(Schools + 1) per Cell')
        plt.tight_layout()
        plt.savefig(output_plot_path, dpi=dpi, bbox_inches="tight")
        print(f"Heatmap saved to {output_plot_path}")
        plt.close()
    except Exception as e:
        print(f"Error generating heatmap: {e}")


if __name__ == "__main__":
    main() 