#!/usr/bin/env python3
"""
Landslide Heatmap Visualization (Standalone)

This script is for manual/standalone visualization of NASA LHASA-F landslide hazard data.
It is NOT used in the main impact analysis pipeline.

This script creates a simple heatmap visualization of NASA LHASA-F landslide hazard data
for Nicaragua, following the same pattern as hurricane heatmaps.
"""

import rasterio
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import requests
from shapely.geometry import shape
from pathlib import Path
import os
import re

from src.utils.config_utils import load_config, get_config_value
from src.utils.logging_utils import setup_logging, get_logger
from src.utils.path_utils import ensure_directory, get_data_path


def get_nicaragua_boundary():
    """Get Nicaragua boundary for plotting."""
    try:
        countries_url = "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"
        countries = requests.get(countries_url).json()
        
        nic_poly = None
        for feat in countries["features"]:
            props = feat.get("properties", {})
            if any(
                isinstance(v, str) and v.strip().lower() == "nicaragua"
                for v in props.values()
            ):
                nic_poly = shape(feat["geometry"])
                break
        
        if nic_poly is not None:
            return gpd.GeoDataFrame(geometry=[nic_poly], crs="EPSG:4326")
        else:
            return None
    except Exception as e:
        print(f"Warning: Could not load Nicaragua boundary: {e}")
        return None


def plot_landslide_heatmap(
    tiff_path: str,
    output_dir: str = "data/results/landslide",
    title: str = "NASA LHASA-F Landslide Hazard Prediction",
    cmap: str = "YlOrRd",
    figsize: tuple = (12, 10),
    dpi: int = 300,
    show_boundary: bool = True,
    log_scale: bool = False
):
    """
    Create a heatmap visualization of landslide hazard data.
    
    Args:
        tiff_path: Path to the GeoTIFF file
        output_dir: Directory to save the plot
        title: Plot title
        cmap: Colormap for the heatmap
        figsize: Figure size
        dpi: DPI for saved image
        show_boundary: Whether to show Nicaragua boundary
        log_scale: Whether to use log scale for better visualization
    """
    logger = get_logger(__name__)
    
    # Ensure output directory exists
    output_path = get_data_path(output_dir)
    ensure_directory(output_path)
    
    try:
        # Load the GeoTIFF
        with rasterio.open(tiff_path) as src:
            data = src.read(1)  # Read first band
            transform = src.transform
            crs = src.crs
            logger.info(f"Loaded GeoTIFF: {data.shape} pixels, CRS: {crs}")
            logger.info(f"Data range: {data.min():.3f} to {data.max():.3f}")
            bounds = src.bounds
            logger.info(f"Bounds: {bounds}")
            nodata = src.nodata

        # Create the plot axis early to avoid 'ax' errors
        fig, ax = plt.subplots(figsize=figsize)

        # Robust masking: mask nodata (using np.isclose), negative values, and nan values
        if nodata is not None:
            mask = np.isclose(data, nodata) | (data < 0) | np.isnan(data)
            masked_data = np.ma.masked_where(mask, data)
        else:
            masked_data = np.ma.masked_where((data < 0) | np.isnan(data), data)

        plot_data = masked_data
        valid_data = masked_data.compressed()
        if valid_data.size > 0:
            vmin = 0
            vmax = valid_data.max()
        else:
            vmin = 0
            vmax = 1

        # Prepare data for plotting
        if log_scale and valid_data.size > 0 and valid_data.max() > 0:
            plot_data = np.ma.log10(plot_data + 1)  # log10(x + 1) to handle zeros
            colorbar_label = "Log10(Landslide Hazard + 1)"
        else:
            colorbar_label = "Landslide Hazard"

        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

        # Plot the raster data with proper masking and dynamic colorbar
        im = ax.imshow(
            plot_data,
            extent=extent,
            cmap=cmap,
            origin='upper',
            aspect='auto',
            vmin=vmin,
            vmax=vmax
        )
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.036, pad=0.04)
        cbar.set_label('Landslide Probability')
        
        # Add Nicaragua boundary if requested
        if show_boundary:
            nicaragua_gdf = get_nicaragua_boundary()
            if nicaragua_gdf is not None:
                nicaragua_gdf.plot(
                    ax=ax, 
                    color="none", 
                    edgecolor="black", 
                    linewidth=2, 
                    alpha=0.8
                )
                logger.info("Added Nicaragua boundary to plot")
        
        # Set plot properties
        ax.set_xlabel("Longitude (°E)", fontsize=12)
        ax.set_ylabel("Latitude (°N)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        
        # Set axis limits to Nicaragua bounds
        ax.set_xlim(-87.7, -82.7)
        ax.set_ylim(10.7, 15.1)
        
        plt.tight_layout()
        
        # Save the plot
        filename = f"landslide_heatmap_{Path(tiff_path).stem}.png"
        plot_path = output_path / filename
        plt.savefig(plot_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved landslide heatmap: {plot_path}")
        
        plt.close()
        return str(plot_path)
        
    except Exception as e:
        logger.error(f"Error creating landslide heatmap: {e}")
        return None


def main():
    """Main function to create landslide heatmap for the latest 48h forecast."""
    logger = setup_logging(__name__)
    logger.info("=" * 60)
    logger.info("LANDSLIDE HAZARD HEATMAP VISUALIZATION (48h Forecast)")
    logger.info("=" * 60)
    try:
        # Find the most recent 48h landslide GeoTIFF with new naming convention
        landslide_dir = get_data_path("data/preprocessed/landslide")
        tiff_files = list(landslide_dir.glob("landslide_forecast_48h_*_nicaragua.tif"))
        if not tiff_files:
            logger.error("No 48h landslide GeoTIFF files found!")
            return
        # Use the most recent file
        latest_tiff = max(tiff_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Using landslide data: {latest_tiff}")
        # Extract forecast start time for output filename
        match = re.match(r"landslide_forecast_48h_(\d{8}T\d{4})_nicaragua.tif", latest_tiff.name)
        if match:
            start = match.group(1)
            plot_filename = f"landslide_heatmap_48h_{start}_nicaragua.png"
        else:
            plot_filename = "landslide_heatmap_48h_unknown_nicaragua.png"
        # Create heatmap
        plot_path = plot_landslide_heatmap(
            tiff_path=str(latest_tiff),
            output_dir="data/results/landslide",
            title="NASA LHASA-F Landslide Probability (48h Forecast)\nNicaragua",
            cmap="YlOrRd",
            log_scale=False
        )
        # Rename/move plot to match new convention if needed
        if plot_path and Path(plot_path).name != plot_filename:
            new_plot_path = Path(plot_path).parent / plot_filename
            os.rename(plot_path, new_plot_path)
            logger.info(f"Renamed plot to: {new_plot_path}")
        if plot_path:
            logger.info("=" * 60)
            logger.info("✅ 48h LANDSLIDE HEATMAP CREATED SUCCESSFULLY!")
            logger.info("=" * 60)
            logger.info(f"Plot saved to: {plot_path}")
            logger.info("Data source: NASA LHASA-F (probability)")
            logger.info("Coverage: Nicaragua")
            logger.info("Use: Probability grid for impact analysis")
        else:
            logger.error("❌ Failed to create 48h landslide heatmap")
    except Exception as e:
        logger.error(f"Error in main function: {e}")


if __name__ == "__main__":
    main() 