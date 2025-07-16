#!/usr/bin/env python3
"""
Landslide Heatmap Visualization

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
            
            # Get bounds
            bounds = src.bounds
            logger.info(f"Bounds: {bounds}")
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create a masked array to handle no-data values
        # NASA LHASA-F typically uses 255 or 0 for no-data areas
        # We'll mask out areas with no landslide hazard data
        
        # First, let's analyze the data distribution
        unique_values, counts = np.unique(data, return_counts=True)
        logger.info(f"Data distribution:")
        for val, count in zip(unique_values, counts):
            percentage = (count / data.size) * 100
            logger.info(f"  Value {val}: {count} pixels ({percentage:.1f}%)")
        
        # Check for common no-data patterns
        # If 255 is the most common value, it's likely no-data
        most_common_value = unique_values[np.argmax(counts)]
        most_common_count = np.max(counts)
        most_common_percentage = (most_common_count / data.size) * 100
        
        logger.info(f"Most common value: {most_common_value} ({most_common_percentage:.1f}% of pixels)")
        
        # Create mask for no-data values
        # Mask out the most common value if it's likely no-data (more than 30% of pixels)
        if most_common_percentage > 30:
            logger.info(f"Masking out no-data value: {most_common_value}")
            masked_data = np.ma.masked_where(data == most_common_value, data)
            
            # Get the actual hazard range (excluding no-data)
            valid_data = data[data != most_common_value]
            if len(valid_data) > 0:
                logger.info(f"Valid hazard data range: {valid_data.min():.3f} to {valid_data.max():.3f}")
                # Set the colorbar range to actual hazard data
                vmin, vmax = valid_data.min(), valid_data.max()
            else:
                logger.warning("No valid hazard data found")
                vmin, vmax = None, None
        else:
            # If no clear no-data pattern, just mask zeros
            logger.info("No clear no-data pattern detected, masking zeros only")
            masked_data = np.ma.masked_where(data == 0, data)
            vmin, vmax = None, None
        
        plot_data = masked_data
        
        # Prepare data for plotting
        if log_scale and plot_data.max() is not None and plot_data.max() > 0:
            # Use log scale for better visualization
            plot_data = np.ma.log10(plot_data + 1)  # log10(x + 1) to handle zeros
            colorbar_label = "Log10(Landslide Hazard + 1)"
            logger.info("Using log scale for visualization")
        else:
            colorbar_label = "Landslide Hazard"
            logger.info("Using linear scale for visualization")
        
        # Create extent for imshow
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        
        # Plot the raster data with proper masking
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
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(colorbar_label, fontsize=12)
        
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
    """Main function to create landslide heatmap."""
    logger = setup_logging(__name__)
    
    logger.info("=" * 60)
    logger.info("LANDSLIDE HAZARD HEATMAP VISUALIZATION")
    logger.info("=" * 60)
    
    try:
        # Find the most recent landslide GeoTIFF
        landslide_dir = get_data_path("data/preprocessed/landslide")
        tiff_files = list(landslide_dir.glob("*.tif"))
        
        if not tiff_files:
            logger.error("No landslide GeoTIFF files found!")
            return
        
        # Use the most recent file
        latest_tiff = max(tiff_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Using landslide data: {latest_tiff}")
        
        # Create heatmap
        plot_path = plot_landslide_heatmap(
            tiff_path=str(latest_tiff),
            output_dir="data/results/landslide",
            title="NASA LHASA-F Landslide Hazard Prediction\nNicaragua - Tomorrow's Forecast",
            cmap="YlOrRd",
            log_scale=False
        )
        
        if plot_path:
            logger.info("=" * 60)
            logger.info("✅ LANDSLIDE HEATMAP CREATED SUCCESSFULLY!")
            logger.info("=" * 60)
            logger.info(f"Plot saved to: {plot_path}")
            logger.info("Data source: NASA LHASA-F")
            logger.info("Coverage: Nicaragua")
            logger.info("Use: Exposure grid for impact analysis")
        else:
            logger.error("❌ Failed to create landslide heatmap")
            
    except Exception as e:
        logger.error(f"Error in main function: {e}")


if __name__ == "__main__":
    main() 