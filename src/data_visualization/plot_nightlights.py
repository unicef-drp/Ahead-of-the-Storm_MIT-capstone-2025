#!/usr/bin/env python3
"""
Visualize nightlights data for Nicaragua.

This script creates comprehensive visualizations of the processed nightlights data.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import rasterio
import geopandas as gpd
from src.utils.hurricane_geom import get_nicaragua_boundary
from src.utils.config_utils import load_config, get_project_root
from src.utils.path_utils import get_data_path


def plot_nightlights_data():
    """Create comprehensive nightlights visualizations."""
    print("=" * 60)
    print("VISUALIZING NIGHTLIGHTS DATA")
    print("=" * 60)

    # Load configuration
    config = load_config("config/nightlights_config.yaml")
    project_root = get_project_root()

    # Setup paths
    processed_dir = get_data_path("data/preprocessed/nightlights/processed")
    output_dir = get_data_path("data/results/nightlights")
    output_dir.mkdir(exist_ok=True)

    print(f"Processed data directory: {processed_dir}")
    print(f"Output directory: {output_dir}")

    # Find nightlights files
    nightlights_files = list(processed_dir.glob("*nicaragua*.tif"))

    if not nightlights_files:
        print("No Nicaragua nightlights files found!")
        return

    print(f"Found {len(nightlights_files)} Nicaragua nightlights files")

    # Get Nicaragua boundary for overlay
    nicaragua_gdf = get_nicaragua_boundary()

    # Process each file
    for file_path in nightlights_files:
        print(f"\nVisualizing: {file_path.name}")

        try:
            with rasterio.open(file_path) as src:
                data = src.read(1)

                print(f"  Data shape: {data.shape}")
                print(f"  Data CRS: {src.crs}")
                print(f"  Data bounds: {src.bounds}")

                # Create comprehensive visualization
                create_comprehensive_plot(
                    data,
                    src.transform,
                    src.crs,
                    nicaragua_gdf,
                    file_path.stem,
                    output_dir,
                )

        except Exception as e:
            print(f"  Error visualizing {file_path.name}: {e}")


def create_comprehensive_plot(
    data, transform, crs, nicaragua_gdf, filename, output_dir
):
    """Create a comprehensive visualization of the nightlights data."""

    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Original data
    im1 = axes[0, 0].imshow(data, cmap="viridis")
    axes[0, 0].set_title("Original Nightlights Data")
    plt.colorbar(im1, ax=axes[0, 0])

    # 2. Log scale
    log_data = np.log10(data + 1)
    im2 = axes[0, 1].imshow(log_data, cmap="viridis")
    axes[0, 1].set_title("Log Scale Nightlights")
    plt.colorbar(im2, ax=axes[0, 1])

    # 3. Square root scale
    sqrt_data = np.sqrt(data)
    im3 = axes[0, 2].imshow(sqrt_data, cmap="viridis")
    axes[0, 2].set_title("Square Root Scale Nightlights")
    plt.colorbar(im3, ax=axes[0, 2])

    # 4. Histogram of original data
    valid_data = data[data != -9999]  # Remove nodata
    axes[1, 0].hist(valid_data.flatten(), bins=50, alpha=0.7)
    axes[1, 0].set_title("Original Data Histogram")
    axes[1, 0].set_xlabel("Radiance")
    axes[1, 0].set_ylabel("Frequency")

    # 5. Histogram of log data
    valid_log_data = log_data[data != -9999]
    axes[1, 1].hist(valid_log_data.flatten(), bins=50, alpha=0.7)
    axes[1, 1].set_title("Log Scale Histogram")
    axes[1, 1].set_xlabel("Log Radiance")
    axes[1, 1].set_ylabel("Frequency")

    # 6. Map with boundary overlay
    # Calculate extent for proper display
    x_min = transform[2]
    y_max = transform[5]
    x_max = transform[2] + transform[0] * data.shape[1]
    y_min = transform[5] + transform[4] * data.shape[0]

    # Display the data with proper extent
    im6 = axes[1, 2].imshow(
        log_data, cmap="viridis", extent=[x_min, x_max, y_min, y_max]
    )
    axes[1, 2].set_title("Nightlights with Nicaragua Boundary")
    axes[1, 2].set_xlabel("Longitude")
    axes[1, 2].set_ylabel("Latitude")
    plt.colorbar(im6, ax=axes[1, 2])

    # Add Nicaragua boundary for reference
    nicaragua_gdf.plot(
        ax=axes[1, 2], edgecolor="red", facecolor="none", linewidth=2, alpha=0.8
    )

    plt.tight_layout()
    plt.savefig(
        output_dir / f"{filename}_comprehensive.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Create a separate map-style plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Display the data with proper extent
    im = ax.imshow(log_data, cmap="viridis", extent=[x_min, x_max, y_min, y_max])
    ax.set_title(f"Nicaragua Nightlights - {filename}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.colorbar(im, ax=ax, label="Log Radiance")

    # Add Nicaragua boundary for reference
    nicaragua_gdf.plot(ax=ax, edgecolor="red", facecolor="none", linewidth=2, alpha=0.8)

    plt.tight_layout()
    plt.savefig(output_dir / f"{filename}_map.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Print statistics
    print(f"  Data statistics:")
    print(f"    - Total pixels: {data.size}")
    print(f"    - Valid pixels: {len(valid_data)}")
    print(f"    - Coverage: {len(valid_data)/data.size*100:.1f}%")
    print(f"    - Min radiance: {valid_data.min():.2f}")
    print(f"    - Max radiance: {valid_data.max():.2f}")
    print(f"    - Mean radiance: {valid_data.mean():.2f}")
    print(f"    - Std radiance: {valid_data.std():.2f}")

    print(f"  Saved plots to: {output_dir}")


def plot_impact_prone_regions():
    """Create visualization of impact-prone regions based on population/nightlights ratio."""
    print("=" * 60)
    print("VISUALIZING IMPACT-PRONE REGIONS")
    print("=" * 60)

    # Load the preprocessed grid with vulnerability data
    grid_path = get_data_path(
        "data/preprocessed/nightlights/processed/nightlights_population_grid.gpkg"
    )

    if not Path(grid_path).exists():
        print(f"❌ Preprocessed grid not found: {grid_path}")
        print("Please run src/data_prep/preprocess_nightlights_grid.py first.")
        return

    print(f"Loading preprocessed grid: {grid_path}")
    grid_gdf = gpd.read_file(grid_path)

    # Get Nicaragua boundary
    nicaragua_gdf = get_nicaragua_boundary()

    # Create output directory
    output_dir = get_data_path("data/results/nightlights")
    output_dir.mkdir(exist_ok=True)

    # Create comprehensive impact-prone visualization
    create_impact_prone_plot(grid_gdf, nicaragua_gdf, output_dir)

    # Create standalone nightlights grid visualization
    create_nightlights_grid_plot(grid_gdf, nicaragua_gdf, output_dir)

    print(f"✅ Impact-prone regions visualization completed!")
    print(f"   Output saved to: {output_dir}")


def create_impact_prone_plot(grid_gdf, nicaragua_gdf, output_dir):
    """Create comprehensive visualization of impact-prone regions."""

    # Filter grid to only cells that intersect with Nicaragua
    nicaragua_bounds = nicaragua_gdf.total_bounds
    grid_in_nicaragua = grid_gdf.cx[
        nicaragua_bounds[0] : nicaragua_bounds[2],
        nicaragua_bounds[1] : nicaragua_bounds[3],
    ]

    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Vulnerability Index Map (Nicaragua only)
    impact_prone_gdf = grid_gdf[grid_gdf["impact_prone"] == True]

    # Plot all grid cells in light gray
    grid_gdf.plot(ax=axes[0, 0], color="lightgray", alpha=0.3, edgecolor="none")

    # Plot vulnerability index as color
    grid_gdf.plot(
        ax=axes[0, 0],
        column="vulnerability_index",
        cmap="Reds",
        alpha=0.7,
        legend=True,
        legend_kwds={"label": "Vulnerability Index"},
    )

    # Add Nicaragua boundary
    nicaragua_gdf.plot(ax=axes[0, 0], edgecolor="black", facecolor="none", linewidth=2)
    axes[0, 0].set_title("Vulnerability Index Map")
    axes[0, 0].set_xlabel("Longitude")
    axes[0, 0].set_ylabel("Latitude")

    # 2. Log Vulnerability Index Map
    grid_gdf["log_vulnerability"] = np.log10(grid_gdf["vulnerability_index"] + 1)
    grid_gdf.plot(
        ax=axes[0, 1],
        column="log_vulnerability",
        cmap="Reds",
        alpha=0.7,
        legend=True,
        legend_kwds={"label": "Log Vulnerability Index"},
    )
    nicaragua_gdf.plot(ax=axes[0, 1], edgecolor="black", facecolor="none", linewidth=2)
    axes[0, 1].set_title("Log Vulnerability Index Map")
    axes[0, 1].set_xlabel("Longitude")
    axes[0, 1].set_ylabel("Latitude")

    # 3. Impact-Prone Regions Map
    # Plot all grid cells in light gray
    grid_gdf.plot(ax=axes[0, 2], color="lightgray", alpha=0.3, edgecolor="none")

    # Plot impact-prone cells in red
    if len(impact_prone_gdf) > 0:
        impact_prone_gdf.plot(
            ax=axes[0, 2], color="red", alpha=0.7, edgecolor="darkred", linewidth=0.5
        )

    # Add Nicaragua boundary
    nicaragua_gdf.plot(ax=axes[0, 2], edgecolor="black", facecolor="none", linewidth=2)
    axes[0, 2].set_title("Impact-Prone Regions")
    axes[0, 2].set_xlabel("Longitude")
    axes[0, 2].set_ylabel("Latitude")

    # 4. Population Distribution Map (Log Scale)
    grid_gdf["log_population"] = np.log10(grid_gdf["population_total"] + 1)
    grid_gdf.plot(
        ax=axes[1, 0],
        column="log_population",
        cmap="Blues",
        alpha=0.7,
        legend=True,
        legend_kwds={"label": "Log10(Population + 1)"},
    )
    nicaragua_gdf.plot(ax=axes[1, 0], edgecolor="black", facecolor="none", linewidth=2)
    axes[1, 0].set_title("Population Distribution (Log Scale)")
    axes[1, 0].set_xlabel("Longitude")
    axes[1, 0].set_ylabel("Latitude")

    # 5. Nightlights Distribution Map (Log Scale)
    grid_gdf["log_nightlights"] = np.log10(grid_gdf["nightlights_sum"] + 1)
    grid_gdf.plot(
        ax=axes[1, 1],
        column="log_nightlights",
        cmap="YlOrRd",
        alpha=0.7,
        legend=True,
        legend_kwds={"label": "Log10(Nightlights + 1)"},
    )
    nicaragua_gdf.plot(ax=axes[1, 1], edgecolor="black", facecolor="none", linewidth=2)
    axes[1, 1].set_title("Nightlights Distribution (Log Scale)")
    axes[1, 1].set_xlabel("Longitude")
    axes[1, 1].set_ylabel("Latitude")

    # 6. Impact-Prone Regions with Statistics
    # Plot all grid cells in light gray
    grid_gdf.plot(ax=axes[1, 2], color="lightgray", alpha=0.3, edgecolor="none")

    # Plot impact-prone cells in red
    if len(impact_prone_gdf) > 0:
        impact_prone_gdf.plot(
            ax=axes[1, 2], color="red", alpha=0.7, edgecolor="darkred", linewidth=0.5
        )

    # Add Nicaragua boundary
    nicaragua_gdf.plot(ax=axes[1, 2], edgecolor="black", facecolor="none", linewidth=2)

    # Add statistics as text
    total_cells = len(grid_gdf)
    impact_prone_cells = len(impact_prone_gdf)
    impact_prone_percentage = impact_prone_cells / total_cells * 100

    stats_text = f"Impact-Prone: {impact_prone_cells}/{total_cells} ({impact_prone_percentage:.1f}%)"
    axes[1, 2].text(
        0.02,
        0.98,
        stats_text,
        transform=axes[1, 2].transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    axes[1, 2].set_title("Impact-Prone Regions Summary")
    axes[1, 2].set_xlabel("Longitude")
    axes[1, 2].set_ylabel("Latitude")

    plt.tight_layout()
    plt.savefig(
        output_dir / "impact_prone_regions_comprehensive.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Create a separate focused map-style plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Plot all grid cells in light gray
    grid_gdf.plot(ax=ax, color="lightgray", alpha=0.3, edgecolor="none")

    # Plot impact-prone cells in red
    if len(impact_prone_gdf) > 0:
        impact_prone_gdf.plot(
            ax=ax, color="red", alpha=0.7, edgecolor="darkred", linewidth=0.5
        )

    # Add Nicaragua boundary
    nicaragua_gdf.plot(ax=ax, edgecolor="black", facecolor="none", linewidth=2)

    ax.set_title(
        "Impact-Prone Regions in Nicaragua\n(Population/Nightlights Ratio > Threshold)"
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Add statistics as text
    stats_text = f"Impact-Prone Cells: {impact_prone_cells}/{total_cells} ({impact_prone_percentage:.1f}%)"
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(
        output_dir / "impact_prone_regions_map.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def create_nightlights_grid_plot(grid_gdf, nicaragua_gdf, output_dir):
    """Create a standalone visualization of the nightlights grid."""

    # Create figure with multiple subplots for different visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Nightlights Distribution (Log Scale) - Dark to bright like actual nightlights
    grid_gdf["log_nightlights"] = np.log10(grid_gdf["nightlights_sum"] + 1)
    grid_gdf.plot(
        ax=axes[0, 0],
        column="log_nightlights",
        cmap="plasma",  # Dark purple to bright yellow - more vibrant than viridis
        alpha=1.0,  # Full opacity for vibrant colors
        legend=True,
        legend_kwds={"label": "Log10(Nightlights + 1)"},
    )
    nicaragua_gdf.plot(ax=axes[0, 0], edgecolor="white", facecolor="none", linewidth=2)
    axes[0, 0].set_title("Nightlights Distribution (Log Scale)")
    axes[0, 0].set_xlabel("Longitude")
    axes[0, 0].set_ylabel("Latitude")

    # 2. Vulnerability Index - Red color scheme
    grid_gdf.plot(
        ax=axes[0, 1],
        column="vulnerability_index",
        cmap="Reds",
        alpha=0.7,
        legend=True,
        legend_kwds={"label": "Vulnerability Index"},
    )
    nicaragua_gdf.plot(ax=axes[0, 1], edgecolor="black", facecolor="none", linewidth=2)
    axes[0, 1].set_title("Vulnerability Index (Population/Nightlights)")
    axes[0, 1].set_xlabel("Longitude")
    axes[0, 1].set_ylabel("Latitude")

    # 3. Population Distribution (Log Scale) - Blue color scheme
    grid_gdf["log_population"] = np.log10(grid_gdf["population_total"] + 1)
    grid_gdf.plot(
        ax=axes[1, 0],
        column="log_population",
        cmap="Blues",
        alpha=0.7,
        legend=True,
        legend_kwds={"label": "Log10(Population + 1)"},
    )
    nicaragua_gdf.plot(ax=axes[1, 0], edgecolor="black", facecolor="none", linewidth=2)
    axes[1, 0].set_title("Population Distribution (Log Scale)")
    axes[1, 0].set_xlabel("Longitude")
    axes[1, 0].set_ylabel("Latitude")

    # 4. Impact-Prone Regions - Binary visualization
    impact_prone_gdf = grid_gdf[grid_gdf["impact_prone"] == True]

    # Plot all grid cells in light gray
    grid_gdf.plot(ax=axes[1, 1], color="lightgray", alpha=0.3, edgecolor="none")

    # Plot impact-prone cells in red
    if len(impact_prone_gdf) > 0:
        impact_prone_gdf.plot(
            ax=axes[1, 1], color="red", alpha=0.7, edgecolor="darkred", linewidth=0.5
        )

    # Add Nicaragua boundary
    nicaragua_gdf.plot(ax=axes[1, 1], edgecolor="black", facecolor="none", linewidth=2)

    # Add statistics
    total_cells = len(grid_gdf)
    impact_prone_cells = len(impact_prone_gdf)
    impact_prone_percentage = impact_prone_cells / total_cells * 100

    stats_text = f"Impact-Prone: {impact_prone_cells}/{total_cells} ({impact_prone_percentage:.1f}%)"
    axes[1, 1].text(
        0.02,
        0.98,
        stats_text,
        transform=axes[1, 1].transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    axes[1, 1].set_title("Impact-Prone Regions")
    axes[1, 1].set_xlabel("Longitude")
    axes[1, 1].set_ylabel("Latitude")

    plt.tight_layout()
    plt.savefig(
        output_dir / "nightlights_grid_comprehensive.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Create a standalone nightlights-only visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Plot nightlights with proper color scheme - dark for low, bright for high (like actual nightlights)
    grid_gdf.plot(
        ax=ax,
        column="log_nightlights",
        cmap="plasma",  # Dark purple to bright yellow - more vibrant than viridis
        alpha=1.0,  # Full opacity for vibrant colors
        legend=True,
        legend_kwds={"label": "Log10(Nightlights + 1)"},
    )

    # Add Nicaragua boundary
    nicaragua_gdf.plot(ax=ax, edgecolor="white", facecolor="none", linewidth=2)

    ax.set_title("Nicaragua Nightlights Distribution")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    plt.tight_layout()
    plt.savefig(
        output_dir / "nightlights_distribution.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(f"✅ Created nightlights grid visualizations:")
    print(f"   - nightlights_grid_comprehensive.png (4-panel analysis)")
    print(f"   - nightlights_distribution.png (standalone nightlights map)")


def main():
    """Main function."""
    plot_nightlights_data()
    plot_impact_prone_regions()


if __name__ == "__main__":
    main()
