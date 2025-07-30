#!/usr/bin/env python3
"""
Process nightlights data for Nicaragua.

This script processes downloaded nightlights data and creates properly aligned
Nicaragua-specific datasets with correct CRS handling.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas as gpd
from src.utils.hurricane_geom import get_nicaragua_boundary
from src.utils.config_utils import load_config, get_config_value, get_project_root
from src.utils.path_utils import get_data_path, ensure_directory


def process_nightlights_data():
    """Process nightlights data and create Nicaragua-specific dataset."""
    print("=" * 60)
    print("PROCESSING NIGHTLIGHTS DATA")
    print("=" * 60)

    # Load configuration
    config = load_config("config/nightlights_config.yaml")
    project_root = get_project_root()

    # Setup paths
    raw_dir = get_data_path("data/raw/nightlights")
    processed_dir = get_data_path("data/preprocessed/nightlights/processed")
    ensure_directory(processed_dir)

    print(f"Raw data directory: {raw_dir}")
    print(f"Processed data directory: {processed_dir}")

    # Find all nightlights files
    nightlights_files = list(raw_dir.glob("*nightlights*.tif"))

    if not nightlights_files:
        print("No nightlights files found in raw directory!")
        return

    print(f"Found {len(nightlights_files)} nightlights files")

    # Get Nicaragua boundary
    nicaragua_gdf = get_nicaragua_boundary()
    print(f"Nicaragua boundary CRS: {nicaragua_gdf.crs}")
    print(f"Nicaragua boundary bounds: {nicaragua_gdf.total_bounds}")

    # Process each file
    processed_files = []

    for file_path in nightlights_files:
        print(f"\nProcessing: {file_path.name}")

        try:
            with rasterio.open(file_path) as src:
                print(f"  Source CRS: {src.crs}")
                print(f"  Source bounds: {src.bounds}")
                print(f"  Source shape: {src.shape}")

                # Check if CRS match
                if src.crs != nicaragua_gdf.crs:
                    print(f"  CRS mismatch! Reprojecting boundary...")
                    nicaragua_reprojected = nicaragua_gdf.to_crs(src.crs)
                else:
                    nicaragua_reprojected = nicaragua_gdf

                # Mask the data to Nicaragua
                masked_data, masked_transform = mask(
                    src, nicaragua_reprojected.geometry, crop=True, nodata=src.nodata
                )

                print(f"  Masked data shape: {masked_data.shape}")

                # Save masked data
                output_path = processed_dir / f"{file_path.stem}_nicaragua.tif"

                with rasterio.open(
                    output_path,
                    "w",
                    driver="GTiff",
                    height=masked_data.shape[1],
                    width=masked_data.shape[2],
                    count=1,
                    dtype=masked_data.dtype,
                    crs=src.crs,
                    transform=masked_transform,
                    nodata=src.nodata,
                ) as dst:
                    dst.write(masked_data)

                processed_files.append(output_path)
                print(f"  Saved to: {output_path}")

        except Exception as e:
            print(f"  Error processing {file_path.name}: {e}")

    # If we have multiple files, create an average
    if len(processed_files) > 1:
        print(f"\nCreating average from {len(processed_files)} files...")
        create_average_nightlights(processed_files, processed_dir)

    print("\n✅ Nightlights processing completed!")


def create_average_nightlights(file_paths, output_dir):
    """Create an average from multiple nightlights files."""
    try:
        # Read all files and stack them
        stacked_data = []
        reference_transform = None
        reference_crs = None

        for file_path in file_paths:
            with rasterio.open(file_path) as src:
                data = src.read(1)  # Read first band
                # Handle no-data values
                data = np.where(data == src.nodata, 0, data)
                stacked_data.append(data)

                if reference_transform is None:
                    reference_transform = src.transform
                    reference_crs = src.crs

        # Convert to numpy array and average
        stacked_array = np.stack(stacked_data)
        averaged_data = np.mean(stacked_array, axis=0)

        # Save averaged file
        output_path = output_dir / "nightlights_nicaragua_average.tif"

        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=averaged_data.shape[0],
            width=averaged_data.shape[1],
            count=1,
            dtype=averaged_data.dtype,
            crs=reference_crs,
            transform=reference_transform,
            nodata=-9999,
        ) as dst:
            dst.write(averaged_data, 1)

        print(f"Saved averaged nightlights to: {output_path}")

        # Create verification plot
        create_verification_plot(averaged_data, reference_transform, reference_crs)

    except Exception as e:
        print(f"Error creating average: {e}")


def create_verification_plot(data, transform, crs):
    """Create a verification plot of the processed data."""
    print("Creating verification plot...")

    # Create output directory
    output_dir = get_data_path("data/results/nightlights")
    output_dir.mkdir(exist_ok=True)

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Original data
    im1 = axes[0, 0].imshow(data, cmap="viridis")
    axes[0, 0].set_title("Nicaragua Nightlights - Processed")
    plt.colorbar(im1, ax=axes[0, 0])

    # Log scale
    log_data = np.log10(data + 1)
    im2 = axes[0, 1].imshow(log_data, cmap="viridis")
    axes[0, 1].set_title("Nicaragua Nightlights - Log Scale")
    plt.colorbar(im2, ax=axes[0, 1])

    # Histogram
    valid_data = data[data != -9999]  # Remove nodata
    axes[1, 0].hist(valid_data.flatten(), bins=50, alpha=0.7)
    axes[1, 0].set_title("Nightlights Histogram (Valid Data Only)")
    axes[1, 0].set_xlabel("Radiance")
    axes[1, 0].set_ylabel("Frequency")

    # Log histogram
    valid_log_data = log_data[data != -9999]
    axes[1, 1].hist(valid_log_data.flatten(), bins=50, alpha=0.7)
    axes[1, 1].set_title("Log Scale Histogram (Valid Data Only)")
    axes[1, 1].set_xlabel("Log Radiance")
    axes[1, 1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(
        output_dir / "nicaragua_nightlights_processed.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(f"Saved verification plot to: {output_dir}")


def main():
    """Main function."""
    process_nightlights_data()


if __name__ == "__main__":
    main()
