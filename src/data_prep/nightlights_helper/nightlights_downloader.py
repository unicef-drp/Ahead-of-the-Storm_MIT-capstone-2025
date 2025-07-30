"""
Nightlights Data Downloader for VIIRS DNB Data

This module provides functionality to download VIIRS Day/Night Band (DNB) nightlights data
from NASA Earthdata using the earthaccess library.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
import earthaccess
from rasterio.io import MemoryFile
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import rasterio
from shapely.geometry import box

from src.utils.config_utils import load_config, get_config_value, get_project_root
from src.utils.logging_utils import setup_logging
from src.utils.path_utils import get_data_path, ensure_directory
from src.utils.hurricane_geom import get_nicaragua_boundary


class NightlightsDownloader:
    """
    Downloads VIIRS DNB nightlights data from NASA Earthdata.

    This class handles downloading and processing VIIRS Day/Night Band (DNB) data
    for nightlights analysis, specifically for Nicaragua.
    """

    def __init__(self, config_path: str = "config/nightlights_config.yaml"):
        """
        Initialize the nightlights downloader with configuration.

        Args:
            config_path: Path to the configuration YAML file
        """
        self.project_root = get_project_root()
        self.config = load_config(config_path)
        self.logger = setup_logging(__name__)
        self._setup_directories()
        self._authenticate_earthdata()

    def _setup_directories(self):
        """Create necessary directories for data storage."""
        download_config = self.config.get("download", {})
        output_dir = get_config_value(
            download_config, "output_directory", "data/raw/nightlights"
        )
        self.output_dir = get_data_path(output_dir)
        ensure_directory(self.output_dir)
        self.logger.info(f"Created output directory: {self.output_dir}")

        processing_config = self.config.get("processing", {})
        processed_dir = get_config_value(
            processing_config,
            "processed_output_directory",
            "data/preprocessed/nightlights/processed",
        )
        self.processed_dir = get_data_path(processed_dir)
        ensure_directory(self.processed_dir)
        self.logger.info(f"Created processed directory: {self.processed_dir}")

    def _authenticate_earthdata(self):
        """Authenticate with NASA Earthdata."""
        try:
            earthaccess.login()
            self.logger.info("Successfully authenticated with NASA Earthdata")
        except Exception as e:
            self.logger.error(f"Failed to authenticate with Earthdata: {e}")
            self.logger.info("Please ensure you have valid Earthdata credentials")
            raise

    def _get_nicaragua_bounds(self) -> Tuple[float, float, float, float]:
        """
        Get the bounding box for Nicaragua.

        Returns:
            Tuple of (min_lon, min_lat, max_lon, max_lat)
        """
        nicaragua_gdf = get_nicaragua_boundary()
        bounds = nicaragua_gdf.total_bounds
        # Add small buffer to ensure complete coverage
        buffer = 0.1
        return (
            bounds[0] - buffer,  # min_lon
            bounds[1] - buffer,  # min_lat
            bounds[2] + buffer,  # max_lon
            bounds[3] + buffer,  # max_lat
        )

    def _search_nightlights_data(
        self, start_date: str, end_date: str, product_type: str = "annual"
    ) -> List:
        """
        Search for VIIRS DNB nightlights data within the specified date range.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            product_type: Type of product to search for ("annual", "monthly", or "daily")

        Returns:
            List of available datasets
        """
        bounds = self._get_nicaragua_bounds()

        # Choose product based on type
        if product_type == "annual":
            short_name = "VNP46A4"  # Annual composite
        elif product_type == "monthly":
            short_name = "VNP46A3"  # Monthly composite
        else:
            short_name = "VNP46A1"  # Daily data

        # Search for VIIRS DNB data
        datasets = earthaccess.search_data(
            short_name=short_name,
            temporal=(start_date, end_date),
            bounding_box=bounds,
        )

        self.logger.info(f"Found {len(datasets)} {short_name} datasets")
        return datasets

    def _download_and_process_dataset(self, dataset, output_path: str) -> bool:
        """
        Download and process a single VIIRS dataset.

        Args:
            dataset: Earthdata dataset object
            output_path: Path to save the processed data

        Returns:
            True if successful, False otherwise
        """
        try:
            # Download the dataset
            files = earthaccess.download(dataset, output_path)

            if not files:
                self.logger.warning(f"No files downloaded for dataset {dataset}")
                return False

            # Process the downloaded files
            for file_path in files:
                if file_path.endswith(".h5"):
                    self._process_h5_file(file_path, output_path)
                elif file_path.endswith(".tif"):
                    self._process_tif_file(file_path, output_path)

            return True

        except Exception as e:
            self.logger.error(f"Error processing dataset {dataset}: {e}")
            return False

    def _process_h5_file(self, file_path: str, output_dir: str):
        """
        Process HDF5 file containing VIIRS DNB data.

        Args:
            file_path: Path to the HDF5 file
            output_dir: Output directory for processed data
        """
        try:
            import h5py

            with h5py.File(file_path, "r") as f:
                # Explore the structure to find DNB data
                print(f"Keys in {file_path}: {list(f.keys())}")

                # Look for DNB data in various possible locations
                dnb_data = None
                dnb_key = None

                # Paths for annual composite data (VNP46A4)
                annual_composite_paths = [
                    "HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/AllAngle_Composite_Snow_Free",
                    "HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/NearNadir_Composite_Snow_Free",
                    "HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/OffNadir_Composite_Snow_Free",
                ]

                # Paths for daily data (VNP46A1)
                daily_paths = [
                    "HDFEOS/GRIDS/VNP_Grid_DNB/Data Fields/DNB_At_Sensor_Radiance_500m",
                    "HDFEOS/GRIDS/VNP_Grid_DNB/Data Fields/DNB_At_Sensor_Radiance_500m_Linear",
                    "HDFEOS/GRIDS/VNP_Grid_DNB/Data Fields/DNB_At_Sensor_Radiance_500m_Stripe_Removed",
                    "HDFEOS/GRIDS/VNP_Grid_DNB/Data Fields/DNB_At_Sensor_Radiance_500m_Stripe_Removed_Linear",
                    "HDFEOS/GRIDS/VNP_Grid_DNB/Data Fields/DNB_At_Sensor_Radiance_500m_Stripe_Removed_Linear_Unit_Scale_Factor",
                    "HDFEOS/GRIDS/VNP_Grid_DNB/Data Fields/DNB_At_Sensor_Radiance_500m_Stripe_Removed_Linear_Unit_Scale_Factor_Uncertainty",
                    "HDFEOS/GRIDS/VNP_Grid_DNB/Data Fields/DNB_At_Sensor_Radiance_500m_Stripe_Removed_Linear_Unit_Scale_Factor_Uncertainty_Index",
                    "HDFEOS/GRIDS/VNP_Grid_DNB/Data Fields/DNB_At_Sensor_Radiance_500m_Stripe_Removed_Linear_Unit_Scale_Factor_Uncertainty_Index_Uncertainty",
                    "HDFEOS/GRIDS/VNP_Grid_DNB/Data Fields/DNB_At_Sensor_Radiance_500m_Stripe_Removed_Linear_Unit_Scale_Factor_Uncertainty_Index_Uncertainty_Uncertainty",
                    "HDFEOS/GRIDS/VNP_Grid_DNB/Data Fields/DNB_At_Sensor_Radiance_500m_Stripe_Removed_Linear_Unit_Scale_Factor_Uncertainty_Index_Uncertainty_Uncertainty_Uncertainty",
                ]

                # Try annual composite paths first (preferred)
                for path in annual_composite_paths:
                    if path in f:
                        dnb_data = f[path][:]
                        dnb_key = path
                        break

                # If not found, try daily paths
                if dnb_data is None:
                    for path in daily_paths:
                        if path in f:
                            dnb_data = f[path][:]
                            dnb_key = path
                            break

                if dnb_data is None:
                    # Try to find any dataset with 'DNB' in the name
                    def find_dnb(name, obj):
                        nonlocal dnb_data, dnb_key
                        if isinstance(obj, h5py.Dataset) and "DNB" in name.upper():
                            dnb_data = obj[:]
                            dnb_key = name

                    f.visititems(find_dnb)

                if dnb_data is None:
                    self.logger.warning(f"No DNB data found in {file_path}")
                    return

                # Convert to GeoTIFF
                output_path = (
                    Path(output_dir) / f"{Path(file_path).stem}_nightlights.tif"
                )

                # Get the actual lat/lon arrays from the HDF5 file
                lat_array = f["HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/lat"][:]
                lon_array = f["HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields/lon"][:]

                # Calculate the correct transform from the lat/lon arrays
                # The data is in a regular grid, so we can calculate the pixel size
                lat_resolution = (lat_array[-1] - lat_array[0]) / (
                    lat_array.shape[0] - 1
                )
                lon_resolution = (lon_array[-1] - lon_array[0]) / (
                    lon_array.shape[0] - 1
                )

                # Create the correct transform
                # Note: lat_array goes from 10 to 20 (south to north), so we need to flip it
                transform = rasterio.Affine(
                    lon_resolution,
                    0,
                    lon_array[0],  # x_res, x_skew, x_origin
                    0,
                    lat_resolution,
                    lat_array[0],  # y_skew, y_res, y_origin (start from lat_array[0])
                )

                with rasterio.open(
                    output_path,
                    "w",
                    driver="GTiff",
                    height=dnb_data.shape[0],
                    width=dnb_data.shape[1],
                    count=1,
                    dtype=dnb_data.dtype,
                    crs="EPSG:4326",
                    transform=transform,
                ) as dst:
                    dst.write(dnb_data, 1)

                self.logger.info(f"Processed {file_path} -> {output_path}")

        except Exception as e:
            self.logger.error(f"Error processing H5 file {file_path}: {e}")

    def _process_tif_file(self, file_path: str, output_dir: str):
        """
        Process GeoTIFF file containing VIIRS DNB data.

        Args:
            file_path: Path to the GeoTIFF file
            output_dir: Output directory for processed data
        """
        try:
            # Read the GeoTIFF
            with rasterio.open(file_path) as src:
                # Get Nicaragua boundary for masking
                nicaragua_gdf = get_nicaragua_boundary()

                # Mask the data to Nicaragua
                masked_data, masked_transform = mask(
                    src, nicaragua_gdf.geometry, crop=True, nodata=src.nodata
                )

                # Save masked data
                output_path = Path(output_dir) / f"{Path(file_path).stem}_masked.tif"

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

                self.logger.info(f"Processed {file_path} -> {output_path}")

        except Exception as e:
            self.logger.error(f"Error processing TIF file {file_path}: {e}")

    def download_nightlights_data(
        self,
        start_date: str = "2020-01-01",
        end_date: str = "2020-12-31",
        product_type: str = "annual",
    ) -> Dict[str, int]:
        """
        Download VIIRS DNB nightlights data for the specified date range.
        Uses annual or monthly composites to avoid cloud cover issues.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            product_type: Type of product ("annual", "monthly", or "daily")

        Returns:
            Dictionary with download statistics
        """
        self.logger.info(
            f"Starting nightlights data download for {start_date} to {end_date} using {product_type} composites"
        )

        # Search for available datasets
        datasets = self._search_nightlights_data(start_date, end_date, product_type)

        if not datasets:
            self.logger.warning("No datasets found for the specified date range")
            return {"successful": 0, "failed": 0, "skipped": 0, "total": 0}

        # Download and process datasets
        successful = 0
        failed = 0
        skipped = 0
        downloaded_files = []

        for i, dataset in enumerate(datasets, 1):
            self.logger.info(f"Processing dataset {i}/{len(datasets)}")

            try:
                if self._download_and_process_dataset(dataset, self.output_dir):
                    successful += 1
                    # Collect downloaded file paths for averaging
                    downloaded_files.extend(self._get_downloaded_files())
                else:
                    failed += 1
            except Exception as e:
                self.logger.error(f"Error processing dataset {i}: {e}")
                failed += 1

        # If we have multiple monthly composites, average them
        if product_type == "monthly" and len(downloaded_files) > 1:
            self.logger.info(f"Averaging {len(downloaded_files)} monthly composites")
            self._average_nightlights_files(downloaded_files)

        stats = {
            "successful": successful,
            "failed": failed,
            "skipped": skipped,
            "total": len(datasets),
        }

        self.logger.info(f"Download completed: {stats}")
        return stats

    def _get_downloaded_files(self) -> List[str]:
        """Get list of recently downloaded nightlights files."""
        nightlights_files = list(self.output_dir.glob("*nightlights*.tif"))
        return [str(f) for f in nightlights_files]

    def _average_nightlights_files(self, file_paths: List[str]):
        """Average multiple nightlights files into a single composite file."""
        if not file_paths:
            return

        try:
            # Read all files and stack them
            stacked_data = []
            reference_transform = None
            reference_crs = None

            for file_path in file_paths:
                with rasterio.open(file_path) as src:
                    data = src.read(1)  # Read first band
                    stacked_data.append(data)

                    if reference_transform is None:
                        reference_transform = src.transform
                        reference_crs = src.crs

            # Convert to numpy array and average
            stacked_array = np.stack(stacked_data)
            averaged_data = np.mean(stacked_array, axis=0)

            # Save averaged file
            output_path = self.processed_dir / "nightlights_averaged.tif"

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

            self.logger.info(f"Saved averaged nightlights to {output_path}")

        except Exception as e:
            self.logger.error(f"Error averaging nightlights files: {e}")

    def get_download_summary(self) -> Dict:
        """
        Get a summary of the downloaded nightlights data.

        Returns:
            Dictionary with download summary information
        """
        return {
            "data_type": "VIIRS DNB Nightlights",
            "source": "NASA Earthdata",
            "coverage": "Nicaragua",
            "output_directory": str(self.output_dir),
            "processed_directory": str(self.processed_dir),
            "date_range": {"start": "2020-01-01", "end": "2020-12-31"},
        }


def main():
    """Main function to download nightlights data."""
    print("=" * 60)
    print("NIGHTLIGHTS DATA DOWNLOAD")
    print("VIIRS DNB from NASA Earthdata")
    print("=" * 60)

    try:
        # Initialize the nightlights downloader
        print("Initializing nightlights data downloader...")
        downloader = NightlightsDownloader()

        # Download the nightlights data (try annual first, then monthly)
        print("\nStarting data download...")

        # Try annual composite first
        download_stats = downloader.download_nightlights_data(product_type="annual")

        if download_stats["successful"] == 0:
            print("No annual composites found, trying monthly composites...")
            download_stats = downloader.download_nightlights_data(
                product_type="monthly"
            )

        if download_stats["successful"] == 0:
            print("No monthly composites found, trying daily data...")
            download_stats = downloader.download_nightlights_data(product_type="daily")

        # Get and display summary
        print("\nGenerating download summary...")
        summary = downloader.get_download_summary()

        # Display results
        print("\n" + "=" * 60)
        print("DOWNLOAD COMPLETED!")
        print("=" * 60)
        print(f"Data Type: {summary['data_type']}")
        print(f"Source: {summary['source']}")
        print(f"Coverage: {summary['coverage']}")
        print(
            f"Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}"
        )
        print(f"Output Directory: {summary['output_directory']}")
        print(f"Processed Directory: {summary['processed_directory']}")
        print("\nDownload Statistics:")
        print(f"  - Successful: {download_stats['successful']}")
        print(f"  - Failed: {download_stats['failed']}")
        print(f"  - Skipped: {download_stats['skipped']}")
        print(f"  - Total: {download_stats['total']}")
        print("=" * 60)

        if download_stats["failed"] > 0:
            print(f"\n⚠️  Warning: {download_stats['failed']} downloads failed.")
            print("Check the logs for details on failed downloads.")

        print("\n✅ Nightlights data download completed!")
        print("You can now use the downloaded data for vulnerability analysis.")

    except FileNotFoundError as e:
        print(f"❌ Configuration error: {e}")
        print(
            "Please ensure the configuration file exists at: config/nightlights_config.yaml"
        )
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during download: {e}")
        print("Check the logs for more details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
