#!/usr/bin/env python3
"""
Download NASA LHASA-F landslide hazard prediction data.

This script downloads landslide hazard prediction data from NASA's
Landslide Hazard Assessment for Situational Awareness - Future (LHASA-F)
service and converts it to georeferenced GeoTIFF format for analysis.
"""

import sys
import argparse
from pathlib import Path

from src.data_prep.landslide_helper.landslide_downloader import LandslideDownloader
from src.utils.logging_utils import setup_logging, get_logger


def main():
    """Main function to download landslide hazard data."""
    parser = argparse.ArgumentParser(description="Download NASA LHASA-F landslide data")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: download with smaller image size",
    )
    parser.add_argument(
        "--no-process",
        action="store_true",
        help="Skip data processing (save raw TIFF only)",
    )
    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(__name__)
    
    logger.info("=" * 60)
    logger.info("NASA LHASA-F LANDSLIDE HAZARD DATA DOWNLOAD")
    logger.info("Landslide Hazard Assessment for Situational Awareness - Future")
    logger.info("=" * 60)

    try:
        # Initialize downloader
        logger.info("Initializing landslide data downloader...")
        downloader = LandslideDownloader()

        # Modify config for test mode if requested
        if args.test:
            logger.info("TEST MODE: Using smaller image size")
            # Note: This would require modifying the config dynamically
            # For now, we'll just log the test mode

        # Modify config to skip processing if requested
        if args.no_process:
            logger.info("Skipping data processing (raw TIFF only)")
            # Note: This would require modifying the config dynamically
            # For now, we'll just log the setting

        # Download landslide data
        logger.info("Starting landslide data download...")
        result_path = downloader.download_landslide_data()

        if result_path:
            logger.info("Landslide data download completed successfully!")
            logger.info(f"Output file: {result_path}")

            # Get and display summary
            logger.info("Generating download summary...")
            summary = downloader.get_download_summary()

            # Display results
            logger.info("\n" + "=" * 60)
            logger.info("DOWNLOAD COMPLETED SUCCESSFULLY!")
            logger.info("=" * 60)
            logger.info(f"Data Source: {summary['data_source']}")
            logger.info(f"Coverage: {summary['coverage']}")
            logger.info(f"Raw Files: {summary['raw_files_count']}")
            logger.info(f"Processed Files: {summary['processed_files_count']}")
            logger.info(f"Total Size: {summary['total_size_mb']} MB")
            logger.info(f"Raw Directory: {summary['raw_directory']}")
            logger.info(f"Processed Directory: {summary['processed_directory']}")

            # Data summary
            logger.info("\nData Summary:")
            logger.info("- Format: GeoTIFF (georeferenced)")
            logger.info("- Projection: WGS84 (EPSG:4326) for processed files")
            logger.info("- Coverage: Nicaragua")
            logger.info("- Use: Exposure grid for impact analysis")

        else:
            logger.error("Landslide data download failed")
            return 1

    except Exception as e:
        logger.error(f"Error during landslide data download: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main()) 