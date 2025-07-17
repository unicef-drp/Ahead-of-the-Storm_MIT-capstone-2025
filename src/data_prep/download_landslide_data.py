#!/usr/bin/env python3
"""
Download NASA LHASA-F landslide hazard prediction data (direct download).

This script downloads the latest landslide hazard probability GeoTIFF(s) from NASA's
Landslide Hazard Assessment for Situational Awareness - Future (LHASA-F) directory.
"""

import sys
from pathlib import Path
from src.data_prep.landslide_helper.landslide_downloader import LandslideDownloader
from src.utils.logging_utils import setup_logging

def main():
    """Main function to download landslide hazard data."""
    logger = setup_logging(__name__)
    logger.info("=" * 60)
    logger.info("NASA LHASA-F LANDSLIDE HAZARD DATA DOWNLOAD (DIRECT)")
    logger.info("Landslide Hazard Assessment for Situational Awareness - Future")
    logger.info("=" * 60)
    try:
        # Initialize downloader
        logger.info("Initializing landslide data downloader...")
        downloader = LandslideDownloader()
        # Download landslide data (latest 48h forecast)
        logger.info("Starting landslide data download...")
        result_paths = downloader.download_landslide_data()
        if result_paths:
            logger.info("Landslide data download completed successfully!")
            for path in result_paths:
                logger.info(f"Output file: {path}")
            # Get and display summary
            logger.info("Generating download summary...")
            summary = downloader.get_download_summary()
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
            logger.info("\nData Summary:")
            logger.info("- Format: GeoTIFF (probability)")
            logger.info("- Projection: WGS84 (EPSG:4326) for processed files")
            logger.info("- Coverage: Global")
            logger.info("- Use: Probability grid for impact analysis")
        else:
            logger.error("Landslide data download failed")
            return 1
    except Exception as e:
        logger.error(f"Error during landslide data download: {e}")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main()) 