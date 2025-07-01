#!/usr/bin/env python3
"""
Main script to download Nicaragua data from OpenStreetMap.

This script downloads various types of geographic data for Nicaragua
including roads, schools, hospitals, vaccination centers,
childcare facilities, and baby goods stores.
"""

import sys

from src.data_prep.osm_helper import OSMDataDownloader
from src.utils.config_utils import load_config
from src.utils.logging_utils import setup_logging, get_logger


def main():
    """Main function to download Nicaragua data."""
    # Setup logging
    logger = setup_logging(__name__)

    logger.info("Starting Nicaragua data download")

    try:
        # Initialize data downloader
        downloader = OSMDataDownloader()

        # Download all categories
        results = downloader.download_all()

        # Print summary
        logger.info("Download completed successfully!")
        logger.info("Downloaded files:")
        for category, filepath in results.items():
            logger.info(f"  {category}: {filepath}")

        logger.info(f"Total categories downloaded: {len(results)}")

    except Exception as e:
        logger.error(f"Error during data download: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
