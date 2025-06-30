#!/usr/bin/env python3
"""
Download Nicaragua population data from WorldPop.

This script downloads high-resolution population data including age/sex structures.
"""

import sys
import logging
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from data_prep.census_helper.population_downloader import PopulationDownloader


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("population_download.log"),
        ],
    )


def main():
    """Main function to download population data."""
    parser = argparse.ArgumentParser(description="Download Nicaragua population data")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: download only first 4 age/sex files",
    )
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting Nicaragua population data download")

    try:
        # Initialize downloader
        downloader = PopulationDownloader()

        # Download all data
        results = downloader.download_all(test_mode=args.test)

        if results:
            logger.info("Population data download completed successfully!")
            logger.info("Downloaded files:")

            for data_type, filepath in results.items():
                logger.info(f"  {data_type}: {filepath}")

            logger.info(f"Total data types downloaded: {len(results)}")

            # Summary
            logger.info("\nData Summary:")
            logger.info("- Total population: High-resolution spatial data")
            if args.test:
                logger.info("- Age/sex structures: Test mode (4 files)")
            else:
                logger.info("- Age/sex structures: Complete dataset (36 files)")
            logger.info("- Spatial resolution: 100m grid")
            logger.info("- Coverage: Nicaragua")

        else:
            logger.error("No data was downloaded successfully")
            return 1

    except Exception as e:
        logger.error(f"Error during population data download: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
