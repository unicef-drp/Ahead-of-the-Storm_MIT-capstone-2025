#!/usr/bin/env python3
"""
Main script to download Nicaragua data from OpenStreetMap.

This script downloads various types of geographic data for Nicaragua
including roads, schools, hospitals, population centers, vaccination centers,
childcare facilities, and baby goods stores.
"""

import logging
import sys
import yaml
from pathlib import Path

from src.data_prep.data_downloader import DataDownloader


def load_config(config_path: str = "config/data_config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(config: dict):
    """Set up logging configuration from config file."""
    log_config = config.get("logging", {})

    logging.basicConfig(
        level=getattr(logging, log_config.get("level", "INFO")),
        format=log_config.get(
            "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ),
        handlers=[
            logging.FileHandler(log_config.get("file", "data_download.log")),
            logging.StreamHandler(sys.stdout),
        ],
    )


def main():
    """Main function to download Nicaragua data."""
    # Load configuration
    config = load_config()

    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)

    logger.info("Starting Nicaragua data download")

    try:
        # Initialize data downloader
        downloader = DataDownloader()

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
