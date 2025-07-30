#!/usr/bin/env python3
"""
Main script to download VIIRS DNB nightlights data from NASA Earthdata.

This script downloads VIIRS Day/Night Band (DNB) data for nightlights analysis
in Nicaragua, which will be used for vulnerability assessment.

Usage:
    python download_nightlights_data.py
"""

import sys
import os
from pathlib import Path

from src.data_prep.nightlights_helper.nightlights_downloader import (
    NightlightsDownloader,
)


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

        # Download the nightlights data
        print("\nStarting data download...")
        download_stats = downloader.download_nightlights_data()

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
