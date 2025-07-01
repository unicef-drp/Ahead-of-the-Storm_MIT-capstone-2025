#!/usr/bin/env python3
"""
Main script to download Hurricane Rafael (2024) data from Google Weather Lab's FNV3 model.

This script downloads ensemble forecast data for Hurricane Rafael, including all ensemble members,
lead times, and relevant variables (wind speed, pressure, etc.) for plotting forecast tracks.

Usage:
    python download_hurricane_data.py
"""

import sys
import os
from pathlib import Path

from hurricane_helper.hurricane_downloader import HurricaneDownloader


def main():
    """Main function to download Hurricane Rafael data."""
    print("="*60)
    print("HURRICANE RAFAEL (2024) DATA DOWNLOAD")
    print("Google Weather Lab FNV3 Ensemble Model")
    print("="*60)
    
    try:
        # Initialize the hurricane downloader
        print("Initializing hurricane data downloader...")
        downloader = HurricaneDownloader()
        
        # Download the hurricane data
        print("\nStarting data download...")
        download_stats = downloader.download_hurricane_data()
        
        # Process the downloaded data
        print("\nProcessing downloaded data...")
        process_stats = downloader.process_downloaded_data()
        
        # Get and display summary
        print("\nGenerating download summary...")
        summary = downloader.get_download_summary()
        
        # Display results
        print("\n" + "="*60)
        print("DOWNLOAD COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Model: {summary['model']}")
        print(f"Hurricane: Rafael (2024)")
        print(f"Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        print(f"Raw Files Downloaded: {summary['raw_files_count']}")
        print(f"Processed Files: {summary['processed_files_count']}")
        print(f"Total Data Size: {summary['total_size_mb']} MB")
        print(f"Output Directory: {summary['output_directory']}")
        print(f"Processed Directory: {summary['processed_directory']}")
        print("\nDownload Statistics:")
        print(f"  - Successful: {download_stats['successful']}")
        print(f"  - Failed: {download_stats['failed']}")
        print(f"  - Skipped: {download_stats['skipped']}")
        print(f"  - Total: {download_stats['total']}")
        print("\nProcessing Statistics:")
        print(f"  - Processed: {process_stats['processed']}")
        print(f"  - Errors: {process_stats['errors']}")
        print(f"  - Total: {process_stats['total']}")
        print("="*60)
        
        if download_stats['failed'] > 0:
            print(f"\n⚠️  Warning: {download_stats['failed']} downloads failed.")
            print("Check the logs for details on failed downloads.")
        
        if process_stats['errors'] > 0:
            print(f"\n⚠️  Warning: {process_stats['errors']} files had processing errors.")
            print("Check the logs for details on processing errors.")
        
        print("\n✅ Data download and processing completed!")
        print("You can now use the downloaded data for plotting hurricane tracks and analysis.")
        
    except FileNotFoundError as e:
        print(f"❌ Configuration error: {e}")
        print("Please ensure the configuration file exists at: config/hurricane_config.yaml")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during download: {e}")
        print("Check the logs for more details.")
        sys.exit(1)


if __name__ == "__main__":
    main() 