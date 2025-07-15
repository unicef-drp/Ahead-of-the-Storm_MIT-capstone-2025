"""
Main script to download and process ECMWF rainfall ensemble data for Nicaragua.

This script loads configuration, downloads the real-time ECMWF ensemble precipitation forecast,
saves the raw data, subsets it to the Nicaragua region, and saves the processed data as NetCDF.
"""

from src.data_prep.rainfall_helper.rainfall_downloader import main as rainfall_download_main

if __name__ == "__main__":
    rainfall_download_main() 