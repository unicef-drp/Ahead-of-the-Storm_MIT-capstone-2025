"""
Population data downloader for census and demographic data.

This module provides functionality to download WorldPop spatial population data
including age/sex structured data.
"""

import requests
import numpy as np
import json
from typing import Dict, Optional, Any, List
import time

from src.utils.config_utils import load_config, get_config_value
from src.utils.logging_utils import setup_logging, get_logger
from src.utils.path_utils import ensure_directory, get_data_path


class PopulationDownloader:
    """Downloader for WorldPop spatial population data including age/sex structures."""

    def __init__(self, config_path: str = "config/census_config.yaml"):
        """Initialize the population downloader."""
        self.config = load_config(config_path)
        self.logger = setup_logging(__name__)
        self.session = requests.Session()
        self._setup_directories()

    def _setup_directories(self):
        """Create output directories if they don't exist."""
        raw_dir = get_config_value(
            self.config, "output.raw_data_dir", "data/raw/census"
        )
        self.raw_dir = get_data_path(raw_dir)
        ensure_directory(self.raw_dir)
        self.logger.info(f"Raw data directory: {self.raw_dir}")

    def _download_file(
        self, url: str, filename: str, timeout: int = 300
    ) -> Optional[str]:
        """Download a file from URL with retry logic."""
        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                self.logger.info(f"Downloading {filename} (attempt {attempt + 1})")
                response = self.session.get(url, timeout=timeout)
                response.raise_for_status()

                filepath = self.raw_dir / filename
                with open(filepath, "wb") as f:
                    f.write(response.content)

                self.logger.info(f"Successfully downloaded {filename}")
                return str(filepath)

            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2**attempt))
                else:
                    self.logger.error(
                        f"All {max_retries} download attempts failed for {filename}"
                    )
                    return None

    def _generate_age_sex_urls(self) -> List[Dict[str, str]]:
        """Generate URLs for all age/sex combinations."""
        base_url = "https://data.worldpop.org/GIS/AgeSex_structures/Global_2000_2020_Constrained_UNadj/2020/NIC"
        country_code = "nic"
        year = "2020"

        # Age groups: 0, 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80
        age_groups = [
            0,
            1,
            5,
            10,
            15,
            20,
            25,
            30,
            35,
            40,
            45,
            50,
            55,
            60,
            65,
            70,
            75,
            80,
        ]
        genders = ["f", "m"]  # female, male

        urls = []

        for gender in genders:
            for age in age_groups:
                filename = f"{country_code}_{gender}_{age}_{year}_constrained_UNadj.tif"
                url = f"{base_url}/{filename}"
                urls.append(
                    {
                        "url": url,
                        "filename": filename,
                        "gender": gender,
                        "age": age,
                        "year": year,
                    }
                )

        return urls

    def download_worldpop_data(self) -> Optional[str]:
        """Download WorldPop total population data for Nicaragua."""
        if not self.config["data_sources"]["worldpop"]["enabled"]:
            self.logger.info("WorldPop data source is disabled")
            return None

        direct_url = self.config["data_sources"]["worldpop"]["direct_url"]
        filename = "nicaragua_population_2020_constrained.tif"

        self.logger.info(
            f"Downloading WorldPop total population data from: {direct_url}"
        )
        result = self._download_file(direct_url, filename)

        if result:
            self.logger.info(
                f"Successfully downloaded WorldPop total population data: {result}"
            )
            return result
        else:
            self.logger.warning("Failed to download WorldPop total population data")
            return None

    def download_age_sex_data(self, test_mode: bool = False) -> Dict[str, str]:
        """Download all age/sex structured population data."""
        if not self.config["data_sources"]["worldpop"]["enabled"]:
            self.logger.info("WorldPop data source is disabled")
            return {}

        self.logger.info("Starting download of age/sex structured population data")

        urls = self._generate_age_sex_urls()

        # In test mode, only download first few files
        if test_mode:
            urls = urls[:4]  # Download first 4 files for testing
            self.logger.info("TEST MODE: Downloading only first 4 age/sex files")

        results = {}
        successful_downloads = 0
        total_downloads = len(urls)

        for i, url_info in enumerate(urls, 1):
            self.logger.info(
                f"Downloading {i}/{total_downloads}: {url_info['filename']}"
            )

            result = self._download_file(url_info["url"], url_info["filename"])
            if result:
                results[url_info["filename"]] = result
                successful_downloads += 1

                # Add a small delay between downloads to be respectful to the server
                time.sleep(0.5)
            else:
                self.logger.warning(f"Failed to download {url_info['filename']}")

        self.logger.info(
            f"Age/sex data download completed: {successful_downloads}/{total_downloads} successful"
        )
        return results

    def get_population_statistics(self, filepath: str) -> Dict[str, Any]:
        """Get basic population statistics from WorldPop GeoTIFF."""
        try:
            import rasterio

            self.logger.info("Extracting population statistics from GeoTIFF")

            with rasterio.open(filepath) as src:
                data = src.read(1)  # Read the first band

                # Calculate statistics
                total_population = np.sum(data[data > 0])
                max_population = np.max(data)
                min_population = np.min(data[data > 0])
                mean_population = np.mean(data[data > 0])
                populated_cells = np.count_nonzero(data)
                total_cells = data.size

                stats = {
                    "total_population": float(total_population),
                    "max_population_per_cell": float(max_population),
                    "min_population_per_cell": float(min_population),
                    "mean_population_per_cell": float(mean_population),
                    "populated_cells": int(populated_cells),
                    "total_cells": int(total_cells),
                    "spatial_resolution": "100m",
                    "crs": str(src.crs),
                    "transform": str(src.transform),
                }

                self.logger.info(
                    f"Population statistics: Total population = {total_population:,.0f}"
                )
                return stats

        except ImportError:
            self.logger.error(
                "rasterio not available. Install with: pip install rasterio"
            )
            return {}
        except Exception as e:
            self.logger.error(f"Error extracting population statistics: {e}")
            return {}

    def download_all(self, test_mode: bool = False) -> Dict[str, str]:
        """Download all WorldPop population data including age/sex structures."""
        results = {}

        try:
            # Download total population data
            worldpop_file = self.download_worldpop_data()
            if worldpop_file:
                results["worldpop_total"] = worldpop_file

                # Get basic statistics for verification
                stats = self.get_population_statistics(worldpop_file)
                if stats:
                    self.logger.info(
                        f"Total population data verified: {stats['total_population']:,.0f} total population"
                    )

            # Download age/sex structured data
            age_sex_files = self.download_age_sex_data(test_mode=test_mode)
            results.update(age_sex_files)

        except Exception as e:
            self.logger.error(f"Error during population data download: {e}")

        return results
