"""
Hurricane Data Downloader for Google Weather Lab FNV3 Model

This module provides functionality to download hurricane forecast data from
Google Weather Lab's experimental FNV3 model, specifically for ensemble forecasts.
"""

import os
import sys
import yaml
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import time
from urllib.parse import urljoin
import re


class HurricaneDownloader:
    """
    Downloads hurricane forecast data from Google Weather Lab's FNV3 model.

    This class handles downloading ensemble forecast data for hurricanes,
    including all ensemble members, lead times, and relevant variables.
    """

    def __init__(self, config_path: str = "config/hurricane_config.yaml"):
        """
        Initialize the hurricane downloader with configuration.

        Args:
            config_path: Path to the configuration YAML file
        """
        # Get project root (3 levels up from this file: hurricane_helper/data_prep/src/)
        current_file = os.path.abspath(__file__)
        self.project_root = Path(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
            )
        )
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._setup_directories()

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            # Use absolute path from project root
            config_file = self.project_root / config_path
            with open(config_file, "r") as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")

    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get("logging", {})
        log_level = getattr(logging, log_config.get("level", "INFO"))

        # Create logs directory if it doesn't exist
        log_file = log_config.get("log_file", "logs/hurricane_download.log")
        log_file = self.project_root / log_file
        log_dir = log_file.parent
        if not log_dir.exists():
            os.makedirs(log_dir)

        # Configure logging
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                (
                    logging.StreamHandler(sys.stdout)
                    if log_config.get("console_output", True)
                    else logging.NullHandler()
                ),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def _setup_directories(self):
        """Create necessary directories for data storage."""
        download_config = self.config.get("download", {})
        output_dir = download_config.get("output_directory", "data/raw/weatherlab")

        # Use absolute paths from project root
        output_dir = self.project_root / output_dir
        if download_config.get("create_directories", True):
            os.makedirs(output_dir, exist_ok=True)
            self.logger.info(f"Created output directory: {output_dir}")

        # Also create processed data directory
        processing_config = self.config.get("processing", {})
        processed_dir = processing_config.get(
            "processed_output_directory", "data/preprocessed/weatherlab/processed"
        )
        processed_dir = self.project_root / processed_dir
        os.makedirs(processed_dir, exist_ok=True)

        self.output_dir = output_dir
        self.processed_dir = processed_dir

    def _generate_forecast_times(self) -> List[datetime]:
        """
        Generate list of forecast times to download data for.

        Returns:
            List of datetime objects for each forecast time in the range
        """
        hurricane_config = self.config.get("hurricane", {})
        start_date = datetime.strptime(hurricane_config["start_date"], "%Y-%m-%d")
        end_date = datetime.strptime(hurricane_config["end_date"], "%Y-%m-%d")

        # Get forecast times (default to 6-hour intervals if not specified)
        forecast_times = hurricane_config.get(
            "forecast_times", ["00:00", "06:00", "12:00", "18:00"]
        )

        forecast_datetimes = []
        current_date = start_date
        while current_date <= end_date:
            for time_str in forecast_times:
                # Parse time string (HH:MM format)
                hour, minute = map(int, time_str.split(":"))
                forecast_dt = current_date.replace(hour=hour, minute=minute)
                forecast_datetimes.append(forecast_dt)
            current_date += timedelta(days=1)

        self.logger.info(
            f"Generated {len(forecast_datetimes)} forecast times from {start_date} to {end_date}"
        )
        self.logger.info(f"Forecast intervals: {forecast_times}")
        return forecast_datetimes

    def _build_download_url(self, forecast_dt: datetime) -> str:
        """
        Build the download URL for a specific forecast time.

        Args:
            forecast_dt: Forecast datetime to build URL for

        Returns:
            Complete download URL
        """
        model_config = self.config.get("model", {})
        download_config = self.config.get("download", {})

        # Format date and time as required by the API
        date_str = forecast_dt.strftime("%Y_%m_%d")
        time_str = forecast_dt.strftime("%H_%M")

        # Build filename
        filename = f"{model_config['name']}_{date_str}T{time_str}_{model_config['track_type']}.{model_config['format']}"

        # Build full URL
        base_url = download_config["base_url"]
        url = f"{base_url}/{model_config['name']}/{model_config['type']}/{model_config['track_type']}/{model_config['format']}/{filename}"

        return url

    def _download_file(self, url: str, output_path: str) -> bool:
        """
        Download a single file from the given URL.

        Args:
            url: URL to download from
            output_path: Local path to save the file

        Returns:
            True if download successful, False otherwise
        """
        error_config = self.config.get("error_handling", {})
        max_retries = error_config.get("max_retries", 3)
        retry_delay = error_config.get("retry_delay_seconds", 5)

        for attempt in range(max_retries):
            try:
                self.logger.info(f"Downloading: {url}")
                response = requests.get(url, timeout=30)

                if response.status_code == 200:
                    # Create directory if it doesn't exist
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)

                    with open(output_path, "wb") as f:
                        f.write(response.content)

                    self.logger.info(f"Successfully downloaded: {output_path}")
                    return True
                else:
                    self.logger.warning(f"HTTP {response.status_code} for {url}")

            except requests.RequestException as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")

            if attempt < max_retries - 1:
                time.sleep(retry_delay)

        # If we get here, all attempts failed
        self.logger.error(f"Failed to download after {max_retries} attempts: {url}")
        return False

    def _save_failed_download(self, url: str, error: str):
        """Save failed download information to log file."""
        error_config = self.config.get("error_handling", {})
        failed_file = error_config.get(
            "failed_downloads_file", "logs/failed_downloads.txt"
        )
        failed_file = self.project_root / failed_file

        # Create logs directory if it doesn't exist
        log_dir = failed_file.parent
        if not log_dir.exists():
            os.makedirs(log_dir)

        with open(failed_file, "a") as f:
            f.write(f"{datetime.now().isoformat()},{url},{error}\n")

    def download_hurricane_data(self) -> Dict[str, int]:
        """
        Download hurricane data for all forecast times in the configured range.

        Returns:
            Dictionary with download statistics
        """
        forecast_times = self._generate_forecast_times()
        download_config = self.config.get("download", {})
        overwrite = download_config.get("overwrite_existing", False)

        successful_downloads = 0
        failed_downloads = 0
        skipped_downloads = 0

        self.logger.info(
            f"Starting download of hurricane data for {len(forecast_times)} forecast times"
        )

        for forecast_dt in forecast_times:
            url = self._build_download_url(forecast_dt)
            date_str = forecast_dt.strftime("%Y_%m_%d")
            time_str = forecast_dt.strftime("%H_%M")
            output_filename = f"FNV3_{date_str}_{time_str}_ensemble_data.csv"
            output_path = os.path.join(self.output_dir, output_filename)

            # Check if file already exists
            if os.path.exists(output_path) and not overwrite:
                self.logger.info(f"Skipping existing file: {output_path}")
                skipped_downloads += 1
                continue

            # Download the file
            if self._download_file(url, output_path):
                successful_downloads += 1
            else:
                failed_downloads += 1
                if self.config.get("error_handling", {}).get(
                    "save_failed_downloads", True
                ):
                    self._save_failed_download(url, "Download failed")

        stats = {
            "successful": successful_downloads,
            "failed": failed_downloads,
            "skipped": skipped_downloads,
            "total": len(forecast_times),
        }

        self.logger.info(f"Download completed. Statistics: {stats}")
        return stats

    def process_downloaded_data(self) -> Dict[str, int]:
        """
        Process and clean the downloaded hurricane data.

        Returns:
            Dictionary with processing statistics
        """
        processing_config = self.config.get("processing", {})
        if not processing_config.get("process_after_download", True):
            self.logger.info("Data processing disabled in configuration")
            return {"processed": 0, "errors": 0}

        self.logger.info("Starting data processing...")

        processed_files = 0
        error_files = 0

        # Get all downloaded CSV files
        csv_files = list(Path(self.output_dir).glob("*.csv"))

        for csv_file in csv_files:
            try:
                self.logger.info(f"Processing: {csv_file}")

                # Read the CSV file, skipping comment lines
                df = pd.read_csv(csv_file, comment="#")

                # Basic data cleaning and validation
                df_cleaned = self._clean_hurricane_data(df)

                # Save processed data
                output_filename = f"processed_{csv_file.name}"
                output_path = os.path.join(self.processed_dir, output_filename)
                df_cleaned.to_csv(output_path, index=False)

                processed_files += 1
                self.logger.info(f"Successfully processed: {output_path}")

            except Exception as e:
                error_files += 1
                self.logger.error(f"Error processing {csv_file}: {e}")

        stats = {
            "processed": processed_files,
            "errors": error_files,
            "total": len(csv_files),
        }

        self.logger.info(f"Data processing completed. Statistics: {stats}")
        return stats

    def _clean_hurricane_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate hurricane forecast data.

        Args:
            df: Raw hurricane data DataFrame

        Returns:
            Cleaned DataFrame
        """
        # Map column names to expected format
        column_mapping = {
            "lat": "latitude",
            "lon": "longitude",
            "maximum_sustained_wind_speed_knots": "wind_speed",
            "minimum_sea_level_pressure_hpa": "pressure",
            "init_time": "forecast_time",
            "valid_time": "valid_time",
            "lead_time": "lead_time",
            "sample": "ensemble_member",
            "track_id": "track_id",
        }

        # Rename columns
        df_cleaned = df.rename(columns=column_mapping)

        # Filter for Hurricane Rafael (AL182024) specifically
        hurricane_config = self.config.get("hurricane", {})
        target_track_id = hurricane_config.get("track_id", "AL182024")

        if "track_id" in df_cleaned.columns:
            original_count = len(df_cleaned)
            df_cleaned = df_cleaned[df_cleaned["track_id"] == target_track_id]
            filtered_count = len(df_cleaned)
            self.logger.info(
                f"Filtered for {target_track_id}: {original_count} -> {filtered_count} records"
            )

        # Convert lead_time from string to hours
        if "lead_time" in df_cleaned.columns:
            df_cleaned["lead_time"] = df_cleaned["lead_time"].apply(
                lambda x: pd.Timedelta(x).total_seconds() / 3600 if pd.notna(x) else 0
            )

        # Remove rows with missing critical data
        critical_columns = ["latitude", "longitude", "wind_speed", "pressure"]
        df_cleaned = df_cleaned.dropna(subset=critical_columns)

        # Convert data types
        numeric_columns = ["latitude", "longitude", "wind_speed", "pressure"]
        for col in numeric_columns:
            if col in df_cleaned.columns:
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors="coerce")

        # Remove invalid coordinates
        df_cleaned = df_cleaned[
            (df_cleaned["latitude"] >= -90)
            & (df_cleaned["latitude"] <= 90)
            & (df_cleaned["longitude"] >= -180)
            & (df_cleaned["longitude"] <= 180)
        ]

        # Remove invalid wind speeds and pressures
        df_cleaned = df_cleaned[
            (df_cleaned["wind_speed"] >= 0) & (df_cleaned["pressure"] > 0)
        ]

        # Sort by ensemble member, forecast time, and lead time
        sort_columns = ["ensemble_member", "forecast_time", "lead_time"]
        available_sort_columns = [
            col for col in sort_columns if col in df_cleaned.columns
        ]
        if available_sort_columns:
            df_cleaned = df_cleaned.sort_values(available_sort_columns)

        return df_cleaned

    def get_download_summary(self) -> Dict:
        """
        Get a summary of downloaded data.

        Returns:
            Dictionary with download summary information
        """
        csv_files = list(Path(self.output_dir).glob("*.csv"))
        processed_files = list(Path(self.processed_dir).glob("*.csv"))

        total_size = sum(f.stat().st_size for f in csv_files if f.exists())

        summary = {
            "raw_files_count": len(csv_files),
            "processed_files_count": len(processed_files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "date_range": {
                "start": self.config["hurricane"]["start_date"],
                "end": self.config["hurricane"]["end_date"],
            },
            "model": self.config["model"]["name"],
            "output_directory": self.output_dir,
            "processed_directory": self.processed_dir,
        }

        return summary


def main():
    """Main function to run the hurricane data downloader."""
    try:
        # Initialize downloader
        downloader = HurricaneDownloader()

        # Download data
        download_stats = downloader.download_hurricane_data()

        # Process data
        process_stats = downloader.process_downloaded_data()

        # Get summary
        summary = downloader.get_download_summary()

        print("\n" + "=" * 50)
        print("HURRICANE DATA DOWNLOAD SUMMARY")
        print("=" * 50)
        print(f"Model: {summary['model']}")
        print(
            f"Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}"
        )
        print(f"Raw Files: {summary['raw_files_count']}")
        print(f"Processed Files: {summary['processed_files_count']}")
        print(f"Total Size: {summary['total_size_mb']} MB")
        print(f"Download Stats: {download_stats}")
        print(f"Processing Stats: {process_stats}")
        print("=" * 50)

    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise


if __name__ == "__main__":
    main()
