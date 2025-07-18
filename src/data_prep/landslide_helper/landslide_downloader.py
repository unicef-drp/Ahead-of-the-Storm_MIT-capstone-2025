"""
Landslide hazard data downloader for NASA LHASA-F.

This module provides functionality to download landslide hazard prediction data
from NASA's Landslide Hazard Assessment for Situational Awareness - Future (LHASA-F)
service and convert it to georeferenced GeoTIFF format for analysis.
"""

import requests
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Any
import time
import os
from pathlib import Path
import re
from bs4 import BeautifulSoup

from src.utils.config_utils import load_config, get_config_value
from src.utils.logging_utils import setup_logging, get_logger
from src.utils.path_utils import ensure_directory, get_data_path

class LandslideDownloader:
    """Downloader for NASA LHASA-F landslide hazard prediction data (direct download)."""

    def __init__(self, config_path: str = "config/landslide_config.yaml"):
        self.config = load_config(config_path)
        self.logger = setup_logging(__name__)
        self.session = requests.Session()
        self._setup_directories()

    def _setup_directories(self):
        raw_dir = get_config_value(
            self.config, "output.raw_data_dir", "data/raw/landslide"
        )
        processed_dir = get_config_value(
            self.config, "output.processed_data_dir", "data/preprocessed/landslide"
        )
        self.raw_dir = get_data_path(raw_dir)
        self.processed_dir = get_data_path(processed_dir)
        ensure_directory(self.raw_dir)
        ensure_directory(self.processed_dir)
        self.logger.info(f"Raw data directory: {self.raw_dir}")
        self.logger.info(f"Processed data directory: {self.processed_dir}")

    def _list_available_files(self) -> list:
        """List available GeoTIFF files in the NASA directory."""
        source_url = get_config_value(self.config, "source_url")
        resp = self.session.get(source_url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        files = []
        for link in soup.find_all('a'):
            href = link.get('href', '')
            if href.endswith('.tif'):
                files.append(href)
        self.logger.info(f"Found {len(files)} GeoTIFF files in NASA directory.")
        return files

    def _parse_forecast_window(self, filename: str) -> Optional[int]:
        """Parse forecast window in hours from filename."""
        # Example: 20250716T0600+20250718T0600.tif
        match = re.match(r"(\d{8}T\d{4})\+(\d{8}T\d{4})\.tif", filename)
        if not match:
            return None
        start, end = match.groups()
        try:
            dt_start = datetime.strptime(start, "%Y%m%dT%H%M")
            dt_end = datetime.strptime(end, "%Y%m%dT%H%M")
            delta = dt_end - dt_start
            return int(delta.total_seconds() // 3600)
        except Exception:
            return None

    def _find_latest_files(self, files: list, windows: list) -> Dict[int, str]:
        """Find the latest file for each requested forecast window (in hours)."""
        latest = {}
        for w in windows:
            candidates = [f for f in files if self._parse_forecast_window(f) == w]
            if not candidates:
                self.logger.warning(f"No files found for {w}h forecast window.")
                continue
            # Sort by end time in filename
            candidates.sort(key=lambda f: f.split('+')[1])
            latest[w] = candidates[-1]
        return latest

    def _download_file(self, url: str, out_path: Path) -> bool:
        """Download a file from URL to out_path."""
        self.logger.info(f"Downloading {url} ...")
        resp = self.session.get(url, stream=True)
        if resp.status_code == 200:
            with open(out_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            self.logger.info(f"Saved to {out_path}")
            return True
        else:
            self.logger.error(f"Failed to download {url}: {resp.status_code}")
            return False

    def download_landslide_data(self) -> Optional[list]:
        """Download the latest landslide hazard GeoTIFF(s) for the configured forecast window(s), clip to Nicaragua. Avoid redownloading if file exists."""
        try:
            source_url = get_config_value(self.config, "source_url")
            forecast_windows = get_config_value(self.config, "forecast_windows", [48])
            files = self._list_available_files()
            latest_files = self._find_latest_files(files, forecast_windows)
            downloaded = []
            for w, fname in latest_files.items():
                # Parse forecast window and start time from NASA filename
                match = re.match(r"(\d{8}T\d{4})\+(\d{8}T\d{4})", fname)
                if match:
                    start, _ = match.groups()
                    raw_name = f"landslide_forecast_48h_{start}.tif"
                else:
                    raw_name = f"landslide_forecast_48h_unknown.tif"
                raw_file = self.raw_dir / raw_name
                if not raw_file.exists():
                    file_url = source_url + fname
                    if self._download_file(file_url, raw_file):
                        self.logger.info(f"Downloaded new file for {w}h: {raw_file}")
                    else:
                        self.logger.error(f"Failed to download file for {w}h: {file_url}")
                        continue
                else:
                    self.logger.info(f"Found existing raw file for {w}h: {raw_file}, skipping download.")
                # Clip to Nicaragua
                # Use the same start time for processed file
                match = re.match(r"landslide_forecast_48h_(\d{8}T\d{4})\.tif", raw_file.name)
                if match:
                    start = match.group(1)
                    processed_name = f"landslide_forecast_48h_{start}_nicaragua.tif"
                else:
                    processed_name = f"landslide_forecast_48h_unknown_nicaragua.tif"
                clipped_path = self._clip_to_nicaragua(str(raw_file), processed_name)
                downloaded.append(clipped_path)
            if downloaded:
                self.logger.info(f"Downloaded and processed files: {downloaded}")
                return downloaded
            else:
                self.logger.error("No files downloaded or processed.")
                return None
        except Exception as e:
            self.logger.error(f"Error during landslide data download: {e}")
            return None

    def get_download_summary(self) -> Dict[str, Any]:
        raw_files = list(self.raw_dir.glob("*.tif"))
        processed_files = list(self.processed_dir.glob("*.tif"))
        total_size = sum(f.stat().st_size for f in raw_files + processed_files if f.exists())
        summary = {
            "raw_files_count": len(raw_files),
            "processed_files_count": len(processed_files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "data_source": "NASA LHASA-F (direct)",
            "coverage": "Global",
            "raw_directory": str(self.raw_dir),
            "processed_directory": str(self.processed_dir),
        }
        return summary

    def _clip_to_nicaragua(self, tiff_path: str, output_filename: str) -> str:
        """Clip the raster to the Nicaragua boundary and save with provided filename."""
        try:
            import rasterio
            from rasterio.mask import mask
            from src.utils.hurricane_geom import get_nicaragua_boundary
            # Load Nicaragua boundary as a GeoDataFrame
            nicaragua_gdf = get_nicaragua_boundary()
            with rasterio.open(tiff_path) as src:
                # Ensure CRS matches raster
                if nicaragua_gdf.crs != src.crs:
                    nicaragua_gdf = nicaragua_gdf.to_crs(src.crs)
                # Merge all geometries, fix invalid polygons
                nicaragua_geom = [nicaragua_gdf.unary_union.buffer(0).__geo_interface__]
                out_image, out_transform = mask(src, nicaragua_geom, crop=True, nodata=src.nodata)
                out_meta = src.meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform
                })
                output_path = self.processed_dir / output_filename
                with rasterio.open(output_path, 'w', **out_meta) as dst:
                    dst.write(out_image)
            self.logger.info(f"Clipped TIFF to Nicaragua boundary: {output_path}")
            return str(output_path)
        except Exception as e:
            self.logger.error(f"Error clipping TIFF to Nicaragua boundary: {e}")
            return tiff_path 