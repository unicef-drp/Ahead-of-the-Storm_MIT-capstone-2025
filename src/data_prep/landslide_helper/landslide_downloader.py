"""
Landslide hazard data downloader for NASA LHASA-F.

This module provides functionality to download landslide hazard prediction data
from NASA's Landslide Hazard Assessment for Situational Awareness - Future (LHASA-F)
service and convert it to georeferenced GeoTIFF format for analysis.
"""

import requests
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Any, Tuple
import time
import os
from pathlib import Path

from src.utils.config_utils import load_config, get_config_value
from src.utils.logging_utils import setup_logging, get_logger
from src.utils.path_utils import ensure_directory, get_data_path, create_output_filename


class LandslideDownloader:
    """Downloader for NASA LHASA-F landslide hazard prediction data."""

    def __init__(self, config_path: str = "config/landslide_config.yaml"):
        """Initialize the landslide downloader."""
        self.config = load_config(config_path)
        self.logger = setup_logging(__name__)
        self.session = requests.Session()
        self._setup_directories()
        self._calculate_web_mercator_bbox()

    def _setup_directories(self):
        """Create output directories if they don't exist."""
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

    def _calculate_web_mercator_bbox(self):
        """Calculate Web Mercator bounding box from geographic coordinates."""
        try:
            import pyproj
            
            # Get geographic coordinates from config
            lat_min = get_config_value(self.config, "nicaragua_bbox.lat_min")
            lat_max = get_config_value(self.config, "nicaragua_bbox.lat_max")
            lon_min = get_config_value(self.config, "nicaragua_bbox.lon_min")
            lon_max = get_config_value(self.config, "nicaragua_bbox.lon_max")
            
            # Create coordinate transformers
            wgs84_to_web_mercator = pyproj.Transformer.from_crs(
                "EPSG:4326", "EPSG:3857", always_xy=True
            )
            
            # Transform coordinates
            x_min, y_min = wgs84_to_web_mercator.transform(lon_min, lat_min)
            x_max, y_max = wgs84_to_web_mercator.transform(lon_max, lat_max)
            
            # Store in config for later use
            self.web_mercator_bbox = {
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max
            }
            
            self.logger.info(f"Web Mercator bbox: {self.web_mercator_bbox}")
            
        except ImportError:
            self.logger.error("pyproj not available. Install with: pip install pyproj")
            # Fallback to approximate values for Nicaragua
            self.web_mercator_bbox = {
                "x_min": -9760000,
                "y_min": 1180000,
                "x_max": -9200000,
                "y_max": 1700000
            }
            self.logger.warning("Using approximate Web Mercator bbox for Nicaragua")

    def _build_export_url(self) -> str:
        """Build the NASA LHASA-F export URL with parameters."""
        base_url = get_config_value(self.config, "nasa_service.base_url")
        service_name = get_config_value(self.config, "nasa_service.service_name")
        layer_name = get_config_value(self.config, "nasa_service.layer_name")
        # Use the MapServer root export endpoint
        url = f"{base_url}/{service_name}/{layer_name}/MapServer/export"
        self.logger.debug(f"Base export URL: {url}")
        return url

    def _build_export_params(self) -> Dict[str, Any]:
        """Build parameters for the export request."""
        # Get bbox
        bbox = self.web_mercator_bbox
        bbox_str = f"{bbox['x_min']},{bbox['y_min']},{bbox['x_max']},{bbox['y_max']}"
        
        # Get other parameters from config
        image_size = get_config_value(self.config, "export.image_size")
        spatial_ref = get_config_value(self.config, "export.spatial_reference")
        format_type = get_config_value(self.config, "nasa_service.preferred_format")
        dpi = get_config_value(self.config, "export.dpi")
        transparent = get_config_value(self.config, "export.transparent")
        layer_id = get_config_value(self.config, "nasa_service.layer_id")
        
        params = {
            "f": "image",
            "bbox": bbox_str,
            "bboxSR": spatial_ref,
            "imageSR": spatial_ref,
            "size": f"{image_size[0]},{image_size[1]}",
            "format": format_type,
            "dpi": dpi,
            "transparent": str(transparent).lower(),
            "layers": f"show:{layer_id}"
        }
        
        self.logger.debug(f"Export parameters: {params}")
        return params

    def _download_tiff(self, url: str, params: Dict[str, Any]) -> Optional[bytes]:
        """Download TIFF data from NASA LHASA-F service."""
        max_retries = get_config_value(self.config, "error_handling.max_retries", 3)
        retry_delay = get_config_value(self.config, "error_handling.retry_delay_seconds", 5)
        timeout = get_config_value(self.config, "error_handling.timeout_seconds", 300)
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Downloading landslide data (attempt {attempt + 1})")
                
                response = self.session.get(url, params=params, timeout=timeout)
                response.raise_for_status()
                
                # Check if we got an image or an error page
                content_type = response.headers.get('content-type', '')
                if 'text/html' in content_type:
                    self.logger.error(f"Received HTML instead of image: {response.text[:200]}")
                    return None
                
                self.logger.info(f"Successfully downloaded {len(response.content)} bytes")
                return response.content
                
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2**attempt))
                else:
                    self.logger.error(f"All {max_retries} download attempts failed")
                    return None

    def _save_raw_tiff(self, tiff_data: bytes) -> str:
        """Save raw TIFF data to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"nasa_lhasa_raw_{timestamp}.tif"
        filepath = self.raw_dir / filename
        
        with open(filepath, "wb") as f:
            f.write(tiff_data)
        
        self.logger.info(f"Saved raw TIFF to: {filepath}")
        return str(filepath)

    def _check_georeferencing(self, tiff_path: str) -> bool:
        """Check if TIFF has proper georeferencing."""
        try:
            import rasterio
            
            with rasterio.open(tiff_path) as src:
                # Check if CRS is defined
                if src.crs is None:
                    self.logger.warning("TIFF has no CRS information")
                    return False
                
                # Check if transform is defined
                if src.transform is None:
                    self.logger.warning("TIFF has no transform information")
                    return False
                
                self.logger.info(f"TIFF is georeferenced with CRS: {src.crs}")
                return True
                
        except ImportError:
            self.logger.error("rasterio not available. Install with: pip install rasterio")
            return False
        except Exception as e:
            self.logger.error(f"Error checking georeferencing: {e}")
            return False

    def _georeference_tiff(self, tiff_path: str) -> str:
        """Add georeferencing to TIFF if missing."""
        try:
            import rasterio
            from rasterio.transform import from_bounds
            
            # Get bbox in Web Mercator
            bbox = self.web_mercator_bbox
            
            # Read the TIFF
            with rasterio.open(tiff_path) as src:
                data = src.read()
                height, width = data.shape[1], data.shape[2]
            
            # Calculate transform from bounds
            transform = from_bounds(
                bbox['x_min'], bbox['y_min'], bbox['x_max'], bbox['y_max'],
                width, height
            )
            
            # Create georeferenced output path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"nasa_lhasa_georeferenced_{timestamp}.tif"
            output_path = self.raw_dir / output_filename
            
            # Write georeferenced TIFF
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=data.shape[0],
                dtype=data.dtype,
                crs='EPSG:3857',
                transform=transform
            ) as dst:
                dst.write(data)
            
            self.logger.info(f"Georeferenced TIFF saved to: {output_path}")
            return str(output_path)
            
        except ImportError:
            self.logger.error("rasterio not available. Install with: pip install rasterio")
            return tiff_path
        except Exception as e:
            self.logger.error(f"Error georeferencing TIFF: {e}")
            return tiff_path

    def _reproject_to_wgs84(self, tiff_path: str) -> str:
        """Reproject TIFF to WGS84 (EPSG:4326)."""
        try:
            import rasterio
            from rasterio.warp import calculate_default_transform, reproject, Resampling
            
            # Read source
            with rasterio.open(tiff_path) as src:
                # Calculate transform for WGS84
                dst_crs = 'EPSG:4326'
                transform, width, height = calculate_default_transform(
                    src.crs, dst_crs, src.width, src.height, *src.bounds
                )
                
                # Create output path
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"nasa_lhasa_wgs84_{timestamp}.tif"
                output_path = self.processed_dir / output_filename
                
                # Reproject
                with rasterio.open(
                    output_path,
                    'w',
                    driver='GTiff',
                    height=height,
                    width=width,
                    count=src.count,
                    dtype=src.dtypes[0],
                    crs=dst_crs,
                    transform=transform
                ) as dst:
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=rasterio.band(dst, 1),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest
                    )
            
            self.logger.info(f"Reprojected TIFF saved to: {output_path}")
            return str(output_path)
            
        except ImportError:
            self.logger.error("rasterio not available. Install with: pip install rasterio")
            return tiff_path
        except Exception as e:
            self.logger.error(f"Error reprojecting TIFF: {e}")
            return tiff_path

    def download_landslide_data(self) -> Optional[str]:
        """Download landslide hazard data from NASA LHASA-F."""
        try:
            # Build URL and parameters
            url = self._build_export_url()
            params = self._build_export_params()
            
            # Download TIFF data
            tiff_data = self._download_tiff(url, params)
            if tiff_data is None:
                self.logger.error("Failed to download TIFF data")
                return None
            
            # Save raw TIFF
            raw_tiff_path = self._save_raw_tiff(tiff_data)
            
            # Check georeferencing
            if not self._check_georeferencing(raw_tiff_path):
                self.logger.info("Adding georeferencing to TIFF")
                raw_tiff_path = self._georeference_tiff(raw_tiff_path)
            
            # Process if enabled
            if get_config_value(self.config, "processing.process_after_download", True):
                self.logger.info("Processing downloaded data")
                
                # Reproject to WGS84 if enabled
                if get_config_value(self.config, "processing.reproject_to_wgs84", True):
                    final_path = self._reproject_to_wgs84(raw_tiff_path)
                else:
                    final_path = raw_tiff_path
                
                self.logger.info(f"Landslide data processing completed: {final_path}")
                return final_path
            else:
                self.logger.info(f"Landslide data download completed: {raw_tiff_path}")
                return raw_tiff_path
                
        except Exception as e:
            self.logger.error(f"Error during landslide data download: {e}")
            return None

    def get_download_summary(self) -> Dict[str, Any]:
        """Get a summary of downloaded landslide data."""
        raw_files = list(self.raw_dir.glob("*.tif"))
        processed_files = list(self.processed_dir.glob("*.tif"))
        
        total_size = sum(f.stat().st_size for f in raw_files + processed_files if f.exists())
        
        summary = {
            "raw_files_count": len(raw_files),
            "processed_files_count": len(processed_files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "data_source": "NASA LHASA-F",
            "coverage": "Nicaragua",
            "raw_directory": str(self.raw_dir),
            "processed_directory": str(self.processed_dir),
        }
        
        return summary 