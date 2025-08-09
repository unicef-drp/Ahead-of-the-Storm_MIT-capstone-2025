"""
Storm surge data processor for bathymetry and hurricane ensemble data.

This module provides functionality to load and process bathymetry data
and integrate it with hurricane ensemble data for storm surge modeling.
"""

import os
import numpy as np
import pandas as pd
import rasterio
from scipy.interpolate import griddata
import warnings
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
import pyproj

from src.utils.config_utils import load_config, get_config_value
from src.utils.path_utils import get_data_path, ensure_directory

warnings.filterwarnings("ignore")


class SurgeProcessor:
    """Processor for bathymetry data and storm surge calculations."""

    def __init__(self, config_path: str = "config/surge_config.yaml"):
        """Initialize the surge processor."""
        self.config = load_config(config_path)
        self.bathymetry = None
        self.bathymetry_crs = None
        self.bathymetry_transform = None
        self.bathymetry_bounds = None

        # Computational grid
        self.grid = None
        self.grid_lats = None
        self.grid_lons = None
        self.grid_depths = None
        self.dx = None
        self.dy = None

        self._setup_directories()

    def _setup_directories(self):
        """Create output directories if they don't exist."""
        output_dir = get_config_value(
            self.config,
            "surge.output_directory",
            "data/results/impact_analysis/surge_ensemble",
        )
        self.output_dir = get_data_path(output_dir)
        ensure_directory(self.output_dir)

        cache_dir = get_config_value(
            self.config, "surge.cache_dir", "data/results/impact_analysis/cache"
        )
        self.cache_dir = get_data_path(cache_dir)
        ensure_directory(self.cache_dir)

    def find_bathymetry_file(self) -> Optional[str]:
        """Find bathymetry file in the bathymetry directory."""
        bathymetry_dir = get_config_value(
            self.config, "surge.bathymetry_dir", "data/raw/bathymetry"
        )
        bathy_path = get_data_path(bathymetry_dir)

        if not bathy_path.exists():
            print(f"Bathymetry directory not found: {bathy_path}")
            return None

        # Look for common bathymetry file formats
        extensions = [".tif", ".tiff", ".nc", ".h5", ".hdf5"]
        for ext in extensions:
            files = list(bathy_path.glob(f"*{ext}"))
            if files:
                return str(files[0])

        print(f"No bathymetry files found in {bathy_path}")
        return None

    def load_bathymetry_data(self, bathymetry_file: Optional[str] = None) -> bool:
        """Load bathymetry data and check coverage."""
        if bathymetry_file is None:
            bathymetry_file = self.find_bathymetry_file()

        if not bathymetry_file:
            print("No bathymetry file provided or found")
            return False

        print(f"Loading bathymetry data: {bathymetry_file}")

        try:
            with rasterio.open(bathymetry_file) as src:
                print(f"  Bathymetry shape: {src.shape}")
                print(f"  Bathymetry CRS: {src.crs}")
                print(f"  Bathymetry bounds: {src.bounds}")

                # Store metadata
                self.bathymetry_crs = src.crs
                self.bathymetry_transform = src.transform
                self.bathymetry_bounds = src.bounds
                
                # If CRS is None or bounds are in pixel coordinates, fix them
                if self.bathymetry_crs is None or (self.bathymetry_bounds.left == 0 and self.bathymetry_bounds.right == src.width):
                    print("  Warning: Bathymetry file appears to be in pixel coordinates")
                    print("  Setting Nicaragua geographic bounds manually")
                    # Nicaragua bounds: approximately -87.7°W to -82.7°W, 10.7°N to 15.1°N
                    self.bathymetry_bounds = rasterio.coords.BoundingBox(
                        left=-87.7, bottom=10.7, right=-82.7, top=15.1
                    )
                    self.bathymetry_crs = rasterio.crs.CRS.from_epsg(4326)

                # Read bathymetry data (downsample if very large)
                if src.width > 3000 or src.height > 3000:
                    downsample = 2
                    print(f"  Downsampling by factor {downsample}")
                    self.bathymetry = src.read(1)[::downsample, ::downsample]

                    # Adjust transform
                    self.bathymetry_transform = rasterio.Affine(
                        src.transform[0] * downsample,
                        src.transform[1],
                        src.transform[2],
                        src.transform[3],
                        src.transform[4] * downsample,
                        src.transform[5],
                    )
                else:
                    self.bathymetry = src.read(1)

                # Clean nodata values
                if src.nodata is not None:
                    self.bathymetry = np.where(
                        self.bathymetry == src.nodata, np.nan, self.bathymetry
                    )

                # Handle common GEBCO nodata values
                for nodata_val in [-32768, -32767, 32767, -9999, 9999]:
                    self.bathymetry = np.where(
                        self.bathymetry == nodata_val, np.nan, self.bathymetry
                    )

                # Print statistics
                valid_bathy = self.bathymetry[~np.isnan(self.bathymetry)]
                if len(valid_bathy) > 0:
                    water_pct = np.sum(valid_bathy < 0) / len(valid_bathy) * 100
                    print(
                        f"  Bathymetry range: {np.min(valid_bathy):.1f} to {np.max(valid_bathy):.1f} m"
                    )
                    print(f"  Water coverage: {water_pct:.1f}%")
                else:
                    print("Error: No valid bathymetry data found!")
                    return False

                return True

        except Exception as e:
            print(f"Error loading bathymetry: {e}")
            return False

    def determine_modeling_domain(self) -> Dict[str, float]:
        """Determine modeling domain using full bathymetry extent."""
        print("Determining modeling domain...")

        if self.bathymetry_bounds is None:
            raise ValueError("Load bathymetry before determining domain")

        # Always use full bathymetry bounds
        domain = {
            "west": self.bathymetry_bounds.left,
            "east": self.bathymetry_bounds.right,
            "south": self.bathymetry_bounds.bottom,
            "north": self.bathymetry_bounds.top,
        }

        # Calculate domain statistics
        domain_width = abs(domain["east"] - domain["west"])
        domain_height = abs(domain["north"] - domain["south"])
        domain_area = domain_width * domain_height * (111**2)  # km²

        print(f"Using full bathymetry domain:")
        print(
            f"  Bounds: {domain['west']:.2f}°W to {domain['east']:.2f}°W, {domain['south']:.2f}°N to {domain['north']:.2f}°N"
        )

        return domain

    def setup_computational_grid(self, domain: Dict[str, float]) -> bool:
        """Setup computational grid with proper coordinate handling."""
        print("Setting up computational grid...")

        grid_resolution_km = get_config_value(
            self.config, "surge.grid_resolution_km", 3.0
        )

        # Calculate grid dimensions
        deg_per_km_lat = 1.0 / 111.0
        deg_per_km_lon = 1.0 / (
            111.0 * np.cos(np.radians((domain["north"] + domain["south"]) / 2))
        )

        dlat = grid_resolution_km * deg_per_km_lat
        dlon = grid_resolution_km * deg_per_km_lon

        # Create coordinate arrays
        lats = np.arange(domain["south"], domain["north"] + dlat / 2, dlat)
        lons = np.arange(domain["west"], domain["east"] + dlon / 2, dlon)

        self.grid_lons, self.grid_lats = np.meshgrid(lons, lats)

        # Calculate actual grid spacing in meters
        self.dy = grid_resolution_km * 1000
        mean_lat = np.mean(self.grid_lats)
        self.dx = self.dy * np.cos(np.radians(mean_lat))

        print(f"  Grid shape: {self.grid_lats.shape}")
        print(f"  Grid resolution: {grid_resolution_km} km")
        print(f"  Grid spacing: {self.dx:.0f} x {self.dy:.0f} m")

        return True

    def interpolate_bathymetry_to_grid(self) -> bool:
        """Interpolate bathymetry to computational grid using proper coordinate transformation."""
        print("Interpolating bathymetry to computational grid...")

        # Get grid coordinates
        nrows, ncols = self.grid_lats.shape

        # Initialize depth array
        self.grid_depths = np.zeros((nrows, ncols), dtype=np.float32)

        # Convert bathymetry to lat/lon coordinates if needed
        bathy_height, bathy_width = self.bathymetry.shape

        # Create coordinate arrays for bathymetry
        # Handle case where CRS is None (assume it's already in lat/lon)
        if self.bathymetry_crs is None:
            print("  Warning: No CRS found in bathymetry file, assuming EPSG:4326")
            self.bathymetry_crs = rasterio.crs.CRS.from_epsg(4326)
            
        if self.bathymetry_crs.to_string() == "EPSG:4326":
            # Bathymetry is already in lat/lon
            bathy_lons_1d = np.linspace(
                self.bathymetry_bounds.left, self.bathymetry_bounds.right, bathy_width
            )
            bathy_lats_1d = np.linspace(
                self.bathymetry_bounds.top, self.bathymetry_bounds.bottom, bathy_height
            )

            # Create 2D coordinate grids
            bathy_lons_2d, bathy_lats_2d = np.meshgrid(bathy_lons_1d, bathy_lats_1d)
        else:
            # Need to transform bathymetry coordinates
            transformer = pyproj.Transformer.from_crs(
                self.bathymetry_crs, "EPSG:4326", always_xy=True
            )

            # Create coordinate grid for bathymetry
            x_coords = np.linspace(
                self.bathymetry_bounds.left, self.bathymetry_bounds.right, bathy_width
            )
            y_coords = np.linspace(
                self.bathymetry_bounds.top, self.bathymetry_bounds.bottom, bathy_height
            )

            # Transform to lat/lon
            x_grid, y_grid = np.meshgrid(x_coords, y_coords)
            bathy_lons_2d, bathy_lats_2d = transformer.transform(x_grid, y_grid)

        # Flatten coordinate arrays to match bathymetry data
        bathy_lons_flat = bathy_lons_2d.flatten()
        bathy_lats_flat = bathy_lats_2d.flatten()
        bathy_depths_flat = self.bathymetry.flatten()

        # Remove NaN values
        valid_mask = ~np.isnan(bathy_depths_flat)
        bathy_lons_valid = bathy_lons_flat[valid_mask]
        bathy_lats_valid = bathy_lats_flat[valid_mask]
        bathy_depths_valid = bathy_depths_flat[valid_mask]

        print(f"  Valid bathymetry points: {len(bathy_depths_valid):,}")

        if len(bathy_depths_valid) == 0:
            print("Error: No valid bathymetry data for interpolation!")
            return False

        # Interpolate bathymetry to grid points
        grid_lons_flat = self.grid_lons.flatten()
        grid_lats_flat = self.grid_lats.flatten()

        try:
            # Use griddata for interpolation
            print("  Performing spatial interpolation...")
            interpolated_depths = griddata(
                (bathy_lons_valid, bathy_lats_valid),
                bathy_depths_valid,
                (grid_lons_flat, grid_lats_flat),
                method="linear",
                fill_value=0.0,
            )

            self.grid_depths = interpolated_depths.reshape(self.grid_lats.shape)

        except Exception as e:
            print(f"Interpolation failed: {e}")
            print("  Using nearest neighbor fallback...")

            # Fallback to nearest neighbor
            interpolated_depths = griddata(
                (bathy_lons_valid, bathy_lats_valid),
                bathy_depths_valid,
                (grid_lons_flat, grid_lats_flat),
                method="nearest",
                fill_value=0.0,
            )

            self.grid_depths = interpolated_depths.reshape(self.grid_lats.shape)

        # Validate results
        water_points = np.sum(self.grid_depths < 0)
        land_points = np.sum(self.grid_depths >= 0)
        total_points = self.grid_depths.size

        print(f"Bathymetry interpolated:")
        print(
            f"  Water points: {water_points:,} ({water_points / total_points * 100:.1f}%)"
        )
        print(f"  Land points: {land_points:,}")
        print(
            f"  Depth range: {np.min(self.grid_depths):.1f} to {np.max(self.grid_depths):.1f} m"
        )

        if water_points == 0:
            print("Warning: No water areas found in computational grid!")
            return False

        return True

    def get_grid_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        """Get processed grid data for surge calculations."""
        if self.grid_depths is None:
            raise ValueError(
                "Grid data not processed. Call interpolate_bathymetry_to_grid() first."
            )

        return self.grid_lats, self.grid_lons, self.grid_depths, self.dx, self.dy

    def get_domain_bounds(self) -> Dict[str, float]:
        """Get the modeling domain bounds."""
        if self.bathymetry_bounds is None:
            raise ValueError(
                "Bathymetry not loaded. Call load_bathymetry_data() first."
            )

        return {
            "west": self.bathymetry_bounds.left,
            "east": self.bathymetry_bounds.right,
            "south": self.bathymetry_bounds.bottom,
            "north": self.bathymetry_bounds.top,
        }

    def process_bathymetry(self, bathymetry_file: Optional[str] = None) -> bool:
        """Complete bathymetry processing pipeline."""
        print("Processing bathymetry data...")

        # Load bathymetry
        if not self.load_bathymetry_data(bathymetry_file):
            return False

        # Determine domain
        domain = self.determine_modeling_domain()

        # Setup grid
        if not self.setup_computational_grid(domain):
            return False

        # Interpolate bathymetry
        if not self.interpolate_bathymetry_to_grid():
            return False

        print("Bathymetry processing complete!")
        return True
