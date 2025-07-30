#!/usr/bin/env python3
"""
Landslide exposure layer for impact analysis.
"""

import os
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from typing import Dict, Any
from rasterio.mask import geometry_mask
from scipy.ndimage import gaussian_filter
from shapely.geometry import box
from src.impact_analysis.layers.base import ExposureLayer
from src.utils.config_utils import get_config_value
from src.utils.path_utils import get_data_path
from src.utils.hurricane_geom import get_nicaragua_boundary



class LandslideExposureLayer(ExposureLayer):
    def __init__(self, landslide_file, config, cache_dir=None, resampling_method="mean", resolution_context=None):
        super().__init__(config, resolution_context)
        self.landslide_file = landslide_file
        self.resampling_method = resampling_method
        self.grid_gdf = None
        self._prob_grid = None
        self.cache_dir = cache_dir or get_config_value(
            config,
            "impact_analysis.output.cache_directory",
            "data/results/impact_analysis/cache/",
        )
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Ensemble parameters
        self.ensemble_config = get_config_value(
            config, 
            "impact_analysis.landslide.ensemble", 
            {
                "num_members": 50,
                "spatial_correlation_radius": 2,
                "correlation_method": "gaussian_local",
                "random_seed": 42
            }
        )
        self.ensemble_members = None
        self.ensemble_impacts = None

    def get_resolution(self):
        """Get resolution based on context."""
        if self.resolution_context == "landslide_computation":
            return get_config_value(
                self.config, 
                "impact_analysis.grid.landslide_computation_resolution_degrees", 
                0.01
            )
        elif self.resolution_context == "landslide_visualization":
            return get_config_value(
                self.config, 
                "impact_analysis.grid.landslide_visualization_resolution_degrees", 
                0.1
            )
        else:
            return get_config_value(
                self.config, 
                "impact_analysis.grid.resolution_degrees", 
                0.1
            )

    def _cache_path(self):
        resolution = self.get_resolution()
        if self.resolution_context:
            # Use parquet for high-res computation, gpkg for visualization
            if self.resolution_context == "landslide_computation":
                return os.path.join(self.cache_dir, f"landslide_exposure_20250716T0600_{self.resampling_method}_{self.resolution_context}_{resolution}deg.parquet")
            else:
                return os.path.join(self.cache_dir, f"landslide_exposure_20250716T0600_{self.resampling_method}_{self.resolution_context}_{resolution}deg.gpkg")
        else:
            return os.path.join(self.cache_dir, f"landslide_exposure_20250716T0600_{self.resampling_method}.gpkg")

    def compute_grid(self):
        if self.grid_gdf is not None:
            return self.grid_gdf
        
        cache_path = self._cache_path()
        
        def compute_func():
            # Use raster-based computation for high-res
            if self.resolution_context == "landslide_computation":
                from src.impact_analysis.helper.raster_grid import compute_exposure_raster, get_nicaragua_bounds
                
                # If landslide_file is "cached", skip computation since cache will be used
                if self.landslide_file == "cached":
                    print("Using cached landslide data, skipping high-res computation")
                    # Return a dummy grid that will be replaced by cache
                    bounds = get_nicaragua_bounds()
                    grid_res = self.get_resolution()
                    # Create a simple grid structure
                    minx, miny, maxx, maxy = bounds
                    grid_cells = []
                    x_coords = np.arange(minx, maxx, grid_res)
                    y_coords = np.arange(miny, maxy, grid_res)
                    
                    for x in x_coords:
                        for y in y_coords:
                            grid_cells.append(box(x, y, x + grid_res, y + grid_res))
                    
                    grid_gdf = gpd.GeoDataFrame(grid_cells, columns=["geometry"], crs="EPSG:4326")
                    grid_gdf["probability"] = 0.0  # Dummy values, will be replaced by cache
                    return grid_gdf
                
                bounds = get_nicaragua_bounds()
                grid_res = self.get_resolution()
                print(f"High-res exposure grid bounds: {bounds}")
                print(f"High-res exposure grid resolution: {grid_res} degrees")
                grid_gdf = compute_exposure_raster(
                    self.landslide_file, 
                    bounds, 
                    grid_res, 
                    self.resampling_method
                )
                print(f"High-res exposure grid shape: {len(grid_gdf)} cells")
            else:
                # Use vector grid for visualization
                grid_res = self.get_resolution()
                nicaragua_gdf = get_nicaragua_boundary()
                minx, miny, maxx, maxy = nicaragua_gdf.total_bounds
                
                grid_cells = []
                x_coords = np.arange(minx, maxx, grid_res)
                y_coords = np.arange(miny, maxy, grid_res)
                
                for x in x_coords:
                    for y in y_coords:
                        grid_cells.append(box(x, y, x + grid_res, y + grid_res))
                
                grid_gdf = gpd.GeoDataFrame(grid_cells, columns=["geometry"], crs="EPSG:4326")
                
                # Sample landslide raster values to grid
                probabilities = self._sample_raster_to_grid(grid_gdf)
                grid_gdf["probability"] = probabilities
                
                # Ensure non-negative probabilities
                grid_gdf.loc[grid_gdf["probability"] < 0, "probability"] = 0.0
            
            return grid_gdf
        
        # Use the shared cache loading/saving logic
        self.grid_gdf = self._load_or_compute_grid(cache_path, "probability", compute_func)
        self._prob_grid = self.grid_gdf["probability"].values
        return self.grid_gdf

    def _load_or_compute_grid(self, cache_path, value_column, compute_func):
        """
        Shared logic for loading from cache, computing, filling NaNs, and saving.
        - cache_path: path to the cache file
        - value_column: name of the column to check/fill NaNs
        - compute_func: function to call to compute the grid if not cached
        """
        import os
        import geopandas as gpd

        if os.path.exists(cache_path):
            print(f"Loading cached landslide exposure layer ({self.resampling_method}): {cache_path}")
            # Use parquet for high-res computation, gpkg for visualization
            if cache_path.endswith('.parquet'):
                grid_gdf = gpd.read_parquet(cache_path)
            else:
                grid_gdf = gpd.read_file(cache_path)
            # Fill NaNs in value column with 0
            grid_gdf[value_column] = grid_gdf[value_column].fillna(0)
            return grid_gdf
        # Compute grid using provided function
        grid_gdf = compute_func()
        # Fill NaNs in value column with 0
        grid_gdf[value_column] = grid_gdf[value_column].fillna(0)
        # Save using appropriate format
        if cache_path.endswith('.parquet'):
            grid_gdf.to_parquet(cache_path)
        else:
            grid_gdf.to_file(cache_path, driver="GPKG")
        print(f"Saved landslide exposure layer ({self.resampling_method}) to cache: {cache_path}")
        return grid_gdf

    def _sample_raster_to_grid(self, grid_gdf):
        """Sample landslide raster values to grid cells using specified resampling method."""
        import rasterio
        
        probabilities = []
        
        # If landslide_file is "cached", skip raster reading since cache will be used
        if self.landslide_file == "cached":
            print("Using cached landslide data, skipping raster file reading")
            return [0.0] * len(grid_gdf)  # Return zeros, cache will be loaded instead
        
        try:
            with rasterio.open(self.landslide_file) as src:
                raster_data = src.read(1)
                transform = src.transform
                raster_crs = src.crs
                nodata = src.nodata
                
                print(f"Landslide raster info: shape={raster_data.shape}, CRS={raster_crs}, nodata={nodata}")
                print(f"Raster data range: {np.nanmin(raster_data):.3f} to {np.nanmax(raster_data):.3f}")
                print(f"Using resampling method: {self.resampling_method}")
                
                # Reproject grid to raster CRS if needed
                grid_in_raster_crs = grid_gdf.copy()
                if grid_gdf.crs != raster_crs:
                    grid_in_raster_crs = grid_gdf.to_crs(raster_crs)
                
                # Robust masking following landslide_heatmap.py pattern
                if nodata is not None:
                    valid_mask = ~(np.isclose(raster_data, nodata) | (raster_data < 0) | np.isnan(raster_data))
                else:
                    valid_mask = ~((raster_data < 0) | np.isnan(raster_data))
                
                # For each grid cell, calculate probability using specified method
                for idx, cell_geom in enumerate(grid_in_raster_crs.geometry):
                    try:
                        # Create mask for this grid cell
                        cell_mask = ~geometry_mask(
                            [cell_geom.__geo_interface__], 
                            out_shape=raster_data.shape,
                            transform=transform,
                            invert=False
                        )
                        
                        # Combine with valid data mask
                        combined_mask = cell_mask & valid_mask
                        
                        if combined_mask.any():
                            # Calculate probability using specified resampling method
                            cell_values = raster_data[combined_mask]
                            
                            if self.resampling_method == "mean":
                                prob_value = float(np.mean(cell_values))
                            elif self.resampling_method == "min":
                                prob_value = float(np.min(cell_values))
                            elif self.resampling_method == "max":
                                prob_value = float(np.max(cell_values))
                            else:
                                raise ValueError(f"Unknown resampling method: {self.resampling_method}")
                            
                            # Ensure probability is between 0 and 1
                            prob_value = np.clip(prob_value, 0.0, 1.0)
                        else:
                            # No valid data in this cell
                            prob_value = 0.0  # Assign 0 like hurricane layer
                        
                        probabilities.append(prob_value)
                        
                        # Log progress for large grids
                        if (idx + 1) % 500 == 0:
                            print(f"Processed {idx + 1}/{len(grid_in_raster_crs)} grid cells")
                            
                    except Exception as e:
                        print(f"Warning: Error processing grid cell {idx}: {e}")
                        probabilities.append(0.0)  # Assign 0 on error
                
        except Exception as e:
            print(f"Error reading landslide raster: {e}")
            # Return all zeros if raster can't be read
            probabilities = [0.0] * len(grid_gdf)
        
        print(f"Sampled {len(probabilities)} grid cells from landslide raster")
        valid_probs = [p for p in probabilities if p > 0]
        if valid_probs:
            print(f"Valid probability range: {min(valid_probs):.3f} to {max(valid_probs):.3f}")
        print(f"Grid cells with probability > 0: {len(valid_probs)}")
        
        return probabilities

    def apply_spatial_correlation(self, probability_grid, correlation_radius=None):
        """
        Apply spatial correlation to probability grid.
        
        Args:
            probability_grid: 1D numpy array of probabilities
            correlation_radius: Radius for spatial smoothing (in grid cells)
        
        Returns:
            correlated_probabilities: 1D numpy array with spatial correlation
        """
        if correlation_radius is None:
            correlation_radius = self.ensemble_config.get("spatial_correlation_radius", 2)
        
        # For now, use a simpler approach: just apply Gaussian smoothing to the 1D array
        # This maintains the original grid structure without reshaping issues
        prob_array = np.array(probability_grid)
        
        # Apply simple smoothing using convolution with a Gaussian kernel
        # Create a simple 1D Gaussian kernel
        kernel_size = 2 * correlation_radius + 1
        kernel = np.exp(-0.5 * (np.arange(kernel_size) - kernel_size//2)**2 / correlation_radius**2)
        kernel = kernel / kernel.sum()  # Normalize
        
        # Apply convolution
        smoothed = np.convolve(prob_array, kernel, mode='same')
        
        return smoothed

    def generate_ensemble_member(self, correlated_probabilities, seed=None):
        """
        Generate single ensemble member using biased coin flipping.
        
        Args:
            correlated_probabilities: 1D array of spatially correlated probabilities
            seed: Random seed for reproducibility
        
        Returns:
            binary_landslides: 1D binary array (1 = landslide, 0 = no landslide)
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Biased coin flipping
        random_values = np.random.random(len(correlated_probabilities))
        binary_landslides = (random_values < correlated_probabilities).astype(int)
        
        return binary_landslides

    def generate_ensemble(self, num_members=None):
        """
        Generate ensemble of landslide realizations.
        
        Args:
            num_members: Number of ensemble members to generate
        
        Returns:
            ensemble_members: List of binary landslide arrays
        """
        if num_members is None:
            num_members = self.ensemble_config.get("num_members", 50)
        
        # Get mean probability grid from computation resolution
        # Create a temporary exposure layer with computation context
        temp_exposure = LandslideExposureLayer(
            self.landslide_file, 
            self.config, 
            self.cache_dir, 
            self.resampling_method, 
            resolution_context="landslide_computation"
        )
        grid_gdf = temp_exposure.compute_grid()
        mean_probabilities = grid_gdf["probability"].values
        
        print(f"Generating {num_members} ensemble members...")
        
        # Apply spatial correlation
        correlated_probabilities = self.apply_spatial_correlation(mean_probabilities)
        
        # Generate ensemble members
        ensemble_members = []
        base_seed = self.ensemble_config.get("random_seed", 42)
        
        for i in range(num_members):
            member = self.generate_ensemble_member(correlated_probabilities, seed=base_seed + i)
            ensemble_members.append(member)
        
        self.ensemble_members = ensemble_members
        return ensemble_members

    def get_ensemble_impact(self, vulnerability_layer):
        """
        Compute impact for each ensemble member.
        
        Args:
            vulnerability_layer: Vulnerability layer to use for impact calculation
        
        Returns:
            ensemble_impacts: List of impact values for each ensemble member
        """
        if self.ensemble_members is None:
            self.generate_ensemble()
        
        # Get vulnerability grid
        vuln_grid = vulnerability_layer.compute_grid()
        vulnerability_values = vuln_grid[vulnerability_layer.value_column].values
        
        # Compute impact for each ensemble member
        ensemble_impacts = []
        for member in self.ensemble_members:
            # Impact = sum of (landslide * vulnerability) across all cells
            impact = np.sum(member * vulnerability_values)
            ensemble_impacts.append(impact)
        
        self.ensemble_impacts = ensemble_impacts
        return ensemble_impacts

    def get_best_worst_case(self, vulnerability_layer):
        """
        Get best and worst case impacts from ensemble.
        
        Args:
            vulnerability_layer: Vulnerability layer to use for impact calculation
        
        Returns:
            best_case: Minimum impact from ensemble
            worst_case: Maximum impact from ensemble
        """
        ensemble_impacts = self.get_ensemble_impact(vulnerability_layer)
        
        best_case = min(ensemble_impacts)
        worst_case = max(ensemble_impacts)
        
        return best_case, worst_case

    def get_plot_metadata(self) -> Dict[str, Any]:
        """Return metadata for plotting this landslide exposure layer."""
        return {
            "layer_type": "exposure",
            "hazard_type": "Landslide",
            "data_column": "probability",
            "colormap": "YlOrBr",
            "title_template": "Probability of Forecasted Landslide",
            "legend_template": "Landslide Probability per Cell",
            "filename_template": "landslide_exposure_{parameters}",
            "special_features": ["ensemble_method"]
        }

    def plot(self, ax=None, output_dir="data/results/impact_analysis/"):
        """Plot the landslide exposure layer using universal plotting function."""
        from src.impact_analysis.utils.plotting_utils import plot_layer_with_scales
        
        # Use universal plotting function
        plot_layer_with_scales(self, output_dir=output_dir)

    def get_grid_cells(self):
        """Return the grid GeoDataFrame (geometry only, no data columns)."""
        # Use the same grid creation logic as hurricane layer
        grid_res = get_config_value(
            self.config, "impact_analysis.grid.resolution_degrees", 0.1
        )
        nicaragua_gdf = get_nicaragua_boundary()
        minx, miny, maxx, maxy = nicaragua_gdf.total_bounds
        
        grid_cells = []
        x_coords = np.arange(minx, maxx, grid_res)
        y_coords = np.arange(miny, maxy, grid_res)
        
        for x in x_coords:
            for y in y_coords:
                grid_cells.append(box(x, y, x + grid_res, y + grid_res))
                
        return gpd.GeoDataFrame(grid_cells, columns=["geometry"], crs="EPSG:4326")

    def get_computation_grid(self):
        """Get high-resolution grid for computation."""
        if self.resolution_context != "landslide_computation":
            temp_layer = LandslideExposureLayer(
                self.landslide_file, 
                self.config, 
                self.cache_dir, 
                self.resampling_method, 
                "landslide_computation"
            )
            return temp_layer.compute_grid()
        else:
            return self.compute_grid()

    def get_visualization_grid(self):
        """Get standard resolution grid for visualization."""
        if self.resolution_context != "landslide_visualization":
            temp_layer = LandslideExposureLayer(
                self.landslide_file, 
                self.config, 
                self.cache_dir, 
                self.resampling_method, 
                "landslide_visualization"
            )
            return temp_layer.compute_grid()
        else:
            return self.compute_grid()

    def get_landslide_regions(self):
        """Return regions where landslide probability > threshold for compatibility."""
        # For compatibility with impact analysis - return grid cells with probability > 0
        grid_gdf = self.compute_grid()
        return [geom for geom, prob in zip(grid_gdf.geometry, grid_gdf["probability"]) if prob > 0] 