import os
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import box
from src.impact_analysis.layers.base import ExposureLayer
from src.utils.hurricane_geom import get_nicaragua_boundary
from src.utils.config_utils import get_config_value
from src.utils.path_utils import get_data_path


class LandslideExposureLayer(ExposureLayer):
    """Landslide exposure layer that reads raster data and creates probability grid."""
    
    def __init__(self, landslide_file, config, cache_dir=None, resampling_method="mean"):
        super().__init__(config)
        self.landslide_file = landslide_file
        self.resampling_method = resampling_method  # mean, min, max
        self.grid_gdf = None
        self._prob_grid = None
        self.cache_dir = cache_dir or get_config_value(
            config,
            "impact_analysis.output.cache_directory",
            "data/results/impact_analysis/cache/",
        )
        os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_path(self):
        """Generate cache path for landslide exposure layer with resampling method."""
        # Extract date from filename if possible, otherwise use 'unknown'
        import re
        from pathlib import Path
        
        filename = Path(self.landslide_file).name
        match = re.search(r'(\d{8}T\d{4})', filename)
        date_str = match.group(1) if match else 'unknown'
        return os.path.join(self.cache_dir, f"landslide_exposure_{date_str}_{self.resampling_method}.gpkg")

    def compute_grid(self):
        """Compute the landslide exposure probability grid matching hurricane grid structure."""
        if self.grid_gdf is not None:
            return self.grid_gdf
            
        cache_path = self._cache_path()
        if os.path.exists(cache_path):
            print(f"Loading cached landslide exposure layer ({self.resampling_method}): {cache_path}")
            self.grid_gdf = gpd.read_file(cache_path)
            self._prob_grid = self.grid_gdf["probability"].values
            return self.grid_gdf

        # Create the same grid structure as hurricane layer - FULL EXTENT
        grid_res = get_config_value(
            self.config, "impact_analysis.grid.resolution_degrees", 0.1
        )
        nicaragua_gdf = get_nicaragua_boundary()
        minx, miny, maxx, maxy = nicaragua_gdf.total_bounds
        
        # Create grid cells using the same pattern as hurricane layer
        grid_cells = []
        x_coords = np.arange(minx, maxx, grid_res)
        y_coords = np.arange(miny, maxy, grid_res)
        
        for x in x_coords:
            for y in y_coords:
                grid_cells.append(box(x, y, x + grid_res, y + grid_res))
        
        grid_gdf = gpd.GeoDataFrame(grid_cells, columns=["geometry"], crs="EPSG:4326")
        
        # Sample landslide raster values for each grid cell
        probabilities = self._sample_raster_to_grid(grid_gdf)
        grid_gdf["probability"] = probabilities
        
        # Keep ALL grid cells like hurricane layer - assign 0 to cells with no valid data
        grid_gdf["probability"] = grid_gdf["probability"].fillna(0.0)
        grid_gdf.loc[grid_gdf["probability"] < 0, "probability"] = 0.0
        
        self.grid_gdf = grid_gdf
        self._prob_grid = grid_gdf["probability"].values
        
        # Save to cache
        grid_gdf.to_file(cache_path, driver="GPKG")
        print(f"Saved landslide exposure layer ({self.resampling_method}) to cache: {cache_path}")
        return grid_gdf

    def _sample_raster_to_grid(self, grid_gdf):
        """Sample landslide raster values to grid cells using specified resampling method."""
        probabilities = []
        
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

    def plot(self, ax=None, output_dir="data/results/impact_analysis/"):
        """Plot the landslide exposure layer."""
        grid_gdf = self.compute_grid()
        nicaragua_gdf = get_nicaragua_boundary()
        
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 10))
            
        # Plot the probability grid - like hurricane layer, plot ALL cells
        grid_gdf.plot(
            ax=ax,
            column="probability",
            cmap="YlOrRd",
            linewidth=0.1,
            edgecolor="grey",
            alpha=0.7,
            legend=True,
            legend_kwds={"label": f"Landslide Probability per Cell ({self.resampling_method})"},
        )
        
        # Add Nicaragua boundary
        if nicaragua_gdf is not None:
            nicaragua_gdf.plot(
                ax=ax, color="none", edgecolor="black", linewidth=3, alpha=1.0
            )
        
        ax.set_title(f"Landslide Exposure Probability Heatmap ({self.resampling_method.title()})")
        plt.tight_layout()
        
        # Save to disk
        from pathlib import Path
        import re
        
        filename = Path(self.landslide_file).name
        match = re.search(r'(\d{8}T\d{4})', filename)
        date_str = match.group(1) if match else 'unknown'
        
        out_path = os.path.join(output_dir, f"landslide_exposure_{date_str}_{self.resampling_method}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved landslide exposure plot ({self.resampling_method}): {out_path}")
        
        if fig:
            plt.close(fig)

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

    def get_landslide_regions(self):
        """Return regions where landslide probability > threshold for compatibility."""
        # For compatibility with impact analysis - return grid cells with probability > 0
        grid_gdf = self.compute_grid()
        return [geom for geom, prob in zip(grid_gdf.geometry, grid_gdf["probability"]) if prob > 0] 