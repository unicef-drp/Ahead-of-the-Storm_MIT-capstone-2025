import abc
import numpy as np
import geopandas as gpd
from typing import Dict, Any, Tuple
from src.utils.config_utils import get_config_value


class ExposureLayer(abc.ABC):
    """Abstract base class for hazard exposure layers (e.g., hurricane, flood)."""

    def __init__(self, config, resolution_context=None):
        self.config = config
        self.resolution_context = resolution_context

    @abc.abstractmethod
    def compute_grid(self) -> gpd.GeoDataFrame:
        """Compute the exposure probability grid (0-1 per cell)."""
        pass

    @abc.abstractmethod
    def plot(self, ax=None):
        """Plot the exposure layer."""
        pass

    @abc.abstractmethod
    def get_plot_metadata(self) -> Dict[str, Any]:
        """Return metadata for plotting this exposure layer."""
        pass

    def get_plot_data(self) -> Tuple[str, np.ndarray]:
        """Return data column name and values for plotting."""
        grid_gdf = self.compute_grid()
        return "probability", grid_gdf["probability"].values


class VulnerabilityLayer(abc.ABC):
    """Abstract base class for vulnerability layers (e.g., schools, hospitals, population)."""

    def __init__(self, config, resolution_context=None):
        self.config = config
        self.resolution_context = resolution_context  # For landslide high-res computation

    @abc.abstractmethod
    def compute_grid(self) -> gpd.GeoDataFrame:
        """Compute the vulnerability grid (entity count/density per cell)."""
        pass

    @abc.abstractmethod
    def plot(self, ax=None):
        """Plot the vulnerability layer."""
        pass

    @abc.abstractmethod
    def get_plot_metadata(self) -> Dict[str, Any]:
        """Return metadata for plotting this vulnerability layer."""
        pass

    def get_plot_data(self) -> Tuple[str, np.ndarray]:
        """Return data column name and values for plotting."""
        grid_gdf = self.compute_grid()
        value_col = getattr(self, "value_column", "school_count")
        return value_col, grid_gdf[value_col].values

    def get_visualization_grid(self):
        """Get the vulnerability grid at visualization resolution."""
        if self.resolution_context == "landslide_visualization":
            return self.compute_grid()
        else:
            # Create temporary layer with visualization context
            temp_layer = type(self)(
                self.config,
                cache_dir=self.cache_dir,
                resolution_context="landslide_visualization"
            )
            return temp_layer.compute_grid()

    def get_resolution(self):
        """Get the appropriate resolution based on context."""
        if self.resolution_context == "landslide_computation":
            return get_config_value(
                self.config, "impact_analysis.grid.landslide.computation_resolution", 0.01
            )
        elif self.resolution_context == "landslide_visualization":
            return get_config_value(
                self.config, "impact_analysis.grid.landslide.visualization_resolution", 0.1
            )
        else:
            return get_config_value(
                self.config, "impact_analysis.grid.resolution_degrees", 0.1
            )

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
            print(f"Loading cached vulnerability layer: {cache_path}")
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
        print(f"Saved vulnerability layer to cache: {cache_path}")
        return grid_gdf




class ImpactLayer(abc.ABC):
    """Abstract base class for impact layers (combining exposure and vulnerability)."""

    def __init__(
        self,
        exposure_layer: ExposureLayer,
        vulnerability_layer: VulnerabilityLayer,
        config,
    ):
        self.exposure_layer = exposure_layer
        self.vulnerability_layer = vulnerability_layer
        self.config = config

    @abc.abstractmethod
    def compute_impact(self) -> gpd.GeoDataFrame:
        """Compute the impact grid (e.g., expected affected entities per cell)."""
        pass

    @abc.abstractmethod
    def plot(self, ax=None):
        """Plot the impact layer."""
        pass

    @abc.abstractmethod
    def get_plot_metadata(self) -> Dict[str, Any]:
        """Return metadata for plotting this impact layer."""
        pass

    def get_plot_data(self) -> Tuple[str, np.ndarray]:
        """Return data column name and values for plotting."""
        impact_gdf = self.compute_impact()
        return "expected_impact", impact_gdf["expected_impact"].values

    @abc.abstractmethod
    def expected_impact(self) -> float:
        """Compute the expected number of affected entities (sum over grid)."""
        pass

    @abc.abstractmethod
    def best_case(self) -> float:
        """Compute the best-case (minimum) number of affected entities."""
        pass

    @abc.abstractmethod
    def worst_case(self) -> float:
        """Compute the worst-case (maximum) number of affected entities."""
        pass
