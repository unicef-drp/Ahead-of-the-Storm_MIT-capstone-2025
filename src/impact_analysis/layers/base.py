import abc
import numpy as np
import geopandas as gpd


class ExposureLayer(abc.ABC):
    """Abstract base class for hazard exposure layers (e.g., hurricane, flood)."""

    def __init__(self, config):
        self.config = config

    @abc.abstractmethod
    def compute_grid(self) -> gpd.GeoDataFrame:
        """Compute the exposure probability grid (0-1 per cell)."""
        pass

    @abc.abstractmethod
    def plot(self, ax=None):
        """Plot the exposure layer."""
        pass


class VulnerabilityLayer(abc.ABC):
    """Abstract base class for vulnerability layers (e.g., schools, hospitals, population)."""

    def __init__(self, config):
        self.config = config

    @abc.abstractmethod
    def compute_grid(self) -> gpd.GeoDataFrame:
        """Compute the vulnerability grid (entity count/density per cell)."""
        pass

    @abc.abstractmethod
    def plot(self, ax=None):
        """Plot the vulnerability layer."""
        pass

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
            grid_gdf = gpd.read_file(cache_path)
            # Fill NaNs in value column with 0
            grid_gdf[value_column] = grid_gdf[value_column].fillna(0)
            return grid_gdf
        # Compute grid using provided function
        grid_gdf = compute_func()
        # Fill NaNs in value column with 0
        grid_gdf[value_column] = grid_gdf[value_column].fillna(0)
        grid_gdf.to_file(cache_path, driver="GPKG")
        print(f"Saved vulnerability layer to cache: {cache_path}")
        return grid_gdf

    def _plot_vulnerability_grid(
        self,
        grid_gdf,
        value_column,
        cmap,
        legend_label,
        output_dir,
        output_filename,
        plot_title,
        ax=None,
    ):
        import matplotlib.pyplot as plt
        from src.utils.hurricane_geom import get_nicaragua_boundary
        import os

        nicaragua_gdf = get_nicaragua_boundary()
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 10))
        log_col = f"log_{value_column}"
        grid_gdf[log_col] = [np.log10(val + 1) for val in grid_gdf[value_column]]
        grid_gdf.plot(
            ax=ax,
            column=log_col,
            cmap=cmap,
            linewidth=0.1,
            edgecolor="grey",
            alpha=0.7,
            legend=True,
            legend_kwds={"label": legend_label},
        )
        nicaragua_gdf.plot(
            ax=ax, color="none", edgecolor="black", linewidth=3, alpha=1.0
        )
        ax.set_title(plot_title)
        plt.tight_layout()
        out_path = os.path.join(output_dir, output_filename)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved vulnerability plot: {out_path}")
        plt.close(fig)


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
