import os
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box
from typing import Dict, Any
from src.impact_analysis.layers.base import VulnerabilityLayer
from src.utils.config_utils import get_config_value
from src.utils.path_utils import get_data_path
from src.utils.hurricane_geom import get_nicaragua_boundary


class ShelterVulnerabilityLayer(VulnerabilityLayer):
    def __init__(self, config, weighted_by_capacity=False, cache_dir=None, resolution_context=None, use_cache=True):
        super().__init__(config, resolution_context)
        self.weighted_by_capacity = weighted_by_capacity
        self.grid_gdf = None
        self._shelter_grid = None
        self.use_cache = use_cache
        self.cache_dir = cache_dir or get_config_value(
            config,
            "impact_analysis.output.cache_directory",
            "data/results/impact_analysis/cache/",
        )
        os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_path(self):
        suffix = "capacity" if self.weighted_by_capacity else "count"
        resolution = self.get_resolution()
        if self.resolution_context:
            # Use parquet for high-res computation, gpkg for visualization
            if self.resolution_context == "landslide_computation":
                return os.path.join(self.cache_dir, f"shelter_vulnerability_{suffix}_{self.resolution_context}_{resolution}deg.parquet")
            else:
                return os.path.join(self.cache_dir, f"shelter_vulnerability_{suffix}_{self.resolution_context}_{resolution}deg.gpkg")
        else:
            return os.path.join(self.cache_dir, f"shelter_vulnerability_{suffix}.gpkg")

    def compute_grid(self):
        if self.grid_gdf is not None:
            return self.grid_gdf
        cache_path = self._cache_path()
        value_column = self.value_column

        def compute_func():
            if self.resolution_context == "landslide_computation":
                from src.impact_analysis.helper.raster_grid import get_nicaragua_bounds
                
                # Use the same bounds as the exposure layer to ensure grid compatibility
                bounds = get_nicaragua_bounds()
                grid_res = self.get_resolution()
                grid_cells = []
                x_coords = np.arange(bounds[0], bounds[2], grid_res)
                y_coords = np.arange(bounds[1], bounds[3], grid_res)
                for x in x_coords:
                    for y in y_coords:
                        grid_cells.append(box(x, y, x + grid_res, y + grid_res))
                grid_gdf = gpd.GeoDataFrame(
                    grid_cells, columns=["geometry"], crs="EPSG:4326"
                )
            else:
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
                grid_gdf = gpd.GeoDataFrame(
                    grid_cells, columns=["geometry"], crs="EPSG:4326"
                )
            # Load shelters
            shelter_data_path = get_config_value(
                self.config,
                "impact_analysis.input.shelter_data",
                "data/raw/sinapred/shelters_nic.geojson",
            )
            shelter_file = get_data_path(shelter_data_path)
            if shelter_file.exists():
                shelter_gdf = gpd.read_file(shelter_file)
                if shelter_gdf.crs != "EPSG:4326":
                    shelter_gdf = shelter_gdf.to_crs("EPSG:4326")
                if self.weighted_by_capacity:
                    if "Capacidad_" not in shelter_gdf.columns:
                        raise ValueError(
                            "Capacidad_ column not found in shelters data!"
                        )
                    cap_counts = []
                    for cell in grid_gdf.geometry:
                        mask = shelter_gdf.intersects(cell)
                        total = shelter_gdf.loc[mask, "Capacidad_"].sum()
                        cap_counts.append(total)
                    grid_gdf["capacity"] = cap_counts
                else:
                    shelter_counts = []
                    for cell in grid_gdf.geometry:
                        count = shelter_gdf.intersects(cell).sum()
                        shelter_counts.append(count)
                    grid_gdf["shelter_count"] = shelter_counts
            else:
                grid_gdf["shelter_count"] = 0
                grid_gdf["capacity"] = 0
            return grid_gdf

        self.grid_gdf = self._load_or_compute_grid(
            cache_path, value_column, compute_func, use_cache=self.use_cache
        )
        self._shelter_grid = self.grid_gdf[value_column].values
        return self.grid_gdf

    def get_plot_metadata(self) -> Dict[str, Any]:
        """Return metadata for plotting this shelter vulnerability layer."""
        if self.weighted_by_capacity:
            vulnerability_type = "Shelter Population"
            colormap = "YlGn"
        else:
            vulnerability_type = "Shelters"
            colormap = "YlGn"
        
        return {
            "layer_type": "vulnerability",
            "vulnerability_type": vulnerability_type,
            "data_column": self.value_column,
            "colormap": colormap,
            "title_template": "Concentration of {vulnerability_type}",
            "legend_template": "{vulnerability_type} per Cell",
            "filename_template": "{vulnerability_type}_vulnerability_{parameters}",
            "special_features": []
        }

    def plot(self, ax=None, output_dir="data/results/impact_analysis/"):
        """Plot the shelter vulnerability layer using universal plotting function."""
        from src.impact_analysis.utils.plotting_utils import plot_layer_with_scales
        plot_layer_with_scales(self, output_dir=output_dir)

    @property
    def value_column(self):
        return "capacity" if self.weighted_by_capacity else "shelter_count"
