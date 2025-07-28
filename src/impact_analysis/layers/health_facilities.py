import os
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box
from src.impact_analysis.layers.base import VulnerabilityLayer
from src.utils.config_utils import get_config_value
from src.utils.path_utils import get_data_path
from src.utils.hurricane_geom import get_nicaragua_boundary


class HealthFacilityVulnerabilityLayer(VulnerabilityLayer):
    def __init__(self, config, weighted_by_population=False, cache_dir=None, resolution_context=None):
        super().__init__(config, resolution_context)
        self.weighted_by_population = weighted_by_population
        self.grid_gdf = None
        self._facility_grid = None
        self.cache_dir = cache_dir or get_config_value(
            config,
            "impact_analysis.output.cache_directory",
            "data/results/impact_analysis/cache/",
        )
        os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_path(self):
        suffix = "population" if self.weighted_by_population else "count"
        resolution = self.get_resolution()
        if self.resolution_context:
            # Use parquet for high-res computation, gpkg for visualization
            if self.resolution_context == "landslide_computation":
                return os.path.join(self.cache_dir, f"health_facility_vulnerability_{suffix}_{self.resolution_context}_{resolution}deg.parquet")
            else:
                return os.path.join(self.cache_dir, f"health_facility_vulnerability_{suffix}_{self.resolution_context}_{resolution}deg.gpkg")
        else:
            return os.path.join(
                self.cache_dir, f"health_facility_vulnerability_{suffix}.gpkg"
            )

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
            # Load health facilities
            health_data_path = get_config_value(
                self.config,
                "impact_analysis.input.health_facility_data",
                "data/raw/sinapred/health_facilities_nic.geojson",
            )
            health_file = get_data_path(health_data_path)
            if health_file.exists():
                health_gdf = gpd.read_file(health_file)
                if health_gdf.crs != "EPSG:4326":
                    health_gdf = health_gdf.to_crs("EPSG:4326")
                if self.weighted_by_population:
                    if "POBLACIÓN" not in health_gdf.columns:
                        raise ValueError(
                            "POBLACIÓN column not found in health facilities data!"
                        )
                    pop_counts = []
                    for cell in grid_gdf.geometry:
                        mask = health_gdf.intersects(cell)
                        total = health_gdf.loc[mask, "POBLACIÓN"].sum()
                        pop_counts.append(total)
                    grid_gdf["population"] = pop_counts
                else:
                    facility_counts = []
                    for cell in grid_gdf.geometry:
                        count = health_gdf.intersects(cell).sum()
                        facility_counts.append(count)
                    grid_gdf["facility_count"] = facility_counts
            else:
                grid_gdf["facility_count"] = 0
                grid_gdf["population"] = 0
            return grid_gdf

        self.grid_gdf = self._load_or_compute_grid(
            cache_path, value_column, compute_func
        )
        self._facility_grid = self.grid_gdf[value_column].values
        return self.grid_gdf

    def plot(self, ax=None, output_dir="data/results/impact_analysis/"):
        grid_gdf = self.compute_grid()
        if self.weighted_by_population:
            output_filename = "health_facility_population_vulnerability.png"
            plot_title = "Health Facility Vulnerability Heatmap (Log Scale) (Population Weighted)"
            value_column = "population"
            cmap = "Reds"
            legend_label = "Log10(Population + 1) per Cell"
        else:
            output_filename = "health_facility_vulnerability.png"
            plot_title = "Health Facility Vulnerability Heatmap (Log Scale)"
            value_column = "facility_count"
            cmap = "Blues"
            legend_label = "Log10(Facilities + 1) per Cell"
        self._plot_vulnerability_grid(
            grid_gdf,
            value_column=value_column,
            cmap=cmap,
            legend_label=legend_label,
            output_dir=output_dir,
            output_filename=output_filename,
            plot_title=plot_title,
            ax=ax,
        )

    @property
    def value_column(self):
        return "population" if self.weighted_by_population else "facility_count"
