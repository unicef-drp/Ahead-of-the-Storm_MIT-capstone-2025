import numpy as np
import geopandas as gpd # type: ignore
import matplotlib.pyplot as plt # type: ignore
from typing import Dict, Any
from shapely.geometry import box
from src.impact_analysis.layers.base import VulnerabilityLayer
from src.utils.config_utils import get_config_value
from src.utils.path_utils import get_data_path
from src.utils.hurricane_geom import get_nicaragua_boundary
import os


class SchoolVulnerabilityLayer(VulnerabilityLayer):
    def __init__(self, config, cache_dir=None, resolution_context=None, use_cache=True):
        super().__init__(config, resolution_context)
        self.grid_gdf = None
        self._school_grid = None
        self.use_cache = use_cache
        self.cache_dir = cache_dir or get_config_value(
            config,
            "impact_analysis.output.cache_directory",
            "data/results/impact_analysis/cache/",
        )
        os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_path(self):
        resolution = self.get_resolution()
        if self.resolution_context:
            # Use parquet for high-res computation, gpkg for visualization
            if self.resolution_context == "landslide_computation":
                return os.path.join(self.cache_dir, f"school_vulnerability_{self.resolution_context}_{resolution}deg.parquet")
            else:
                return os.path.join(self.cache_dir, f"school_vulnerability_{self.resolution_context}_{resolution}deg.gpkg")
        else:
            return os.path.join(self.cache_dir, "school_vulnerability.gpkg")

    def compute_grid(self):
        if self.grid_gdf is not None:
            return self.grid_gdf
        cache_path = self._cache_path()

        def compute_func():
            grid_res = self.get_resolution()
            nicaragua_gdf = get_nicaragua_boundary()
            bounds = nicaragua_gdf.total_bounds
            
            # Use raster-based computation for high-res
            if self.resolution_context == "landslide_computation":
                from src.impact_analysis.helper.raster_grid import compute_vulnerability_raster, get_nicaragua_bounds
                
                # Use the same bounds as the exposure layer to ensure grid compatibility
                bounds = get_nicaragua_bounds()
                grid_res = self.get_resolution()
                
                school_data_path = get_config_value(
                    self.config,
                    "impact_analysis.input.school_data",
                    "data/raw/osm/schools.geojson",
                )
                schools_file = get_data_path(school_data_path)
                if schools_file.exists():
                    schools_gdf = gpd.read_file(schools_file)
                    grid_gdf = compute_vulnerability_raster(schools_gdf, bounds, grid_res)
                    print(
                        "Unique school_count values:", np.unique(grid_gdf["school_count"])
                    )
                    print("school_count dtype:", grid_gdf["school_count"].dtype)
                    print(f"High-res vulnerability grid shape: {len(grid_gdf)} cells")
                else:
                    # Fallback to vector grid if no schools data
                    grid_gdf = self._create_vector_grid(bounds, grid_res)
                    grid_gdf["school_count"] = 0
            else:
                # Use vector grid for visualization
                grid_gdf = self._create_vector_grid(bounds, grid_res)
                school_data_path = get_config_value(
                    self.config,
                    "impact_analysis.input.school_data",
                    "data/raw/osm/schools.geojson",
                )
                schools_file = get_data_path(school_data_path)
                if schools_file.exists():
                    schools_gdf = gpd.read_file(schools_file)
                    school_counts = []
                    for cell in grid_gdf.geometry:
                        count = schools_gdf.within(cell).sum()
                        school_counts.append(count)
                    grid_gdf["school_count"] = school_counts
                    print(
                        "Unique school_count values:", np.unique(grid_gdf["school_count"])
                    )
                    print("school_count dtype:", grid_gdf["school_count"].dtype)
                else:
                    grid_gdf["school_count"] = 0
            return grid_gdf

        self.grid_gdf = self._load_or_compute_grid(
            cache_path, "school_count", compute_func, use_cache=self.use_cache
        )
        self._school_grid = self.grid_gdf["school_count"].values
        return self.grid_gdf

    def _create_vector_grid(self, bounds, grid_res):
        """Create vector grid for visualization."""
        minx, miny, maxx, maxy = bounds
        grid_cells = []
        x_coords = np.arange(minx, maxx, grid_res)
        y_coords = np.arange(miny, maxy, grid_res)
        for x in x_coords:
            for y in y_coords:
                grid_cells.append(box(x, y, x + grid_res, y + grid_res))
        return gpd.GeoDataFrame(
            grid_cells, columns=["geometry"], crs="EPSG:4326"
        )

    def get_plot_metadata(self) -> Dict[str, Any]:
        """Return metadata for plotting this school vulnerability layer."""
        return {
            "layer_type": "vulnerability",
            "vulnerability_type": "Schools",
            "data_column": "school_count",
            "colormap": "GnBu",
            "title_template": "Concentration of Schools",
            "legend_template": "Schools per Cell",
            "filename_template": "schools_vulnerability_{parameters}",
            "special_features": []
        }

    def plot(self, ax=None, output_dir="data/results/impact_analysis/"):
        """Plot the school vulnerability layer using universal plotting function."""
        from src.impact_analysis.utils.plotting_utils import plot_layer_with_scales
        
        # Use universal plotting function
        plot_layer_with_scales(self, output_dir=output_dir)

    @property
    def value_column(self):
        return "school_count"


class SchoolPopulationVulnerabilityLayer(VulnerabilityLayer):
    def __init__(self, config, cache_dir=None, resolution_context=None, use_cache=True):
        super().__init__(config, resolution_context)
        self.grid_gdf = None
        self._people_grid = None
        self.use_cache = use_cache
        self.cache_dir = cache_dir or get_config_value(
            config,
            "impact_analysis.output.cache_directory",
            "data/results/impact_analysis/cache/",
        )
        os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_path(self):
        resolution = self.get_resolution()
        if self.resolution_context:
            # Use parquet for high-res computation, gpkg for visualization
            if self.resolution_context == "landslide_computation":
                return os.path.join(self.cache_dir, f"school_population_vulnerability_{self.resolution_context}_{resolution}deg.parquet")
            else:
                return os.path.join(self.cache_dir, f"school_population_vulnerability_{self.resolution_context}_{resolution}deg.gpkg")
        else:
            return os.path.join(self.cache_dir, "school_population_vulnerability.gpkg")

    def compute_grid(self):
        if self.grid_gdf is not None:
            return self.grid_gdf
        cache_path = self._cache_path()

        def compute_func():
            grid_res = self.get_resolution()
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
            school_data_path = get_config_value(
                self.config,
                "impact_analysis.input.school_data",
                "data/raw/osm/schools.geojson",
            )
            schools_file = get_data_path(school_data_path)
            if schools_file.exists():
                schools_gdf = gpd.read_file(schools_file)
                # Use NTOTAL (capacity) for each school
                if "NTOTAL" not in schools_gdf.columns:
                    raise ValueError("NTOTAL column not found in schools data!")
                people_counts = []
                for cell in grid_gdf.geometry:
                    mask = schools_gdf.within(cell)
                    total = schools_gdf.loc[mask, "NTOTAL"].sum()
                    people_counts.append(total)
                grid_gdf["people_count"] = people_counts
                print(
                    "Unique people_count values:", np.unique(grid_gdf["people_count"])
                )
                print("people_count dtype:", grid_gdf["people_count"].dtype)
            else:
                grid_gdf["people_count"] = 0
            return grid_gdf

        self.grid_gdf = self._load_or_compute_grid(
            cache_path, "people_count", compute_func, use_cache=self.use_cache
        )
        self._people_grid = self.grid_gdf["people_count"].values
        return self.grid_gdf

    def get_plot_metadata(self) -> Dict[str, Any]:
        """Return metadata for plotting this school population vulnerability layer."""
        return {
            "layer_type": "vulnerability",
            "vulnerability_type": "School Population",
            "data_column": "people_count",
            "colormap": "GnBu",
            "title_template": "Concentration of School Population",
            "legend_template": "School Population per Cell",
            "filename_template": "school_population_vulnerability_{parameters}",
            "special_features": []
        }

    def plot(self, ax=None, output_dir="data/results/impact_analysis/"):
        """Plot the school population vulnerability layer using universal plotting function."""
        from src.impact_analysis.utils.plotting_utils import plot_layer_with_scales
        
        # Use universal plotting function
        plot_layer_with_scales(self, output_dir=output_dir)

    @property
    def value_column(self):
        return "people_count"
