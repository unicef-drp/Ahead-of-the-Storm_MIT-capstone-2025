import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box
from src.impact_analysis.layers.base import VulnerabilityLayer
from src.utils.config_utils import get_config_value
from src.utils.path_utils import get_data_path
from src.utils.hurricane_geom import get_nicaragua_boundary
import os


class SchoolVulnerabilityLayer(VulnerabilityLayer):
    def __init__(self, config, cache_dir=None):
        super().__init__(config)
        self.grid_gdf = None
        self._school_grid = None
        self.cache_dir = cache_dir or get_config_value(
            config,
            "impact_analysis.output.cache_directory",
            "data/results/impact_analysis/cache/",
        )
        os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_path(self):
        return os.path.join(self.cache_dir, "school_vulnerability.gpkg")

    def compute_grid(self):
        if self.grid_gdf is not None:
            return self.grid_gdf
        cache_path = self._cache_path()
        if os.path.exists(cache_path):
            print(f"Loading cached school vulnerability layer: {cache_path}")
            self.grid_gdf = gpd.read_file(cache_path)
            self._school_grid = self.grid_gdf["school_count"].values
            return self.grid_gdf
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
        grid_gdf = gpd.GeoDataFrame(grid_cells, columns=["geometry"], crs="EPSG:4326")
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
            print("Unique school_count values:", np.unique(grid_gdf["school_count"]))
            print("school_count dtype:", grid_gdf["school_count"].dtype)
        else:
            grid_gdf["school_count"] = 0
        self.grid_gdf = grid_gdf
        self._school_grid = grid_gdf["school_count"].values
        # Save to cache
        grid_gdf.to_file(cache_path, driver="GPKG")
        print(f"Saved school vulnerability layer to cache: {cache_path}")
        return grid_gdf

    def plot(self, ax=None, output_dir="data/results/impact_analysis/"):
        grid_gdf = self.compute_grid()
        nicaragua_gdf = get_nicaragua_boundary()
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 10))
        log_counts = [np.log10(count + 1) for count in grid_gdf["school_count"]]
        grid_gdf["log_school_count"] = log_counts
        grid_gdf.plot(
            ax=ax,
            column="log_school_count",
            cmap="Blues",
            linewidth=0.1,
            edgecolor="grey",
            alpha=0.7,
            legend=True,
            legend_kwds={"label": "Log10(Schools + 1) per Cell"},
        )
        nicaragua_gdf.plot(
            ax=ax, color="none", edgecolor="black", linewidth=3, alpha=1.0
        )
        ax.set_title("School Vulnerability Heatmap (Log Scale)")
        plt.tight_layout()
        # Save to disk
        out_path = os.path.join(output_dir, "school_vulnerability.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved school vulnerability plot: {out_path}")
        plt.close(fig)

    @property
    def value_column(self):
        return "school_count"


class SchoolPopulationVulnerabilityLayer(VulnerabilityLayer):
    def __init__(self, config, cache_dir=None):
        super().__init__(config)
        self.grid_gdf = None
        self._people_grid = None
        self.cache_dir = cache_dir or get_config_value(
            config,
            "impact_analysis.output.cache_directory",
            "data/results/impact_analysis/cache/",
        )
        os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_path(self):
        return os.path.join(self.cache_dir, "school_population_vulnerability.gpkg")

    def compute_grid(self):
        if self.grid_gdf is not None:
            return self.grid_gdf
        cache_path = self._cache_path()
        if os.path.exists(cache_path):
            print(f"Loading cached school population vulnerability layer: {cache_path}")
            self.grid_gdf = gpd.read_file(cache_path)
            self._people_grid = self.grid_gdf["people_count"].values
            return self.grid_gdf
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
        grid_gdf = gpd.GeoDataFrame(grid_cells, columns=["geometry"], crs="EPSG:4326")
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
            print("Unique people_count values:", np.unique(grid_gdf["people_count"]))
            print("people_count dtype:", grid_gdf["people_count"].dtype)
        else:
            grid_gdf["people_count"] = 0
        self.grid_gdf = grid_gdf
        self._people_grid = grid_gdf["people_count"].values
        # Save to cache
        grid_gdf.to_file(cache_path, driver="GPKG")
        print(f"Saved school population vulnerability layer to cache: {cache_path}")
        return grid_gdf

    def plot(self, ax=None, output_dir="data/results/impact_analysis/"):
        grid_gdf = self.compute_grid()
        nicaragua_gdf = get_nicaragua_boundary()
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 10))
        log_counts = [np.log10(count + 1) for count in grid_gdf["people_count"]]
        grid_gdf["log_people_count"] = log_counts
        grid_gdf.plot(
            ax=ax,
            column="log_people_count",
            cmap="Oranges",
            linewidth=0.1,
            edgecolor="grey",
            alpha=0.7,
            legend=True,
            legend_kwds={"label": "Log10(People + 1) per Cell"},
        )
        nicaragua_gdf.plot(
            ax=ax, color="none", edgecolor="black", linewidth=3, alpha=1.0
        )
        ax.set_title("School Population Vulnerability Heatmap (Log Scale)")
        plt.tight_layout()
        out_path = os.path.join(output_dir, "school_population_vulnerability.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved school population vulnerability plot: {out_path}")
        plt.close(fig)

    @property
    def value_column(self):
        return "people_count"
