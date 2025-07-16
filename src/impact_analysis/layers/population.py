import os
import numpy as np
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from shapely.geometry import box
from src.impact_analysis.layers.base import VulnerabilityLayer
from src.utils.config_utils import get_config_value
from src.utils.path_utils import get_data_path
from src.utils.hurricane_geom import get_nicaragua_boundary


class PopulationVulnerabilityLayer(VulnerabilityLayer):
    def __init__(self, config, age_groups=None, gender="both", cache_dir=None):
        super().__init__(config)
        self.age_groups = (
            age_groups if age_groups is not None else list(range(0, 85, 5))
        )
        self.gender = gender
        self.grid_gdf = None
        self._population_grid = None
        self.cache_dir = cache_dir or get_config_value(
            config,
            "impact_analysis.output.cache_directory",
            "data/results/impact_analysis/cache/",
        )
        os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_path(self):
        age_str = "_".join(map(str, self.age_groups))
        return os.path.join(self.cache_dir, f"population_{self.gender}_{age_str}.gpkg")

    def compute_grid(self):
        if self.grid_gdf is not None:
            return self.grid_gdf
        cache_path = self._cache_path()
        if os.path.exists(cache_path):
            print(f"Loading cached population vulnerability layer: {cache_path}")
            self.grid_gdf = gpd.read_file(cache_path)
            self._population_grid = self.grid_gdf["population_count"].values
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
        # Load and sum raster data for selected ages/gender
        base_path = get_data_path("data/raw/census")
        files_to_load = []
        for age in self.age_groups:
            if self.gender in ["female", "both"]:
                ffile = base_path / f"nic_f_{age}_2020_constrained_UNadj.tif"
                if ffile.exists():
                    files_to_load.append(str(ffile))
            if self.gender in ["male", "both"]:
                mfile = base_path / f"nic_m_{age}_2020_constrained_UNadj.tif"
                if mfile.exists():
                    files_to_load.append(str(mfile))
        if not files_to_load:
            raise FileNotFoundError(
                "No population raster files found for the specified age groups and gender!"
            )
        # Use the first raster as reference for transform
        with rasterio.open(files_to_load[0]) as src:
            ref_transform = src.transform
            ref_crs = src.crs
            ref_shape = src.read(1).shape
        # Sum all rasters
        combined = np.zeros(ref_shape, dtype=np.float32)
        for fpath in files_to_load:
            with rasterio.open(fpath) as src:
                data = src.read(1)
                # Mask out NoData values (assume -99999 or less is NoData)
                data = np.where(data <= -99999, 0, data)
                combined += data
        # Rasterize to grid cells
        from rasterio.features import geometry_mask
        from rasterio import features

        population_counts = []
        for cell in grid_gdf.geometry:
            # Mask for the cell
            mask = features.geometry_mask(
                [cell], out_shape=combined.shape, transform=ref_transform, invert=True
            )
            population = combined[mask].sum()
            population_counts.append(population)
        grid_gdf["population_count"] = population_counts
        self.grid_gdf = grid_gdf
        self._population_grid = grid_gdf["population_count"].values
        grid_gdf.to_file(cache_path, driver="GPKG")
        print(f"Saved population vulnerability layer to cache: {cache_path}")
        return grid_gdf

    def plot(self, ax=None, output_dir="data/results/impact_analysis/"):
        grid_gdf = self.compute_grid()
        nicaragua_gdf = get_nicaragua_boundary()
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 10))
        log_counts = [np.log10(count + 1) for count in grid_gdf["population_count"]]
        grid_gdf["log_population_count"] = log_counts
        grid_gdf.plot(
            ax=ax,
            column="log_population_count",
            cmap="Purples",
            linewidth=0.1,
            edgecolor="grey",
            alpha=0.7,
            legend=True,
            legend_kwds={"label": "Log10(Population + 1) per Cell"},
        )
        nicaragua_gdf.plot(
            ax=ax, color="none", edgecolor="black", linewidth=3, alpha=1.0
        )
        ax.set_title(
            f"Population Vulnerability Heatmap (Log Scale)\nGender: {self.gender}, Ages: {self.age_groups}"
        )
        plt.tight_layout()
        age_str = "_".join(map(str, self.age_groups))
        out_path = os.path.join(
            output_dir, f"population_vulnerability_{self.gender}_ages_{age_str}.png"
        )
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved population vulnerability plot: {out_path}")
        plt.close(fig)

    @property
    def value_column(self):
        return "population_count"
