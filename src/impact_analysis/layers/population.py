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
    def __init__(self, config, age_groups=None, gender="both", cache_dir=None, resolution_context=None):
        super().__init__(config, resolution_context)
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
        resolution = self.get_resolution()
        if self.resolution_context:
            # Use parquet for high-res computation, gpkg for visualization
            if self.resolution_context == "landslide_computation":
                return os.path.join(self.cache_dir, f"population_{self.gender}_{age_str}_{self.resolution_context}_{resolution}deg.parquet")
            else:
                return os.path.join(self.cache_dir, f"population_{self.gender}_{age_str}_{self.resolution_context}_{resolution}deg.gpkg")
        else:
            return os.path.join(self.cache_dir, f"population_{self.gender}_{age_str}.gpkg")

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
                    [cell],
                    out_shape=combined.shape,
                    transform=ref_transform,
                    invert=True,
                )
                population = combined[mask].sum()
                population_counts.append(population)
            grid_gdf["population_count"] = population_counts
            return grid_gdf

        self.grid_gdf = self._load_or_compute_grid(
            cache_path, "population_count", compute_func
        )
        self._population_grid = self.grid_gdf["population_count"].values
        return self.grid_gdf

    def plot(self, ax=None, output_dir="data/results/impact_analysis/"):
        grid_gdf = self.compute_grid()
        age_str = "_".join(map(str, self.age_groups))
        output_filename = f"population_vulnerability_{self.gender}_ages_{age_str}.png"
        plot_title = f"Population Vulnerability Heatmap (Log Scale)\nGender: {self.gender}, Ages: {self.age_groups}"
        self._plot_vulnerability_grid(
            grid_gdf,
            value_column="population_count",
            cmap="Purples",
            legend_label="Log10(Population + 1) per Cell",
            output_dir=output_dir,
            output_filename=output_filename,
            plot_title=plot_title,
            ax=ax,
        )

    @property
    def value_column(self):
        return "population_count"
