import os
import numpy as np
import geopandas as gpd
import rasterio
from shapely.geometry import box
from src.impact_analysis.layers.base import ExposureLayer
from src.utils.hurricane_geom import get_nicaragua_boundary
from src.utils.config_utils import get_config_value


class FloodExposureLayer(ExposureLayer):
    def __init__(
        self,
        flood_raster_path,
        config,
        cache_dir=None,
        threshold_m=1,
        n_ensemble=50,
        min_flooded_pixels_percent=10,
    ):
        super().__init__(config)
        self.flood_raster_path = flood_raster_path
        self.threshold_m = threshold_m
        self.n_ensemble = n_ensemble
        self.min_flooded_pixels_percent = min_flooded_pixels_percent
        self.grid_gdf = None
        self._prob_grid = None
        self.cache_dir = cache_dir or get_config_value(
            config,
            "impact_analysis.output.cache_directory",
            "data/results/impact_analysis/cache/",
        )
        os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_path(self):
        base = os.path.splitext(os.path.basename(self.flood_raster_path))[0]
        return os.path.join(self.cache_dir, f"flood_exposure_{base}.gpkg")

    def _load_flood_raster(self):
        with rasterio.open(self.flood_raster_path) as src:
            flood_data = src.read(1)
            transform = src.transform
            crs = src.crs
        print(
            f"[FloodExposureLayer] Flood raster min: {flood_data.min()}, max: {flood_data.max()}"
        )
        return flood_data, transform, crs

    def compute_grid(self):
        if self.grid_gdf is not None:
            return self.grid_gdf
        # cache_path = self._cache_path()
        # if os.path.exists(cache_path):
        #    print(f"Loading cached flood exposure layer: {cache_path}")
        #    self.grid_gdf = gpd.read_file(cache_path)
        #    self._prob_grid = self.grid_gdf["probability"].values
        #    return self.grid_gdf
        # Load flood raster and grid
        flood_data, transform, crs = self._load_flood_raster()
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
        print(f"[FloodExposureLayer] Raster CRS: {crs}, Grid CRS: {grid_gdf.crs}")
        # Reproject grid to raster CRS if needed
        if grid_gdf.crs != crs:
            grid_gdf = grid_gdf.to_crs(crs)
        # Rasterize: for each cell, check if any pixel in the cell > threshold
        from rasterio.features import geometry_mask
        from rasterio import features

        flooded_counts = []
        flooded_cells_count = 0
        for cell in grid_gdf.geometry:
            mask = features.geometry_mask(
                [cell],
                out_shape=flood_data.shape,
                transform=transform,
                invert=True,
            )
            # Count total pixels in this cell and how many are flooded
            total_pixels_in_cell = mask.sum()
            flooded_pixels = (flood_data[mask] > self.threshold_m).sum()

            # Calculate percentage of flooded pixels
            if total_pixels_in_cell > 0:
                flooded_percent = (flooded_pixels / total_pixels_in_cell) * 100
                is_flooded = flooded_percent >= self.min_flooded_pixels_percent
            else:
                is_flooded = False

            flooded_counts.append(1 if is_flooded else 0)
            if is_flooded:
                flooded_cells_count += 1
        print(
            f"[FloodExposureLayer] Number of grid cells with >= {self.min_flooded_pixels_percent}% flooded pixels above {self.threshold_m}m threshold: {flooded_cells_count} / {len(grid_gdf)}"
        )
        # Replicate for 50 ensemble members (all the same for now)
        grid_gdf["flood_count"] = np.array(flooded_counts) * self.n_ensemble
        grid_gdf["probability"] = grid_gdf["flood_count"] / self.n_ensemble
        # After masking and before saving/returning, reproject grid back to EPSG:4326
        if grid_gdf.crs != "EPSG:4326":
            grid_gdf = grid_gdf.to_crs("EPSG:4326")
        self.grid_gdf = grid_gdf
        self._prob_grid = grid_gdf["probability"].values
        # grid_gdf.to_file(cache_path, driver="GPKG")
        # print(f"Saved flood exposure layer to cache: {cache_path}")
        return grid_gdf

    def get_grid_cells(self):
        grid_gdf = self.compute_grid()
        return grid_gdf[["geometry"]].copy()

    def get_member_regions(self):
        # For flood, return a list of the union of flooded grid cells, repeated 50 times
        grid_gdf = self.compute_grid()
        flooded_cells = grid_gdf[grid_gdf["probability"] > 0]
        if not flooded_cells.empty:
            region = flooded_cells.unary_union
        else:
            from shapely.geometry import GeometryCollection

            region = GeometryCollection()
        return [region] * 50

    def get_member_ids(self):
        return list(range(50))

    def plot(self, ax=None, output_dir="data/results/impact_analysis/"):
        grid_gdf = self.compute_grid()
        nicaragua_gdf = get_nicaragua_boundary()
        import matplotlib.pyplot as plt

        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 10))
        grid_gdf.plot(
            ax=ax,
            column="probability",
            cmap="Blues",
            linewidth=0.1,
            edgecolor="grey",
            alpha=0.7,
            legend=True,
            legend_kwds={"label": "Flood Probability per Cell"},
        )
        nicaragua_gdf.plot(
            ax=ax, color="none", edgecolor="black", linewidth=3, alpha=1.0
        )
        # Set axis limits to Nicaragua bounding box
        minx, miny, maxx, maxy = nicaragua_gdf.total_bounds
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_title("Flood Exposure Probability Heatmap")
        plt.tight_layout()
        base = os.path.splitext(os.path.basename(self.flood_raster_path))[0]
        out_path = os.path.join(output_dir, f"flood_exposure_{base}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved flood exposure plot: {out_path}")
        plt.close(fig)
