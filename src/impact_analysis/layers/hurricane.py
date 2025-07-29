import os
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from typing import Dict, Any
from shapely.geometry import box, LineString
from src.impact_analysis.layers.base import ExposureLayer
from src.utils.hurricane_geom import (
    get_nicaragua_boundary,
    wind_quadrant_polygon,
    compute_smoothed_wind_region,
)
from src.utils.config_utils import get_config_value


class HurricaneExposureLayer(ExposureLayer):
    def __init__(self, hurricane_df, chosen_forecast, config, cache_dir=None):
        super().__init__(config)
        self.hurricane_df = hurricane_df
        self.chosen_forecast = chosen_forecast
        self.grid_gdf = None
        self._prob_grid = None
        self._member_regions = None
        self._member_ids = None
        self._n_ensemble = None
        self._compute_member_regions()
        self.cache_dir = cache_dir or get_config_value(
            config,
            "impact_analysis.output.cache_directory",
            "data/results/impact_analysis/cache/",
        )
        os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_path(self):
        date_str = str(self.chosen_forecast).replace(":", "-").replace(" ", "_")
        wind_threshold = get_config_value(self.config, "impact_analysis.hurricane.wind_threshold.speed_knots", 50)
        return os.path.join(self.cache_dir, f"hurricane_exposure_{wind_threshold}kt_{date_str}.gpkg")

    def _compute_member_regions(self):
        df_ens = self.hurricane_df[
            self.hurricane_df["forecast_time"] == self.chosen_forecast
        ].copy()
        smoothing_method = get_config_value(
            self.config, "impact_analysis.wind_region.smoothing_method", "bspline"
        )
        smoothing_params = get_config_value(
            self.config,
            "impact_analysis.wind_region.smoothing_params",
            {"smoothing_factor": 0, "num_points": 200},
        )
        member_regions = []
        member_ids = []
        for member in df_ens["ensemble_member"].unique():
            member_data = df_ens[df_ens["ensemble_member"] == member].sort_values(
                "valid_time"
            )
            wind_polys = []
            for _, row in member_data.iterrows():
                lat = row["latitude"]
                lon = row["longitude"]
                # Get configurable wind threshold
                wind_suffix = get_config_value(self.config, "impact_analysis.hurricane.wind_threshold.column_suffix", "50_knot")
                r_ne = row.get(f"radius_{wind_suffix}_winds_ne_km", 0) or 0
                r_se = row.get(f"radius_{wind_suffix}_winds_se_km", 0) or 0
                r_sw = row.get(f"radius_{wind_suffix}_winds_sw_km", 0) or 0
                r_nw = row.get(f"radius_{wind_suffix}_winds_nw_km", 0) or 0
                if any([r_ne, r_se, r_sw, r_nw]):
                    poly = wind_quadrant_polygon(lat, lon, r_ne, r_se, r_sw, r_nw)
                    if poly is not None and poly.is_valid and not poly.is_empty:
                        wind_polys.append(poly)
            continuous_shape = compute_smoothed_wind_region(
                wind_polys, smoothing=smoothing_method, **smoothing_params
            )
            if continuous_shape is not None:
                member_regions.append(continuous_shape)
                member_ids.append(member)
        self._member_regions = member_regions
        self._member_ids = member_ids
        self._n_ensemble = len(member_regions)

    def get_member_ids(self):
        return self._member_ids

    def compute_grid(self):
        if self.grid_gdf is not None:
            return self.grid_gdf
        cache_path = self._cache_path()
        if os.path.exists(cache_path):
            print(f"Loading cached hurricane exposure layer: {cache_path}")
            self.grid_gdf = gpd.read_file(cache_path)
            self._prob_grid = self.grid_gdf["probability"].values
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
        # For each grid cell, count how many member wind regions intersect it
        region_counts = []
        for cell in grid_gdf.geometry:
            count = sum(
                region is not None
                and region.is_valid
                and not region.is_empty
                and region.intersects(cell)
                for region in self._member_regions
            )
            region_counts.append(count)
        grid_gdf["wind_region_count"] = region_counts
        # Normalize to probability
        n_ensemble = self._n_ensemble if self._n_ensemble else 1
        grid_gdf["probability"] = grid_gdf["wind_region_count"] / n_ensemble
        self.grid_gdf = grid_gdf
        self._prob_grid = grid_gdf["probability"].values
        # Save to cache
        grid_gdf.to_file(cache_path, driver="GPKG")
        print(f"Saved hurricane exposure layer to cache: {cache_path}")
        return grid_gdf

    def get_plot_metadata(self) -> Dict[str, Any]:
        """Return metadata for plotting this hurricane exposure layer."""
        return {
            "layer_type": "exposure",
            "hazard_type": "Hurricane",
            "data_column": "probability",
            "colormap": "YlOrRd",
            "title_template": "Probability of Forecasted Hurricane",
            "legend_template": "Hurricane Probability per Cell",
            "filename_template": "hurricane_exposure_{parameters}",
            "special_features": ["forecast_time"]
        }

    def plot(self, ax=None, output_dir="data/results/impact_analysis/"):
        """Plot the hurricane exposure layer using universal plotting function."""
        from src.impact_analysis.utils.plotting_utils import plot_layer_with_scales
        
        # Use universal plotting function
        plot_layer_with_scales(self, output_dir=output_dir)

    def get_grid_cells(self):
        """Return the grid GeoDataFrame (geometry only, no data columns)."""
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

    def get_member_regions(self):
        """Return the list of wind region polygons, one per ensemble member."""
        return self._member_regions
