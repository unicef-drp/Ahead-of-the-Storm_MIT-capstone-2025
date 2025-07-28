import os
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from shapely.geometry import box
from src.impact_analysis.layers.base import ExposureLayer
from src.utils.config_utils import get_config_value
from src.utils.hurricane_geom import get_nicaragua_boundary


class FloodExposureLayer(ExposureLayer):
    """
    PLACEHOLDER IMPLEMENTATION: Flood exposure layer with artificial ensemble generation.

    TODO: REPLACE ENSEMBLE LOGIC WITH REAL FORECAST DATA
    ====================================================
    This class currently uses a placeholder ensemble generation method that creates
    artificial variations from a single flood map. In production, this should:

    1. Load real ensemble flood forecasts (50 different model predictions)
    2. Process each ensemble member independently
    3. Use proper uncertainty quantification from the forecast models
    4. Apply ensemble statistics and probability calculations

    Key methods to replace:
    - _create_ensemble_variations(): Currently creates artificial variations
    - _create_deterministic_ensemble(): Currently duplicates the same map

    The real implementation should load actual ensemble forecast files and process
    them according to the specific forecast model's output format.
    """

    def __init__(
        self,
        flood_raster_path,
        config,
        cache_dir=None,
        threshold_m=1,
        n_ensemble=50,
        min_flooded_pixels_percent=10,
        resampling_method="mean",
    ):
        super().__init__(config)
        self.flood_raster_path = flood_raster_path
        self.threshold_m = threshold_m
        self.n_ensemble = n_ensemble
        self.min_flooded_pixels_percent = min_flooded_pixels_percent
        self.resampling_method = resampling_method
        self.grid_gdf = None
        self._prob_grid = None
        self._member_regions = None
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

    def _create_deterministic_ensemble(self, base_flooded_cells):
        """
        DETERMINISTIC VERSION: Creates 50 identical ensemble members
        based on the original flood raster (all members are the same).

        This replicates the original behavior where all ensemble members
        were identical to the base flood raster.
        """
        print(
            "[FloodExposureLayer] Creating deterministic ensemble (all members identical)..."
        )

        # Create 50 identical ensemble members
        ensemble_members = []
        for member_id in range(self.n_ensemble):
            member_flooded = base_flooded_cells.copy()
            ensemble_members.append(member_flooded)

        print(
            f"[FloodExposureLayer] Created {len(ensemble_members)} identical ensemble members"
        )
        return ensemble_members

    def _create_ensemble_variations(self, base_flooded_cells, grid_gdf):
        """
        PLACEHOLDER IMPLEMENTATION: Creates artificial ensemble variations from a single flood map.

        TODO: REPLACE THIS ENTIRE METHOD WITH REAL ENSEMBLE LOGIC
        ============================================================
        This is a placeholder that artificially creates 50 ensemble members by introducing
        random variations to a single flood map. In the real implementation, this should:

        1. Load actual ensemble forecast data (50 different flood predictions)
        2. Process each ensemble member independently
        3. Use real uncertainty quantification from the forecast model
        4. Apply proper ensemble statistics and probability calculations

        Current placeholder behavior (configurable via flood_config.yaml):
        - Removes originally flooded cells with configurable probability (default: 30%)
        - Adds nearby cells with distance-based probability decay (configurable)
        - Ensures spatial coherence and Nicaragua boundary compliance

        Configuration parameters (in flood_config.yaml):
        - removal_probability: Probability of removing originally flooded cells
        - max_spatial_distance: Maximum distance for adding nearby cells
        - base_addition_probability: Base probability for adding cells at distance 1
        - probability_decay_factor: Decay factor for distance-based probabilities
        """
        print("[FloodExposureLayer] WARNING: Using PLACEHOLDER ensemble generation!")
        print("[FloodExposureLayer] TODO: Replace with real ensemble forecast data")

        ensemble_members = []

        # Get Nicaragua boundary for spatial filtering
        nicaragua_gdf = get_nicaragua_boundary()
        if grid_gdf.crs != nicaragua_gdf.crs:
            nicaragua_gdf = nicaragua_gdf.to_crs(grid_gdf.crs)

        # Create mask for cells inside Nicaragua
        nicaragua_mask = np.zeros(len(grid_gdf), dtype=bool)
        for i, cell_geom in enumerate(grid_gdf.geometry):
            nicaragua_mask[i] = nicaragua_gdf.intersects(cell_geom).any()

        # PLACEHOLDER: Generate 50 artificial ensemble members
        for member_id in range(50):
            # Set random seed for reproducibility per member
            np.random.seed(member_id)

            # Start with the base flooded cells
            member_flooded = base_flooded_cells.copy()
            original_flooded_indices = np.where(base_flooded_cells == 1)[0]

            # PLACEHOLDER VARIATION 1: Randomly remove some originally flooded cells
            if len(original_flooded_indices) > 0:
                # Get removal probability from config
                removal_prob = get_config_value(
                    self.config, "flood.ensemble_generation.removal_probability", 0.3
                )
                remove_count = max(1, int(len(original_flooded_indices) * removal_prob))
                remove_indices = np.random.choice(
                    original_flooded_indices,
                    size=min(remove_count, len(original_flooded_indices)),
                    replace=False,
                )
                member_flooded[remove_indices] = 0

            # PLACEHOLDER VARIATION 2: Add neighboring cells with distance-based probability decay
            if len(original_flooded_indices) > 0:
                # Get ensemble generation parameters from config
                max_spatial_distance = get_config_value(
                    self.config, "flood.ensemble_generation.max_spatial_distance", 4
                )
                base_prob = get_config_value(
                    self.config,
                    "flood.ensemble_generation.base_addition_probability",
                    0.2,
                )
                decay_factor = get_config_value(
                    self.config,
                    "flood.ensemble_generation.probability_decay_factor",
                    0.5,
                )

                # Create a mask for cells that are close to originally flooded cells
                nearby_mask = np.zeros_like(base_flooded_cells, dtype=bool)
                distance_probabilities = np.zeros_like(base_flooded_cells, dtype=float)

                for flooded_idx in original_flooded_indices:
                    # Get the geometry of the flooded cell
                    flooded_cell_geom = grid_gdf.iloc[flooded_idx].geometry

                    # Find cells that are spatially close to this flooded cell
                    for i, cell_geom in enumerate(grid_gdf.geometry):
                        # Calculate distance between cell centers
                        distance = flooded_cell_geom.centroid.distance(
                            cell_geom.centroid
                        )

                        # Convert distance to grid units (approximate)
                        # Assuming grid resolution is 0.1 degrees
                        grid_res_degrees = get_config_value(
                            self.config, "impact_analysis.grid.resolution_degrees", 0.1
                        )

                        # Calculate how many grid cells away this is
                        grid_distance = distance / (
                            grid_res_degrees * 111000
                        )  # Convert to meters

                        # Only consider cells within max_spatial_distance grid cells
                        if 0 < grid_distance <= max_spatial_distance:
                            nearby_mask[i] = True

                            # Distance-based probability decay (exponential)
                            prob = base_prob * (
                                decay_factor ** (int(grid_distance) - 1)
                            )

                            # Take the maximum probability if this cell is affected by multiple flooded cells
                            distance_probabilities[i] = max(
                                distance_probabilities[i], prob
                            )

                # Only consider nearby cells that aren't already flooded AND are inside Nicaragua
                nearby_non_flooded = np.where(
                    nearby_mask & (member_flooded == 0) & nicaragua_mask
                )[0]

                if len(nearby_non_flooded) > 0:
                    # Apply distance-based probabilities to add cells
                    for cell_idx in nearby_non_flooded:
                        prob = distance_probabilities[cell_idx]
                        if np.random.random() < prob:
                            member_flooded[cell_idx] = 1

            # CRITICAL: Ensure cells outside Nicaragua are always zero
            member_flooded = member_flooded & nicaragua_mask

            ensemble_members.append(member_flooded)

        print(
            f"[FloodExposureLayer] Generated {len(ensemble_members)} placeholder ensemble members"
        )
        return ensemble_members

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

        # Create ensemble variations (PLACEHOLDER implementation)
        base_flooded_cells = np.array(flooded_counts)
        if self.resampling_method == "deterministic":
            ensemble_members = self._create_deterministic_ensemble(base_flooded_cells)
        else:
            ensemble_members = self._create_ensemble_variations(
                base_flooded_cells, grid_gdf
            )

        # Calculate probability based on ensemble members
        ensemble_array = np.array(ensemble_members)
        flood_count = ensemble_array.sum(axis=0)
        probability = flood_count / self.n_ensemble

        grid_gdf["flood_count"] = flood_count
        grid_gdf["probability"] = probability

        # After masking and before saving/returning, reproject grid back to EPSG:4326
        if grid_gdf.crs != "EPSG:4326":
            grid_gdf = grid_gdf.to_crs("EPSG:4326")
        self.grid_gdf = grid_gdf
        self._prob_grid = grid_gdf["probability"].values
        self._ensemble_members = ensemble_members
        # grid_gdf.to_file(cache_path, driver="GPKG")
        # print(f"Saved flood exposure layer to cache: {cache_path}")
        return grid_gdf

    def get_grid_cells(self):
        grid_gdf = self.compute_grid()
        return grid_gdf[["geometry"]].copy()

    def get_member_regions(self):
        # PLACEHOLDER: Return slightly different regions for each ensemble member
        # In the real implementation, each member would have its own unique flood extent
        if not hasattr(self, "_ensemble_members"):
            self.compute_grid()

        member_regions = []
        grid_gdf = self.compute_grid()

        for member_id in range(self.n_ensemble):
            # Get flooded cells for this member
            member_flooded = self._ensemble_members[member_id]
            flooded_cells = grid_gdf[member_flooded > 0]

            if not flooded_cells.empty:
                region = flooded_cells.unary_union
            else:
                from shapely.geometry import GeometryCollection

                region = GeometryCollection()

            member_regions.append(region)

        return member_regions

    def get_member_ids(self):
        return list(range(self.n_ensemble))

    def plot_baseline(self, ax=None, output_dir=None):
        """
        Plot the baseline flood extent (original flood map before ensemble variations).
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Use the same grid as compute_grid to ensure consistency
        grid_gdf = self.compute_grid()

        # Load flood raster data using the same method as compute_grid
        flood_data, transform, crs = self._load_flood_raster()

        # Calculate baseline flooded cells using the same logic as compute_grid
        baseline_flooded_counts = []

        # Reproject grid to raster CRS if needed (same as in compute_grid)
        if grid_gdf.crs != crs:
            grid_gdf = grid_gdf.to_crs(crs)

        from rasterio.features import geometry_mask
        from rasterio import features

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

            baseline_flooded_counts.append(1 if is_flooded else 0)

        # Reproject back to EPSG:4326 (same as in compute_grid)
        if grid_gdf.crs != "EPSG:4326":
            grid_gdf = grid_gdf.to_crs("EPSG:4326")

        # Create a GeoDataFrame for plotting
        baseline_grid_gdf = grid_gdf.copy()
        baseline_grid_gdf["flooded"] = baseline_flooded_counts

        # Plot the baseline flood extent
        baseline_grid_gdf.plot(
            column="flooded",
            ax=ax,
            cmap="Reds",
            legend=True,
            legend_kwds={"label": "Flooded (Baseline)"},
        )

        # Add Nicaragua boundary for context
        nicaragua_gdf = get_nicaragua_boundary()
        if baseline_grid_gdf.crs != nicaragua_gdf.crs:
            nicaragua_gdf = nicaragua_gdf.to_crs(baseline_grid_gdf.crs)

        nicaragua_gdf.plot(
            ax=ax, color="none", edgecolor="black", linewidth=2, alpha=0.8
        )

        ax.set_title("Baseline Flood Extent (Original)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        if output_dir:
            plt.savefig(
                os.path.join(output_dir, "baseline_flood_extent.png"),
                dpi=300,
                bbox_inches="tight",
            )

        return ax

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
        ax.set_title("Flood Exposure Probability Heatmap (Ensemble)")
        plt.tight_layout()
        base = os.path.splitext(os.path.basename(self.flood_raster_path))[0]
        out_path = os.path.join(output_dir, f"flood_exposure_{base}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved flood exposure plot: {out_path}")
        plt.close(fig)
