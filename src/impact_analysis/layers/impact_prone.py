import os
import numpy as np
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from typing import Dict, Any
from shapely.geometry import box
from src.impact_analysis.layers.base import VulnerabilityLayer
from src.utils.config_utils import get_config_value
from src.utils.path_utils import get_data_path
from src.utils.hurricane_geom import get_nicaragua_boundary


class ImpactProneVulnerabilityLayer(VulnerabilityLayer):
    """
    Base class for impact-prone vulnerability layers.
    Identifies infrastructure in areas with high population/nightlights ratio.
    """

    def __init__(
        self, config, infrastructure_type, cache_dir=None, resolution_context=None, use_cache=True
    ):
        super().__init__(config, resolution_context)
        self.infrastructure_type = infrastructure_type
        self.grid_gdf = None
        self.use_cache = use_cache
        self.cache_dir = cache_dir or get_config_value(
            config,
            "impact_analysis.output.cache_directory",
            "data/results/impact_analysis/cache/",
        )
        os.makedirs(self.cache_dir, exist_ok=True)

        # Configuration for vulnerability index calculation
        self.epsilon = get_config_value(
            config, "impact_analysis.impact_prone.epsilon", 1.0
        )
        self.threshold_percentile = get_config_value(
            config,
            "impact_analysis.impact_prone.threshold_percentile",
            75.0,  # Consider top 25% as impact-prone
        )

    def _cache_path(self):
        resolution = self.get_resolution()
        if self.resolution_context:
            if self.resolution_context == "landslide_computation":
                return os.path.join(
                    self.cache_dir,
                    f"{self.infrastructure_type}_impact_prone_{self.resolution_context}_{resolution}deg.parquet",
                )
            else:
                return os.path.join(
                    self.cache_dir,
                    f"{self.infrastructure_type}_impact_prone_{self.resolution_context}_{resolution}deg.gpkg",
                )
        else:
            return os.path.join(
                self.cache_dir, f"{self.infrastructure_type}_impact_prone.gpkg"
            )

    def _compute_vulnerability_index(self, population_data, nightlights_data):
        """
        Compute vulnerability index: population / (nightlights + epsilon)
        """
        # Ensure both arrays have the same shape
        if population_data.shape != nightlights_data.shape:
            raise ValueError(
                f"Population and nightlights data have different shapes: {population_data.shape} vs {nightlights_data.shape}"
            )

        # Compute vulnerability index
        vulnerability_index = population_data / (nightlights_data + self.epsilon)

        # Handle division by zero and invalid values
        vulnerability_index = np.where(
            np.isnan(vulnerability_index) | np.isinf(vulnerability_index),
            0,
            vulnerability_index,
        )

        return vulnerability_index

    def _identify_impact_prone_cells(self, vulnerability_index):
        """
        Identify cells that are impact-prone based on threshold.
        """
        # Calculate threshold based on percentile
        threshold = np.percentile(
            vulnerability_index[vulnerability_index > 0], self.threshold_percentile
        )

        # Create binary mask for impact-prone cells
        impact_prone_mask = vulnerability_index >= threshold

        return impact_prone_mask, threshold

    def _load_nightlights_data(self):
        """
        Load and aggregate nightlights data to match grid resolution.
        """
        nightlights_path = get_data_path(
            "data/preprocessed/nightlights/processed/nightlights_nicaragua_average.tif"
        )

        if not os.path.exists(nightlights_path):
            raise FileNotFoundError(f"Nightlights data not found: {nightlights_path}")

        with rasterio.open(nightlights_path) as src:
            nightlights_data = src.read(1)
            transform = src.transform
            crs = src.crs

        return nightlights_data, transform, crs

    def _load_population_data(self):
        """
        Load population data for the same grid cells.
        """
        # Use the same approach as PopulationVulnerabilityLayer
        base_path = get_data_path("data/raw/census")

        # Load all age groups for total population
        age_groups = list(range(0, 85, 5))
        files_to_load = []

        for age in age_groups:
            ffile = base_path / f"nic_f_{age}_2020_constrained_UNadj.tif"
            mfile = base_path / f"nic_m_{age}_2020_constrained_UNadj.tif"

            if ffile.exists():
                files_to_load.append(str(ffile))
            if mfile.exists():
                files_to_load.append(str(mfile))

        if not files_to_load:
            raise FileNotFoundError("No population raster files found!")

        # Sum all population rasters
        with rasterio.open(files_to_load[0]) as src:
            ref_transform = src.transform
            ref_crs = src.crs
            ref_shape = src.read(1).shape

        combined = np.zeros(ref_shape, dtype=np.float32)
        for fpath in files_to_load:
            with rasterio.open(fpath) as src:
                data = src.read(1)
                data = np.where(data <= -99999, 0, data)
                combined += data

        return combined, ref_transform, ref_crs

    def _aggregate_to_grid(self, raster_data, transform, grid_bounds, grid_res):
        """
        Aggregate raster data to grid cells.
        """
        from rasterio.features import geometry_mask

        minx, miny, maxx, maxy = grid_bounds
        grid_cells = []
        x_coords = np.arange(minx, maxx, grid_res)
        y_coords = np.arange(miny, maxy, grid_res)

        for x in x_coords:
            for y in y_coords:
                grid_cells.append(box(x, y, x + grid_res, y + grid_res))

        aggregated_values = []
        for cell in grid_cells:
            mask = geometry_mask(
                [cell],
                out_shape=raster_data.shape,
                transform=transform,
                invert=True,
            )
            value = raster_data[mask].sum()
            aggregated_values.append(value)

        return aggregated_values

    def compute_grid(self):
        if self.grid_gdf is not None:
            return self.grid_gdf
        cache_path = self._cache_path()

        def compute_func():
            # Load preprocessed grid with population and nightlights data
            preprocessed_path = get_data_path(
                "data/preprocessed/nightlights/processed/nightlights_population_grid.gpkg"
            )

            if not os.path.exists(preprocessed_path):
                raise FileNotFoundError(
                    f"Preprocessed grid not found: {preprocessed_path}\n"
                    "Please run src/data_prep/create_nightlights_vulnerability_grid.py first."
                )

            print(f"Loading preprocessed grid: {preprocessed_path}")
            grid_gdf = gpd.read_file(preprocessed_path)

            # Load infrastructure data and filter to impact-prone areas
            infrastructure_count = self._compute_infrastructure_in_impact_prone_areas(
                grid_gdf, grid_gdf["impact_prone"]
            )

            # Add infrastructure count column
            grid_gdf[f"{self.infrastructure_type}_count"] = infrastructure_count

            print(
                f"Infrastructure in impact-prone areas: {np.sum(infrastructure_count)}"
            )

            return grid_gdf

        self.grid_gdf = self._load_or_compute_grid(
            cache_path, f"{self.infrastructure_type}_count", compute_func, use_cache=self.use_cache
        )
        return self.grid_gdf

    def _compute_infrastructure_in_impact_prone_areas(
        self, grid_gdf, impact_prone_mask
    ):
        """
        Compute infrastructure count and apply impact-prone filter.
        """
        # Load infrastructure data
        if self.infrastructure_type == "school":
            data_path = get_config_value(
                self.config,
                "impact_analysis.input.school_data",
                "data/raw/osm/schools.geojson",
            )
        elif self.infrastructure_type == "health_facility":
            data_path = get_config_value(
                self.config,
                "impact_analysis.input.health_facility_data",
                "data/raw/sinapred/health_facilities_nic.geojson",
            )
        elif self.infrastructure_type == "shelter":
            data_path = get_config_value(
                self.config,
                "impact_analysis.input.shelter_data",
                "data/raw/sinapred/shelters_nic.geojson",
            )
        else:
            raise ValueError(
                f"Unsupported infrastructure type: {self.infrastructure_type}"
            )

        infrastructure_file = get_data_path(data_path)
        if not infrastructure_file.exists():
            print(
                f"Warning: {self.infrastructure_type} data not found: {infrastructure_file}"
            )
            return np.zeros(len(grid_gdf))

        infrastructure_gdf = gpd.read_file(infrastructure_file)

        # Ensure CRS compatibility
        if infrastructure_gdf.crs != grid_gdf.crs:
            print(
                f"Transforming {self.infrastructure_type} data from {infrastructure_gdf.crs} to {grid_gdf.crs}"
            )
            infrastructure_gdf = infrastructure_gdf.to_crs(grid_gdf.crs)

        # Count infrastructure in ALL cells first
        infrastructure_counts = []
        for i, cell in enumerate(grid_gdf.geometry):
            count = infrastructure_gdf.within(cell).sum()
            infrastructure_counts.append(count)

        # Convert to numpy array
        infrastructure_counts = np.array(infrastructure_counts)

        # Apply impact-prone filter: only keep infrastructure in impact-prone areas
        filtered_counts = np.where(impact_prone_mask, infrastructure_counts, 0)

        return filtered_counts

    def get_plot_metadata(self) -> Dict[str, Any]:
        """Return metadata for plotting this impact-prone vulnerability layer."""
        return {
            "layer_type": "vulnerability",
            "hazard_type": "Impact-Prone",
            "vulnerability_type": f"{self.infrastructure_type.title()} Impact-Prone",
            "data_column": self.value_column,
            "colormap": "Reds",
            "title_template": "Concentration of {vulnerability_type}",
            "legend_template": "Log10({vulnerability_type} + 1) per Cell",
            "filename_template": "{vulnerability_type}_vulnerability_{parameters}",
            "special_features": []
        }

    def plot(self, ax=None, output_dir="data/results/impact_analysis/"):
        """Plot the impact-prone vulnerability layer using universal plotting function."""
        from src.impact_analysis.utils.plotting_utils import plot_layer_with_scales
        
        # Use universal plotting function
        plot_layer_with_scales(self, output_dir=output_dir)

    @property
    def value_column(self):
        return f"{self.infrastructure_type}_count"


class SchoolImpactProneVulnerabilityLayer(ImpactProneVulnerabilityLayer):
    """Schools that are in impact-prone areas based on population/nightlights ratio."""

    def __init__(self, config, cache_dir=None, resolution_context=None, use_cache=True):
        super().__init__(config, "school", cache_dir, resolution_context, use_cache=use_cache)


class HealthFacilityImpactProneVulnerabilityLayer(ImpactProneVulnerabilityLayer):
    """Health facilities that are in impact-prone areas based on population/nightlights ratio."""

    def __init__(self, config, cache_dir=None, resolution_context=None, use_cache=True):
        super().__init__(config, "health_facility", cache_dir, resolution_context, use_cache=use_cache)


class ShelterImpactProneVulnerabilityLayer(ImpactProneVulnerabilityLayer):
    """Shelters that are in impact-prone areas based on population/nightlights ratio."""

    def __init__(self, config, cache_dir=None, resolution_context=None, use_cache=True):
        super().__init__(config, "shelter", cache_dir, resolution_context, use_cache=use_cache)


class SchoolPopulationImpactProneVulnerabilityLayer(ImpactProneVulnerabilityLayer):
    """Schools weighted by population that are in impact-prone areas based on population/nightlights ratio."""

    def __init__(self, config, cache_dir=None, resolution_context=None, use_cache=True):
        super().__init__(config, "school_population", cache_dir, resolution_context, use_cache=use_cache)

    def _compute_infrastructure_in_impact_prone_areas(
        self, grid_gdf, impact_prone_mask
    ):
        """
        Compute school population and apply impact-prone filter.
        """
        # Load school data
        school_data_path = get_config_value(
            self.config,
            "impact_analysis.input.school_data",
            "data/raw/osm/schools.geojson",
        )
        school_file = get_data_path(school_data_path)

        if not school_file.exists():
            print(f"Warning: School data not found: {school_file}")
            return np.zeros(len(grid_gdf))

        schools_gdf = gpd.read_file(school_file)

        # Ensure CRS compatibility
        if schools_gdf.crs != grid_gdf.crs:
            print(f"Transforming school data from {schools_gdf.crs} to {grid_gdf.crs}")
            schools_gdf = schools_gdf.to_crs(grid_gdf.crs)

        # Load population data for the same grid
        population_data_path = get_data_path("data/raw/census")
        age_groups = [0, 5, 10, 15]  # Children age groups
        files_to_load = []

        for age in age_groups:
            ffile = population_data_path / f"nic_f_{age}_2020_constrained_UNadj.tif"
            mfile = population_data_path / f"nic_m_{age}_2020_constrained_UNadj.tif"

            if ffile.exists():
                files_to_load.append(str(ffile))
            if mfile.exists():
                files_to_load.append(str(mfile))

        if not files_to_load:
            print("Warning: No population raster files found!")
            return np.zeros(len(grid_gdf))

        # Load and sum population data
        with rasterio.open(files_to_load[0]) as src:
            ref_transform = src.transform
            ref_crs = src.crs
            ref_shape = src.read(1).shape

        population_combined = np.zeros(ref_shape, dtype=np.float32)
        for fpath in files_to_load:
            with rasterio.open(fpath) as src:
                data = src.read(1)
                data = np.where(data <= -99999, 0, data)
                population_combined += data

        # Aggregate population to grid cells
        grid_res = self.get_resolution()
        nicaragua_gdf = get_nicaragua_boundary()
        bounds = nicaragua_gdf.total_bounds

        population_per_cell = self._aggregate_to_grid(
            population_combined, ref_transform, bounds, grid_res
        )

        # Count schools and multiply by population for each cell
        school_population_counts = []
        for i, cell in enumerate(grid_gdf.geometry):
            school_count = schools_gdf.within(cell).sum()
            cell_population = (
                population_per_cell[i] if i < len(population_per_cell) else 0
            )
            school_population_counts.append(school_count * cell_population)

        # Convert to numpy array
        school_population_counts = np.array(school_population_counts)

        # Apply impact-prone filter: only keep school population in impact-prone areas
        filtered_counts = np.where(impact_prone_mask, school_population_counts, 0)

        return filtered_counts

    @property
    def value_column(self):
        return "school_population_count"


class HealthFacilityPopulationImpactProneVulnerabilityLayer(
    ImpactProneVulnerabilityLayer
):
    """Health facilities weighted by population that are in impact-prone areas based on population/nightlights ratio."""

    def __init__(self, config, cache_dir=None, resolution_context=None, use_cache=True):
        super().__init__(
            config, "health_facility_population", cache_dir, resolution_context, use_cache=use_cache
        )

    def _compute_infrastructure_in_impact_prone_areas(
        self, grid_gdf, impact_prone_mask
    ):
        """
        Compute health facility population and apply impact-prone filter.
        """
        # Load health facility data
        health_data_path = get_config_value(
            self.config,
            "impact_analysis.input.health_facility_data",
            "data/raw/sinapred/health_facilities_nic.geojson",
        )
        health_file = get_data_path(health_data_path)

        if not health_file.exists():
            print(f"Warning: Health facility data not found: {health_file}")
            return np.zeros(len(grid_gdf))

        health_gdf = gpd.read_file(health_file)

        # Ensure CRS compatibility
        if health_gdf.crs != grid_gdf.crs:
            print(
                f"Transforming health facility data from {health_gdf.crs} to {grid_gdf.crs}"
            )
            health_gdf = health_gdf.to_crs(grid_gdf.crs)

        # Load population data for the same grid
        population_data_path = get_data_path("data/raw/census")
        age_groups = list(range(0, 85, 5))  # All age groups
        files_to_load = []

        for age in age_groups:
            ffile = population_data_path / f"nic_f_{age}_2020_constrained_UNadj.tif"
            mfile = population_data_path / f"nic_m_{age}_2020_constrained_UNadj.tif"

            if ffile.exists():
                files_to_load.append(str(ffile))
            if mfile.exists():
                files_to_load.append(str(mfile))

        if not files_to_load:
            print("Warning: No population raster files found!")
            return np.zeros(len(grid_gdf))

        # Load and sum population data
        with rasterio.open(files_to_load[0]) as src:
            ref_transform = src.transform
            ref_crs = src.crs
            ref_shape = src.read(1).shape

        population_combined = np.zeros(ref_shape, dtype=np.float32)
        for fpath in files_to_load:
            with rasterio.open(fpath) as src:
                data = src.read(1)
                data = np.where(data <= -99999, 0, data)
                population_combined += data

        # Aggregate population to grid cells
        grid_res = self.get_resolution()
        nicaragua_gdf = get_nicaragua_boundary()
        bounds = nicaragua_gdf.total_bounds

        population_per_cell = self._aggregate_to_grid(
            population_combined, ref_transform, bounds, grid_res
        )

        # Count health facilities and multiply by population for each cell
        health_population_counts = []
        for i, cell in enumerate(grid_gdf.geometry):
            health_count = health_gdf.within(cell).sum()
            cell_population = (
                population_per_cell[i] if i < len(population_per_cell) else 0
            )
            health_population_counts.append(health_count * cell_population)

        # Convert to numpy array
        health_population_counts = np.array(health_population_counts)

        # Apply impact-prone filter: only keep health facility population in impact-prone areas
        filtered_counts = np.where(impact_prone_mask, health_population_counts, 0)

        return filtered_counts

    @property
    def value_column(self):
        return "health_facility_population_count"


class ShelterPopulationImpactProneVulnerabilityLayer(ImpactProneVulnerabilityLayer):
    """Shelters weighted by population that are in impact-prone areas based on population/nightlights ratio."""

    def __init__(self, config, cache_dir=None, resolution_context=None, use_cache=True):
        super().__init__(config, "shelter_population", cache_dir, resolution_context, use_cache=use_cache)

    def _compute_infrastructure_in_impact_prone_areas(
        self, grid_gdf, impact_prone_mask
    ):
        """
        Compute shelter population and apply impact-prone filter.
        """
        # Load shelter data
        shelter_data_path = get_config_value(
            self.config,
            "impact_analysis.input.shelter_data",
            "data/raw/sinapred/shelters_nic.geojson",
        )
        shelter_file = get_data_path(shelter_data_path)

        if not shelter_file.exists():
            print(f"Warning: Shelter data not found: {shelter_file}")
            return np.zeros(len(grid_gdf))

        shelter_gdf = gpd.read_file(shelter_file)

        # Ensure CRS compatibility
        if shelter_gdf.crs != grid_gdf.crs:
            print(f"Transforming shelter data from {shelter_gdf.crs} to {grid_gdf.crs}")
            shelter_gdf = shelter_gdf.to_crs(grid_gdf.crs)

        # Load population data for the same grid
        population_data_path = get_data_path("data/raw/census")
        age_groups = list(range(0, 85, 5))  # All age groups
        files_to_load = []

        for age in age_groups:
            ffile = population_data_path / f"nic_f_{age}_2020_constrained_UNadj.tif"
            mfile = population_data_path / f"nic_m_{age}_2020_constrained_UNadj.tif"

            if ffile.exists():
                files_to_load.append(str(ffile))
            if mfile.exists():
                files_to_load.append(str(mfile))

        if not files_to_load:
            print("Warning: No population raster files found!")
            return np.zeros(len(grid_gdf))

        # Load and sum population data
        with rasterio.open(files_to_load[0]) as src:
            ref_transform = src.transform
            ref_crs = src.crs
            ref_shape = src.read(1).shape

        population_combined = np.zeros(ref_shape, dtype=np.float32)
        for fpath in files_to_load:
            with rasterio.open(fpath) as src:
                data = src.read(1)
                data = np.where(data <= -99999, 0, data)
                population_combined += data

        # Aggregate population to grid cells
        grid_res = self.get_resolution()
        nicaragua_gdf = get_nicaragua_boundary()
        bounds = nicaragua_gdf.total_bounds

        population_per_cell = self._aggregate_to_grid(
            population_combined, ref_transform, bounds, grid_res
        )

        # Count shelters and multiply by population for each cell
        shelter_population_counts = []
        for i, cell in enumerate(grid_gdf.geometry):
            shelter_count = shelter_gdf.within(cell).sum()
            cell_population = (
                population_per_cell[i] if i < len(population_per_cell) else 0
            )
            shelter_population_counts.append(shelter_count * cell_population)

        # Convert to numpy array
        shelter_population_counts = np.array(shelter_population_counts)

        # Apply impact-prone filter: only keep shelter population in impact-prone areas
        filtered_counts = np.where(impact_prone_mask, shelter_population_counts, 0)

        return filtered_counts

    @property
    def value_column(self):
        return "shelter_population_count"
