import os
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from src.impact_analysis.layers.base import VulnerabilityLayer
from src.impact_analysis.layers.population import PopulationVulnerabilityLayer
from src.utils.config_utils import get_config_value
from src.utils.path_utils import get_data_path
from src.utils.hurricane_geom import get_nicaragua_boundary
import pandas as pd
import unicodedata
import matplotlib.pyplot as plt

# Hardcoded poverty data (should be moved to config or CSV for production)
POVERTY_TABLE = pd.DataFrame(
    {
        "Region": [
            "Boaco",
            "Carazo",
            "Chinandega",
            "Chontales",
            "Esteli",
            "Granada",
            "Jinotega",
            "Leon",
            "Madriz",
            "Managua",
            "Masaya",
            "Matagalpa",
            "Nueva Segovia",
            "Raan",
            "Raas",
            "Rio San Juan",
            "Rivas",
        ],
        "H": [
            28.1,
            5.5,
            9.7,
            18.5,
            10.3,
            8.2,
            43.8,
            9.0,
            26.0,
            3.8,
            4.8,
            22.9,
            26.7,
            35.5,
            32.1,
            28.9,
            7.8,
        ],
        "Severe_Poverty": [
            10.3,
            0.1,
            1.4,
            4.9,
            1.2,
            2.2,
            19.5,
            2.1,
            6.7,
            1.3,
            0.9,
            7.4,
            9.6,
            15.2,
            12.1,
            9.7,
            1.0,
        ],
    }
)


# Normalize region names for robust join
def normalize_name(name):
    if not isinstance(name, str):
        return name
    name = name.lower().strip()
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    mapping = {
        "atlantico norte": "raan",
        "atlantico sur": "raas",
        "esteli": "esteli",
        "leon": "leon",
        "rio san juan": "rio san juan",
    }
    if name in mapping:
        return mapping[name]
    return name


POVERTY_TABLE["region_norm"] = POVERTY_TABLE["Region"].apply(normalize_name)


class PovertyVulnerabilityLayer(VulnerabilityLayer):
    def __init__(self, config, age_groups=None, gender="both", cache_dir=None, resolution_context=None):
        super().__init__(config, resolution_context)
        self.age_groups = (
            age_groups if age_groups is not None else list(range(0, 85, 5))
        )
        self.gender = gender
        self.grid_gdf = None
        self._poverty_grid = None
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
                return os.path.join(self.cache_dir, f"poverty_vulnerability_{self.gender}_ages_{age_str}_{self.resolution_context}_{resolution}deg.parquet")
            else:
                return os.path.join(self.cache_dir, f"poverty_vulnerability_{self.gender}_ages_{age_str}_{self.resolution_context}_{resolution}deg.gpkg")
        else:
            return os.path.join(
                self.cache_dir, f"poverty_vulnerability_{self.gender}_ages_{age_str}.gpkg"
            )

    def compute_grid(self):
        if self.grid_gdf is not None:
            return self.grid_gdf
        cache_path = self._cache_path()

        def compute_func():
            # Get population grid first
            pop_layer = PopulationVulnerabilityLayer(
                self.config, self.age_groups, self.gender, self.cache_dir, self.resolution_context
            )
            pop_gdf = pop_layer.compute_grid()
            
            # Load regions
            regions_gdf = gpd.read_file(
                get_data_path("data/raw/gadm/nicaragua_departments.geojson")
            )
            regions_gdf["region_norm"] = regions_gdf["NAME_1"].apply(normalize_name)
            
            # Merge poverty data
            regions_gdf = regions_gdf.merge(
                POVERTY_TABLE[["region_norm", "H"]], on="region_norm", how="left"
            )
            
            # Compute average poverty rate for regions adjacent to Lago Nicaragua
            adjacent_regions = ["rivas", "granada", "masaya", "managua"]
            avg_adjacent_h = (
                POVERTY_TABLE[POVERTY_TABLE["region_norm"].isin(adjacent_regions)][
                    "H"
                ].mean()
                / 100.0
            )
            
            # Use efficient spatial join between population grid and regions
            joined_gdf = gpd.sjoin(pop_gdf, regions_gdf, how="left", predicate="within")
            
            # Vectorized poverty calculation: population * poverty_rate
            # Initialize poverty counts
            joined_gdf["poverty_count"] = 0.0
            
            # Handle Lago Nicaragua case
            lago_mask = joined_gdf["region_norm"] == "lago nicaragua"
            joined_gdf.loc[lago_mask, "poverty_count"] = joined_gdf.loc[lago_mask, "population_count"] * avg_adjacent_h
            
            # Handle other regions
            other_mask = (joined_gdf["region_norm"] != "lago nicaragua") & (~joined_gdf["region_norm"].isna())
            joined_gdf.loc[other_mask, "poverty_count"] = joined_gdf.loc[other_mask, "population_count"] * (joined_gdf.loc[other_mask, "H"] / 100.0)
            
            # Group by original grid cell and sum poverty counts (in case of multiple regions)
            result_gdf = joined_gdf.groupby(joined_gdf.index).agg({
                "geometry": "first",
                "population_count": "first",
                "poverty_count": "sum"
            }).reset_index(drop=True)
            
            # Ensure we have a clean GeoDataFrame
            final_gdf = gpd.GeoDataFrame({
                "geometry": result_gdf["geometry"],
                "population_count": result_gdf["population_count"],
                "poverty_count": result_gdf["poverty_count"]
            }, crs="EPSG:4326")
            
            return final_gdf

        self.grid_gdf = self._load_or_compute_grid(
            cache_path, "poverty_count", compute_func
        )
        self._poverty_grid = self.grid_gdf["poverty_count"].values
        return self.grid_gdf

    def plot(self, ax=None, output_dir="data/results/impact_analysis/"):
        grid_gdf = self.compute_grid()
        age_str = "_".join(map(str, self.age_groups))
        output_filename = f"poverty_vulnerability_{self.gender}_ages_{age_str}.png"
        plot_title = f"People in Poverty Vulnerability Heatmap (Log Scale)\nGender: {self.gender}, Ages: {self.age_groups}"
        self._plot_vulnerability_grid(
            grid_gdf,
            value_column="poverty_count",
            cmap="Purples",
            legend_label="Log10(People in Poverty + 1) per Cell",
            output_dir=output_dir,
            output_filename=output_filename,
            plot_title=plot_title,
            ax=ax,
        )

    @property
    def value_column(self):
        return "poverty_count"


class SeverePovertyVulnerabilityLayer(VulnerabilityLayer):
    def __init__(self, config, age_groups=None, gender="both", cache_dir=None, resolution_context=None):
        super().__init__(config, resolution_context)
        self.age_groups = (
            age_groups if age_groups is not None else list(range(0, 85, 5))
        )
        self.gender = gender
        self.grid_gdf = None
        self._severe_poverty_grid = None
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
                return os.path.join(self.cache_dir, f"severepoverty_vulnerability_{self.gender}_ages_{age_str}_{self.resolution_context}_{resolution}deg.parquet")
            else:
                return os.path.join(self.cache_dir, f"severepoverty_vulnerability_{self.gender}_ages_{age_str}_{self.resolution_context}_{resolution}deg.gpkg")
        else:
            return os.path.join(
                self.cache_dir,
                f"severepoverty_vulnerability_{self.gender}_ages_{age_str}.gpkg",
            )

    def compute_grid(self):
        if self.grid_gdf is not None:
            return self.grid_gdf
        cache_path = self._cache_path()

        def compute_func():
            # Get population grid first
            pop_layer = PopulationVulnerabilityLayer(
                self.config, self.age_groups, self.gender, self.cache_dir, self.resolution_context
            )
            pop_gdf = pop_layer.compute_grid()
            
            # Load regions
            regions_gdf = gpd.read_file(
                get_data_path("data/raw/gadm/nicaragua_departments.geojson")
            )
            regions_gdf["region_norm"] = regions_gdf["NAME_1"].apply(normalize_name)
            
            # Merge poverty data
            regions_gdf = regions_gdf.merge(
                POVERTY_TABLE[["region_norm", "Severe_Poverty"]],
                on="region_norm",
                how="left",
            )
            
            # Compute average severe poverty rate for regions adjacent to Lago Nicaragua
            adjacent_regions = ["rivas", "granada", "masaya", "managua"]
            avg_adjacent_s = (
                POVERTY_TABLE[POVERTY_TABLE["region_norm"].isin(adjacent_regions)][
                    "Severe_Poverty"
                ].mean()
                / 100.0
            )
            
            # Use efficient spatial join between population grid and regions
            joined_gdf = gpd.sjoin(pop_gdf, regions_gdf, how="left", predicate="within")
            
            # Vectorized severe poverty calculation: population * severe_poverty_rate
            # Initialize severe poverty counts
            joined_gdf["severepoverty_count"] = 0.0
            
            # Handle Lago Nicaragua case
            lago_mask = joined_gdf["region_norm"] == "lago nicaragua"
            joined_gdf.loc[lago_mask, "severepoverty_count"] = joined_gdf.loc[lago_mask, "population_count"] * avg_adjacent_s
            
            # Handle other regions
            other_mask = (joined_gdf["region_norm"] != "lago nicaragua") & (~joined_gdf["region_norm"].isna())
            joined_gdf.loc[other_mask, "severepoverty_count"] = joined_gdf.loc[other_mask, "population_count"] * (joined_gdf.loc[other_mask, "Severe_Poverty"] / 100.0)
            
            # Group by original grid cell and sum severe poverty counts (in case of multiple regions)
            result_gdf = joined_gdf.groupby(joined_gdf.index).agg({
                "geometry": "first",
                "population_count": "first",
                "severepoverty_count": "sum"
            }).reset_index(drop=True)
            
            # Ensure we have a clean GeoDataFrame
            final_gdf = gpd.GeoDataFrame({
                "geometry": result_gdf["geometry"],
                "population_count": result_gdf["population_count"],
                "severepoverty_count": result_gdf["severepoverty_count"]
            }, crs="EPSG:4326")
            
            return final_gdf

        self.grid_gdf = self._load_or_compute_grid(
            cache_path, "severepoverty_count", compute_func
        )
        self._severe_poverty_grid = self.grid_gdf["severepoverty_count"].values
        return self.grid_gdf

    def plot(self, ax=None, output_dir="data/results/impact_analysis/"):
        grid_gdf = self.compute_grid()
        age_str = "_".join(map(str, self.age_groups))
        output_filename = f"severe_poverty_vulnerability_{self.gender}_ages_{age_str}.png"
        plot_title = f"People in Severe Poverty Vulnerability Heatmap (Log Scale)\nGender: {self.gender}, Ages: {self.age_groups}"
        self._plot_vulnerability_grid(
            grid_gdf,
            value_column="severepoverty_count",
            cmap="Reds",
            legend_label="Log10(People in Severe Poverty + 1) per Cell",
            output_dir=output_dir,
            output_filename=output_filename,
            plot_title=plot_title,
            ax=ax,
        )

    @property
    def value_column(self):
        return "severepoverty_count"
