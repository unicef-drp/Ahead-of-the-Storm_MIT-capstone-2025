"""
Vaccination vulnerability layer for the Ahead of the Storm project.

This module provides functionality to create vulnerability layers based on
unvaccinated population data at the administrative level 2 (department).
"""

import os
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from typing import Dict, Any
from src.impact_analysis.layers.base import VulnerabilityLayer
from src.impact_analysis.layers.population import PopulationVulnerabilityLayer
from src.utils.config_utils import get_config_value
from src.utils.path_utils import get_data_path
from src.utils.hurricane_geom import get_nicaragua_boundary
import pandas as pd
import unicodedata
import matplotlib.pyplot as plt

# Hardcoded vaccination data (should be moved to config or CSV for production)
VACCINATION_TABLE = pd.DataFrame(
    {
        "Region": [
            "Nueva Segovia",
            "Jinotega", 
            "Madriz",
            "Estelí",
            "Chinandega",
            "León",
            "Matagalpa",
            "Boaco",
            "Managua",
            "Masaya",
            "Chontales",
            "Granada",
            "Carazo",
            "Rivas",
            "Río San Juan",
            "RACCN",
            "RACCS",
        ],
        "Vaccination_Rate": [
            95.5,
            89.6,
            97.7,
            96.9,
            82.2,
            87.7,
            89.0,
            91.9,
            74.2,
            81.3,
            98.0,
            87.2,
            93.8,
            96.7,
            89.3,
            68.5,
            82.6,
        ],
    }
)

# Calculate unvaccinated rates
VACCINATION_TABLE["Unvaccinated_Rate"] = 100.0 - VACCINATION_TABLE["Vaccination_Rate"]


# Normalize region names for robust join
def normalize_vaccination_name(name):
    """Normalize region names for vaccination data join."""
    if not isinstance(name, str):
        return name
    name = name.lower().strip()
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    
    mapping = {
        "raccn": "atlantico norte",
        "raccs": "atlantico sur",
        "rio san juan": "rio san juan",
        "esteli": "esteli",
        "leon": "leon",
    }
    if name in mapping:
        return mapping[name]
    return name


VACCINATION_TABLE["region_norm"] = VACCINATION_TABLE["Region"].apply(normalize_vaccination_name)


class UnvaccinatedVulnerabilityLayer(VulnerabilityLayer):
    def __init__(self, config, age_groups=None, gender="both", cache_dir=None, resolution_context=None, use_cache=True):
        super().__init__(config, resolution_context)
        self.age_groups = (
            age_groups if age_groups is not None else list(range(0, 85, 5))
        )
        self.gender = gender
        self.grid_gdf = None
        self._unvaccinated_grid = None
        self.use_cache = use_cache
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
                return os.path.join(self.cache_dir, f"unvaccinated_vulnerability_{self.gender}_ages_{age_str}_{self.resolution_context}_{resolution}deg.parquet")
            else:
                return os.path.join(self.cache_dir, f"unvaccinated_vulnerability_{self.gender}_ages_{age_str}_{self.resolution_context}_{resolution}deg.gpkg")
        else:
            return os.path.join(
                self.cache_dir, f"unvaccinated_vulnerability_{self.gender}_ages_{age_str}.gpkg"
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
            regions_gdf["region_norm"] = regions_gdf["NAME_1"].apply(normalize_vaccination_name)
            
            # Merge vaccination data
            regions_gdf = regions_gdf.merge(
                VACCINATION_TABLE[["region_norm", "Unvaccinated_Rate"]], on="region_norm", how="left"
            )
            
            # Compute average unvaccinated rate for regions adjacent to Lago Nicaragua
            adjacent_regions = ["rivas", "granada", "masaya", "managua"]
            avg_adjacent_unvaccinated = (
                VACCINATION_TABLE[VACCINATION_TABLE["region_norm"].isin(adjacent_regions)][
                    "Unvaccinated_Rate"
                ].mean()
                / 100.0
            )
            
            # Use efficient spatial join between population grid and regions
            # Use 'intersects' instead of 'within' to capture small regions like Masaya
            joined_gdf = gpd.sjoin(pop_gdf, regions_gdf, how="left", predicate="intersects")
            
            # For each grid cell, keep only the first region assignment (most likely the primary one)
            # This prevents double-counting when grid cells intersect multiple regions
            joined_gdf = joined_gdf.drop_duplicates(subset='geometry', keep='first')
            
            # Vectorized unvaccinated calculation: population * unvaccinated_rate
            # Initialize unvaccinated counts
            joined_gdf["unvaccinated_count"] = 0.0
            
            # Handle Lago Nicaragua case
            lago_mask = joined_gdf["region_norm"] == "lago nicaragua"
            joined_gdf.loc[lago_mask, "unvaccinated_count"] = joined_gdf.loc[lago_mask, "population_count"] * avg_adjacent_unvaccinated
            
            # Handle other regions
            other_mask = (joined_gdf["region_norm"] != "lago nicaragua") & (~joined_gdf["region_norm"].isna())
            joined_gdf.loc[other_mask, "unvaccinated_count"] = joined_gdf.loc[other_mask, "population_count"] * (joined_gdf.loc[other_mask, "Unvaccinated_Rate"] / 100.0)
            
            # Create final result (no need to group since we already deduplicated)
            result_gdf = joined_gdf[["geometry", "population_count", "unvaccinated_count"]].copy()
            
            # Ensure we have a clean GeoDataFrame
            final_gdf = gpd.GeoDataFrame({
                "geometry": result_gdf["geometry"],
                "population_count": result_gdf["population_count"],
                "unvaccinated_count": result_gdf["unvaccinated_count"]
            }, crs="EPSG:4326")
            
            return final_gdf

        self.grid_gdf = self._load_or_compute_grid(
            cache_path, "unvaccinated_count", compute_func, use_cache=self.use_cache
        )
        self._unvaccinated_grid = self.grid_gdf["unvaccinated_count"].values
        return self.grid_gdf

    def get_plot_metadata(self) -> Dict[str, Any]:
        """Return metadata for plotting this unvaccinated vulnerability layer."""
        # Determine if this is children or total population
        if self.age_groups == [0, 5, 10, 15]:
            vulnerability_type = "Unvaccinated Children"
            colormap = "Reds"
        else:
            vulnerability_type = "Unvaccinated Population"
            colormap = "Reds"
        
        return {
            "layer_type": "vulnerability",
            "vulnerability_type": vulnerability_type,
            "data_column": "unvaccinated_count",
            "colormap": colormap,
            "title_template": "Concentration of {vulnerability_type}",
            "legend_template": "{vulnerability_type} per Cell",
            "filename_template": "{vulnerability_type}_vulnerability_{parameters}",
            "special_features": []
        }

    def plot(self, ax=None, output_dir="data/results/impact_analysis/"):
        """Plot the unvaccinated vulnerability layer using universal plotting function."""
        from src.impact_analysis.utils.plotting_utils import plot_layer_with_scales
        plot_layer_with_scales(self, output_dir=output_dir)

    @property
    def value_column(self):
        return "unvaccinated_count" 