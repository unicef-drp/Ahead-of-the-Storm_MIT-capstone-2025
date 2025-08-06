import os
import glob

from src.impact_analysis.layers.hurricane import HurricaneExposureLayer
from src.impact_analysis.layers.landslide import LandslideExposureLayer
from src.impact_analysis.layers.flood import FloodExposureLayer
from src.impact_analysis.layers.population import PopulationVulnerabilityLayer
from src.impact_analysis.layers.schools import (
    SchoolVulnerabilityLayer,
    SchoolPopulationVulnerabilityLayer,
)
from src.impact_analysis.layers.health_facilities import (
    HealthFacilityVulnerabilityLayer,
)
from src.impact_analysis.layers.shelters import ShelterVulnerabilityLayer
from src.impact_analysis.layers.poverty import (
    PovertyVulnerabilityLayer,
    SeverePovertyVulnerabilityLayer,
)
from src.impact_analysis.layers.impact_prone import (
    SchoolImpactProneVulnerabilityLayer,
    HealthFacilityImpactProneVulnerabilityLayer,
    ShelterImpactProneVulnerabilityLayer,
    SchoolPopulationImpactProneVulnerabilityLayer,
    HealthFacilityPopulationImpactProneVulnerabilityLayer,
    ShelterPopulationImpactProneVulnerabilityLayer,
)
from src.impact_analysis.layers.vaccination import UnvaccinatedVulnerabilityLayer
from src.impact_analysis.analysis.hurricane_impact import HurricaneImpactLayer
from src.impact_analysis.analysis.landslide_impact import LandslideImpactLayer
from src.impact_analysis.analysis.flood_impact import FloodImpactLayer
from src.utils.config_utils import get_config_value


def get_exposure_layer(
    exposure_type,
    hurricane_df,
    forecast_time,
    config,
    cache_dir,
    resampling_method="mean",
    resolution_context=None,
    use_cache=True,
):
    if exposure_type == "hurricane":
        return HurricaneExposureLayer(hurricane_df, forecast_time, config, cache_dir, use_cache=use_cache)
    if exposure_type == "flood":
        from src.utils.path_utils import get_data_path

        flood_raster_path = get_config_value(
            config,
            "impact_analysis.input.flood_data.enhanced_flood_file",
            "data/preprocessed/flood/nicaragua_flood_extent_20201117.tif",
        )
        flood_raster_path = str(get_data_path(flood_raster_path))

        return FloodExposureLayer(
            flood_raster_path=flood_raster_path,
            config=config,
            cache_dir=cache_dir,
            resampling_method=resampling_method,
            use_cache=use_cache,
        )
    if exposure_type == "landslide":
        # landslide_df is not used, instead config must specify the landslide file
        from src.utils.path_utils import get_data_path
        import re

        # Check if cached landslide exposure exists first
        cache_dir = cache_dir or get_config_value(
            config,
            "impact_analysis.output.cache_directory",
            "data/results/impact_analysis/cache/",
        )

        # Check for existing cache files
        cache_pattern = os.path.join(cache_dir, "landslide_exposure_*")
        existing_cache = glob.glob(cache_pattern)

        if existing_cache and use_cache:
            # Use cached data - provide a dummy file path since caching will handle it
            print(
                f"Found {len(existing_cache)} cached landslide exposure files, using cached data"
            )
            landslide_file = "cached"  # Dummy path since cache will be used
        else:
            # Find the latest landslide file only if no cache exists
            landslide_dir = get_data_path("data/preprocessed/landslide")
            landslide_files = list(
                landslide_dir.glob("landslide_forecast_48h_*_nicaragua.tif")
            )
            if not landslide_files:
                raise FileNotFoundError("No landslide data files found!")
            # Use the most recent file
            landslide_file = str(max(landslide_files, key=lambda x: x.stat().st_mtime))
            print(f"Using landslide file: {landslide_file}")

        return LandslideExposureLayer(
            landslide_file=landslide_file,
            config=config,
            cache_dir=cache_dir,
            resampling_method=resampling_method,
            resolution_context=resolution_context,
            use_cache=use_cache,
        )
    raise ValueError(f"Unknown exposure type: {exposure_type}")


def get_vulnerability_layer(vuln_type, config, cache_dir, resolution_context=None, use_cache=True):
    if vuln_type == "schools":
        return SchoolVulnerabilityLayer(
            config, cache_dir=cache_dir, resolution_context=resolution_context, use_cache=use_cache
        )
    if vuln_type == "school_population":
        return SchoolPopulationVulnerabilityLayer(
            config, cache_dir=cache_dir, resolution_context=resolution_context, use_cache=use_cache
        )
    if vuln_type == "health_facilities":
        return HealthFacilityVulnerabilityLayer(
            config,
            weighted_by_population=False,
            cache_dir=cache_dir,
            resolution_context=resolution_context,
            use_cache=use_cache,
        )
    if vuln_type == "health_facilities_population":
        return HealthFacilityVulnerabilityLayer(
            config,
            weighted_by_population=True,
            cache_dir=cache_dir,
            resolution_context=resolution_context,
            use_cache=use_cache,
        )
    if vuln_type == "shelters":
        return ShelterVulnerabilityLayer(
            config,
            weighted_by_capacity=False,
            cache_dir=cache_dir,
            resolution_context=resolution_context,
            use_cache=use_cache,
        )
    if vuln_type == "shelters_population":
        return ShelterVulnerabilityLayer(
            config,
            weighted_by_capacity=True,
            cache_dir=cache_dir,
            resolution_context=resolution_context,
            use_cache=use_cache,
        )
    if vuln_type == "population":
        return PopulationVulnerabilityLayer(
            config,
            age_groups=list(range(0, 85, 5)),
            gender="both",
            cache_dir=cache_dir,
            resolution_context=resolution_context,
            use_cache=use_cache,
        )
    if vuln_type == "children":
        return PopulationVulnerabilityLayer(
            config,
            age_groups=[0, 5, 10, 15],
            gender="both",
            cache_dir=cache_dir,
            resolution_context=resolution_context,
            use_cache=use_cache,
        )
    if vuln_type == "poverty":
        return PovertyVulnerabilityLayer(
            config,
            age_groups=list(range(0, 85, 5)),
            gender="both",
            cache_dir=cache_dir,
            resolution_context=resolution_context,
            use_cache=use_cache,
        )
    if vuln_type == "poverty_children":
        return PovertyVulnerabilityLayer(
            config,
            age_groups=[0, 5, 10, 15],
            gender="both",
            cache_dir=cache_dir,
            resolution_context=resolution_context,
            use_cache=use_cache,
        )
    if vuln_type == "severe_poverty":
        return SeverePovertyVulnerabilityLayer(
            config,
            age_groups=list(range(0, 85, 5)),
            gender="both",
            cache_dir=cache_dir,
            resolution_context=resolution_context,
            use_cache=use_cache,
        )
    if vuln_type == "severe_poverty_children":
        return SeverePovertyVulnerabilityLayer(
            config,
            age_groups=[0, 5, 10, 15],
            gender="both",
            cache_dir=cache_dir,
            resolution_context=resolution_context,
            use_cache=use_cache,
        )
    if vuln_type == "schools_impact_prone":
        return SchoolImpactProneVulnerabilityLayer(
            config, cache_dir=cache_dir, resolution_context=resolution_context, use_cache=use_cache
        )
    if vuln_type == "health_facilities_impact_prone":
        return HealthFacilityImpactProneVulnerabilityLayer(
            config, cache_dir=cache_dir, resolution_context=resolution_context, use_cache=use_cache
        )
    if vuln_type == "shelters_impact_prone":
        return ShelterImpactProneVulnerabilityLayer(
            config, cache_dir=cache_dir, resolution_context=resolution_context, use_cache=use_cache
        )
    if vuln_type == "school_population_impact_prone":
        return SchoolPopulationImpactProneVulnerabilityLayer(
            config, cache_dir=cache_dir, resolution_context=resolution_context, use_cache=use_cache
        )
    if vuln_type == "health_facilities_population_impact_prone":
        return HealthFacilityPopulationImpactProneVulnerabilityLayer(
            config, cache_dir=cache_dir, resolution_context=resolution_context, use_cache=use_cache
        )
    if vuln_type == "shelters_population_impact_prone":
        return ShelterPopulationImpactProneVulnerabilityLayer(
            config, cache_dir=cache_dir, resolution_context=resolution_context, use_cache=use_cache
        )
    if vuln_type == "unvaccinated":
        return UnvaccinatedVulnerabilityLayer(
            config,
            age_groups=list(range(0, 85, 5)),
            gender="both",
            cache_dir=cache_dir,
            resolution_context=resolution_context,
            use_cache=use_cache,
        )
    if vuln_type == "unvaccinated_children":
        return UnvaccinatedVulnerabilityLayer(
            config,
            age_groups=[0, 5, 10, 15],
            gender="both",
            cache_dir=cache_dir,
            resolution_context=resolution_context,
            use_cache=use_cache,
        )
    raise ValueError(f"Unknown vulnerability type: {vuln_type}")


def get_impact_layer(exposure, vulnerability, config):
    if isinstance(exposure, HurricaneExposureLayer):
        return HurricaneImpactLayer(exposure, vulnerability, config)
    if isinstance(exposure, FloodExposureLayer):
        return FloodImpactLayer(exposure, vulnerability, config)
    if isinstance(exposure, LandslideExposureLayer):
        return LandslideImpactLayer(exposure, vulnerability, config)
    raise ValueError(f"Unknown exposure layer type: {type(exposure)}")
