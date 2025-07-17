from src.impact_analysis.layers.hurricane import HurricaneExposureLayer
from src.impact_analysis.layers.population import PopulationVulnerabilityLayer
from src.impact_analysis.layers.schools import (
    SchoolVulnerabilityLayer,
    SchoolPopulationVulnerabilityLayer,
)
from src.impact_analysis.layers.health_facilities import HealthFacilityVulnerabilityLayer
from src.impact_analysis.layers.shelters import ShelterVulnerabilityLayer
from src.impact_analysis.layers.poverty import (
    PovertyVulnerabilityLayer,
    SeverePovertyVulnerabilityLayer,
)
from src.impact_analysis.analysis.impact import HurricaneImpactLayer


def get_exposure_layer(exposure_type, hurricane_df, forecast_time, config, cache_dir):
    if exposure_type == "hurricane":
        return HurricaneExposureLayer(hurricane_df, forecast_time, config, cache_dir)
    # Add more exposure types (e.g., landslide, flood) as needed
    raise ValueError(f"Unknown exposure type: {exposure_type}")


def get_vulnerability_layer(vuln_type, config, cache_dir):
    if vuln_type == "schools":
        return SchoolVulnerabilityLayer(config, cache_dir=cache_dir)
    if vuln_type == "school_population":
        return SchoolPopulationVulnerabilityLayer(config, cache_dir=cache_dir)
    if vuln_type == "health_facilities":
        return HealthFacilityVulnerabilityLayer(config, weighted_by_population=False, cache_dir=cache_dir)
    if vuln_type == "health_facilities_population":
        return HealthFacilityVulnerabilityLayer(config, weighted_by_population=True, cache_dir=cache_dir)
    if vuln_type == "shelters":
        return ShelterVulnerabilityLayer(config, weighted_by_capacity=False, cache_dir=cache_dir)
    if vuln_type == "shelters_population":
        return ShelterVulnerabilityLayer(config, weighted_by_capacity=True, cache_dir=cache_dir)
    if vuln_type == "population":
        return PopulationVulnerabilityLayer(config, age_groups=list(range(0, 85, 5)), gender="both", cache_dir=cache_dir)
    if vuln_type == "poverty":
        return PovertyVulnerabilityLayer(config, age_groups=list(range(0, 85, 5)), gender="both", cache_dir=cache_dir)
    if vuln_type == "poverty_children":
        return PovertyVulnerabilityLayer(config, age_groups=[0, 5, 10, 15], gender="both", cache_dir=cache_dir)
    if vuln_type == "severe_poverty":
        return SeverePovertyVulnerabilityLayer(config, age_groups=list(range(0, 85, 5)), gender="both", cache_dir=cache_dir)
    if vuln_type == "severe_poverty_children":
        return SeverePovertyVulnerabilityLayer(config, age_groups=[0, 5, 10, 15], gender="both", cache_dir=cache_dir)
    # Add more vulnerability types as needed
    raise ValueError(f"Unknown vulnerability type: {vuln_type}")


def get_impact_layer(exposure, vulnerability, config):
    # For now, only HurricaneImpactLayer is implemented
    return HurricaneImpactLayer(exposure, vulnerability, config) 