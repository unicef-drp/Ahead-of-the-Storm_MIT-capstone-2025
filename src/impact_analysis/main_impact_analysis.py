import os
from src.impact_analysis.layers.population import PopulationVulnerabilityLayer
from src.impact_analysis.layers.schools import (
    SchoolVulnerabilityLayer,
    SchoolPopulationVulnerabilityLayer,
)
from src.impact_analysis.layers.health_facilities import (
    HealthFacilityVulnerabilityLayer,
)
from src.impact_analysis.layers.shelters import ShelterVulnerabilityLayer
from src.impact_analysis.analysis.impact import HurricaneImpactLayer
from src.impact_analysis.layers.poverty import (
    PovertyVulnerabilityLayer,
    SeverePovertyVulnerabilityLayer,
)


def ensure_subdir(base_dir, subfolder):
    subdir = os.path.join(base_dir, subfolder)
    os.makedirs(subdir, exist_ok=True)
    return subdir


def main_impact_analysis(config, exposure, output_dir, cache_dir):
    # --- Hurricane Exposure (shared across all analyses) ---
    print("\n[Hurricane Exposure Layer]")
    exposure.plot(output_dir=output_dir)

    # --- Schools ---
    output_schools = ensure_subdir(output_dir, "hurricane_schools")
    school_vuln = SchoolVulnerabilityLayer(config, cache_dir=cache_dir)
    school_impact = HurricaneImpactLayer(exposure, school_vuln, config)
    print("\n[Vulnerability Layer: Schools]")
    school_vuln.plot(output_dir=output_schools)
    print("\n[Impact Layer: Schools]")
    school_impact.plot(output_dir=output_schools)
    print("\n[Binary Probability: Schools]")
    school_impact.plot_binary_probability(output_dir=output_schools)
    print("\n[Best/Worst Case Overlay Plots: Schools]")
    school_impact.plot_best_worst_case_overlay(output_dir=output_schools)
    print(f"\nExpected affected schools: {school_impact.expected_impact():.2f}")
    print(f"Best case (min) affected schools: {school_impact.best_case():.2f}")
    print(f"Worst case (max) affected schools: {school_impact.worst_case():.2f}")
    school_impact.save_impact_summary(output_dir=output_schools)

    # --- School Population ---
    output_schoolpop = ensure_subdir(output_dir, "hurricane_schoolpopulation")
    school_pop_vuln = SchoolPopulationVulnerabilityLayer(config, cache_dir=cache_dir)
    school_pop_impact = HurricaneImpactLayer(exposure, school_pop_vuln, config)
    print("\n[Vulnerability Layer: School Population]")
    school_pop_vuln.plot(output_dir=output_schoolpop)
    print("\n[Impact Layer: School Population]")
    school_pop_impact.plot(output_dir=output_schoolpop)
    print("\n[Binary Probability: School Population]")
    school_pop_impact.plot_binary_probability(output_dir=output_schoolpop)
    print("\n[Best/Worst Case Overlay Plots: School Population]")
    school_pop_impact.plot_best_worst_case_overlay(output_dir=output_schoolpop)
    print(
        f"\nExpected affected school people: {school_pop_impact.expected_impact():.2f}"
    )
    print(
        f"Best case (min) affected school people: {school_pop_impact.best_case():.2f}"
    )
    print(
        f"Worst case (max) affected school people: {school_pop_impact.worst_case():.2f}"
    )
    school_pop_impact.save_impact_summary(output_dir=output_schoolpop)

    # --- Health Facilities (count) ---
    output_health = ensure_subdir(output_dir, "hurricane_healthfacilities")
    health_vuln = HealthFacilityVulnerabilityLayer(
        config, weighted_by_population=False, cache_dir=cache_dir
    )
    health_impact = HurricaneImpactLayer(exposure, health_vuln, config)
    print("\n[Vulnerability Layer: Health Facilities]")
    health_vuln.plot(output_dir=output_health)
    print("\n[Impact Layer: Health Facilities]")
    health_impact.plot(output_dir=output_health)
    print("\n[Binary Probability: Health Facilities]")
    health_impact.plot_binary_probability(output_dir=output_health)
    print("\n[Best/Worst Case Overlay Plots: Health Facilities]")
    health_impact.plot_best_worst_case_overlay(output_dir=output_health)
    print(
        f"\nExpected affected health facilities: {health_impact.expected_impact():.2f}"
    )
    print(
        f"Best case (min) affected health facilities: {health_impact.best_case():.2f}"
    )
    print(
        f"Worst case (max) affected health facilities: {health_impact.worst_case():.2f}"
    )
    health_impact.save_impact_summary(output_dir=output_health)

    # --- Health Facilities (population weighted) ---
    output_healthpop = ensure_subdir(output_dir, "hurricane_healthfacilitiespopulation")
    healthpop_vuln = HealthFacilityVulnerabilityLayer(
        config, weighted_by_population=True, cache_dir=cache_dir
    )
    healthpop_impact = HurricaneImpactLayer(exposure, healthpop_vuln, config)
    print("\n[Vulnerability Layer: Health Facilities (Population Weighted)]")
    healthpop_vuln.plot(output_dir=output_healthpop)
    print("\n[Impact Layer: Health Facilities (Population Weighted)]")
    healthpop_impact.plot(output_dir=output_healthpop)
    print("\n[Binary Probability: Health Facilities (Population Weighted)]")
    healthpop_impact.plot_binary_probability(output_dir=output_healthpop)
    print("\n[Best/Worst Case Overlay Plots: Health Facilities (Population Weighted)]")
    healthpop_impact.plot_best_worst_case_overlay(output_dir=output_healthpop)
    print(
        f"\nExpected affected health facility population: {healthpop_impact.expected_impact():.2f}"
    )
    print(
        f"Best case (min) affected health facility population: {healthpop_impact.best_case():.2f}"
    )
    print(
        f"Worst case (max) affected health facility population: {healthpop_impact.worst_case():.2f}"
    )
    healthpop_impact.save_impact_summary(output_dir=output_healthpop)

    # --- Shelters (count) ---
    output_shelters = ensure_subdir(output_dir, "hurricane_shelters")
    shelter_vuln = ShelterVulnerabilityLayer(
        config, weighted_by_capacity=False, cache_dir=cache_dir
    )
    shelter_impact = HurricaneImpactLayer(exposure, shelter_vuln, config)
    print("\n[Vulnerability Layer: Shelters]")
    shelter_vuln.plot(output_dir=output_shelters)
    print("\n[Impact Layer: Shelters]")
    shelter_impact.plot(output_dir=output_shelters)
    print("\n[Binary Probability: Shelters]")
    shelter_impact.plot_binary_probability(output_dir=output_shelters)
    print("\n[Best/Worst Case Overlay Plots: Shelters]")
    shelter_impact.plot_best_worst_case_overlay(output_dir=output_shelters)
    print(f"\nExpected affected shelters: {shelter_impact.expected_impact():.2f}")
    print(f"Best case (min) affected shelters: {shelter_impact.best_case():.2f}")
    print(f"Worst case (max) affected shelters: {shelter_impact.worst_case():.2f}")
    shelter_impact.save_impact_summary(output_dir=output_shelters)

    # --- Shelters (capacity weighted) ---
    output_shelterspop = ensure_subdir(output_dir, "hurricane_shelterspopulation")
    shelterpop_vuln = ShelterVulnerabilityLayer(
        config, weighted_by_capacity=True, cache_dir=cache_dir
    )
    shelterpop_impact = HurricaneImpactLayer(exposure, shelterpop_vuln, config)
    print("\n[Vulnerability Layer: Shelters (Capacity Weighted)]")
    shelterpop_vuln.plot(output_dir=output_shelterspop)
    print("\n[Impact Layer: Shelters (Capacity Weighted)]")
    shelterpop_impact.plot(output_dir=output_shelterspop)
    print("\n[Binary Probability: Shelters (Capacity Weighted)]")
    shelterpop_impact.plot_binary_probability(output_dir=output_shelterspop)
    print("\n[Best/Worst Case Overlay Plots: Shelters (Capacity Weighted)]")
    shelterpop_impact.plot_best_worst_case_overlay(output_dir=output_shelterspop)
    print(
        f"\nExpected affected shelter capacity: {shelterpop_impact.expected_impact():.2f}"
    )
    print(
        f"Best case (min) affected shelter capacity: {shelterpop_impact.best_case():.2f}"
    )
    print(
        f"Worst case (max) affected shelter capacity: {shelterpop_impact.worst_case():.2f}"
    )
    shelterpop_impact.save_impact_summary(output_dir=output_shelterspop)

    # --- Population ---
    output_population = ensure_subdir(output_dir, "hurricane_population")
    pop_vuln = PopulationVulnerabilityLayer(
        config, age_groups=list(range(0, 85, 5)), gender="both", cache_dir=cache_dir
    )
    pop_impact = HurricaneImpactLayer(exposure, pop_vuln, config)
    print("\n[Vulnerability Layer: Population]")
    pop_vuln.plot(output_dir=output_population)
    print("\n[Impact Layer: Population]")
    pop_impact.plot(output_dir=output_population)
    print("\n[Binary Probability: Population]")
    pop_impact.plot_binary_probability(output_dir=output_population)
    print("\n[Best/Worst Case Overlay Plots: Population]")
    pop_impact.plot_best_worst_case_overlay(output_dir=output_population)
    print(f"\nExpected affected people: {pop_impact.expected_impact():.2f}")
    print(f"Best case (min) affected people: {pop_impact.best_case():.2f}")
    print(f"Worst case (max) affected people: {pop_impact.worst_case():.2f}")
    pop_impact.save_impact_summary(output_dir=output_population)

    # --- Poverty (Headcount Ratio, all ages) ---
    output_poverty = ensure_subdir(output_dir, "hurricane_poverty")
    poverty_vuln = PovertyVulnerabilityLayer(
        config, age_groups=list(range(0, 85, 5)), gender="both", cache_dir=cache_dir
    )
    poverty_impact = HurricaneImpactLayer(exposure, poverty_vuln, config)
    print("\n[Vulnerability Layer: Poverty (Headcount Ratio, all ages)]")
    poverty_vuln.plot(output_dir=output_poverty)
    print("\n[Impact Layer: Poverty (Headcount Ratio, all ages)]")
    poverty_impact.plot(output_dir=output_poverty)
    print("\n[Binary Probability: Poverty (Headcount Ratio, all ages)]")
    poverty_impact.plot_binary_probability(output_dir=output_poverty)
    print("\n[Best/Worst Case Overlay Plots: Poverty (Headcount Ratio, all ages)]")
    poverty_impact.plot_best_worst_case_overlay(output_dir=output_poverty)
    print(
        f"\nExpected affected people in poverty: {poverty_impact.expected_impact():.2f}"
    )
    print(
        f"Best case (min) affected people in poverty: {poverty_impact.best_case():.2f}"
    )
    print(
        f"Worst case (max) affected people in poverty: {poverty_impact.worst_case():.2f}"
    )
    poverty_impact.save_impact_summary(output_dir=output_poverty)

    # --- Poverty (Headcount Ratio, children) ---
    output_poverty_children = ensure_subdir(output_dir, "hurricane_poverty_children")
    poverty_children_vuln = PovertyVulnerabilityLayer(
        config, age_groups=[0, 5, 10, 15], gender="both", cache_dir=cache_dir
    )
    poverty_children_impact = HurricaneImpactLayer(
        exposure, poverty_children_vuln, config
    )
    print("\n[Vulnerability Layer: Poverty (Headcount Ratio, children)]")
    poverty_children_vuln.plot(output_dir=output_poverty_children)
    print("\n[Impact Layer: Poverty (Headcount Ratio, children)]")
    poverty_children_impact.plot(output_dir=output_poverty_children)
    print("\n[Binary Probability: Poverty (Headcount Ratio, children)]")
    poverty_children_impact.plot_binary_probability(output_dir=output_poverty_children)
    print("\n[Best/Worst Case Overlay Plots: Poverty (Headcount Ratio, children)]")
    poverty_children_impact.plot_best_worst_case_overlay(
        output_dir=output_poverty_children
    )
    print(
        f"\nExpected affected children in poverty: {poverty_children_impact.expected_impact():.2f}"
    )
    print(
        f"Best case (min) affected children in poverty: {poverty_children_impact.best_case():.2f}"
    )
    print(
        f"Worst case (max) affected children in poverty: {poverty_children_impact.worst_case():.2f}"
    )
    poverty_children_impact.save_impact_summary(output_dir=output_poverty_children)

    # --- Severe Poverty (all ages) ---
    output_severepoverty = ensure_subdir(output_dir, "hurricane_severepoverty")
    severepoverty_vuln = SeverePovertyVulnerabilityLayer(
        config, age_groups=list(range(0, 85, 5)), gender="both", cache_dir=cache_dir
    )
    severepoverty_impact = HurricaneImpactLayer(exposure, severepoverty_vuln, config)
    print("\n[Vulnerability Layer: Severe Poverty (all ages)]")
    severepoverty_vuln.plot(output_dir=output_severepoverty)
    print("\n[Impact Layer: Severe Poverty (all ages)]")
    severepoverty_impact.plot(output_dir=output_severepoverty)
    print("\n[Binary Probability: Severe Poverty (all ages)]")
    severepoverty_impact.plot_binary_probability(output_dir=output_severepoverty)
    print("\n[Best/Worst Case Overlay Plots: Severe Poverty (all ages)]")
    severepoverty_impact.plot_best_worst_case_overlay(output_dir=output_severepoverty)
    print(
        f"\nExpected affected people in severe poverty: {severepoverty_impact.expected_impact():.2f}"
    )
    print(
        f"Best case (min) affected people in severe poverty: {severepoverty_impact.best_case():.2f}"
    )
    print(
        f"Worst case (max) affected people in severe poverty: {severepoverty_impact.worst_case():.2f}"
    )
    severepoverty_impact.save_impact_summary(output_dir=output_severepoverty)

    # --- Severe Poverty (children) ---
    output_severepoverty_children = ensure_subdir(
        output_dir, "hurricane_severepoverty_children"
    )
    severepoverty_children_vuln = SeverePovertyVulnerabilityLayer(
        config, age_groups=[0, 5, 10, 15], gender="both", cache_dir=cache_dir
    )
    severepoverty_children_impact = HurricaneImpactLayer(
        exposure, severepoverty_children_vuln, config
    )
    print("\n[Vulnerability Layer: Severe Poverty (children)]")
    severepoverty_children_vuln.plot(output_dir=output_severepoverty_children)
    print("\n[Impact Layer: Severe Poverty (children)]")
    severepoverty_children_impact.plot(output_dir=output_severepoverty_children)
    print("\n[Binary Probability: Severe Poverty (children)]")
    severepoverty_children_impact.plot_binary_probability(
        output_dir=output_severepoverty_children
    )
    print("\n[Best/Worst Case Overlay Plots: Severe Poverty (children)]")
    severepoverty_children_impact.plot_best_worst_case_overlay(
        output_dir=output_severepoverty_children
    )
    print(
        f"\nExpected affected children in severe poverty: {severepoverty_children_impact.expected_impact():.2f}"
    )
    print(
        f"Best case (min) affected children in severe poverty: {severepoverty_children_impact.best_case():.2f}"
    )
    print(
        f"Worst case (max) affected children in severe poverty: {severepoverty_children_impact.worst_case():.2f}"
    )
    severepoverty_children_impact.save_impact_summary(
        output_dir=output_severepoverty_children
    )
