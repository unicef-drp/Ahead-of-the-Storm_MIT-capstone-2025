import os
from src.impact_analysis.layers.population import PopulationVulnerabilityLayer
from src.impact_analysis.layers.schools import (
    SchoolVulnerabilityLayer,
    SchoolPopulationVulnerabilityLayer,
)
from src.impact_analysis.analysis.impact import HurricaneImpactLayer


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
