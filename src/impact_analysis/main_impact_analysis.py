import os
import sys
import yaml
import pandas as pd
from pathlib import Path
import numpy as np
import time

from src.impact_analysis.helper.factories import (
    get_exposure_layer,
    get_vulnerability_layer,
    get_impact_layer,
)
from src.utils.config_utils import load_config, get_config_value
from src.utils.path_utils import get_data_path


def ensure_subdir(base_dir, subfolder):
    subdir = os.path.join(base_dir, subfolder)
    os.makedirs(subdir, exist_ok=True)
    return subdir


def run_analysis(
    config,
    exposure_type,
    vuln_type,
    hurricane_df,
    forecast_time,
    output_dir,
    cache_dir,
    scenario="mean",
    use_cache=True,
):
    """Run impact analysis for any exposure type and vulnerability type."""
    print(
        f"\n=== Running analysis: Exposure={exposure_type}, Vulnerability={vuln_type} ({scenario}) ==="
    )

    # For hurricanes, ignore scenario in folder name
    if exposure_type == "hurricane":
        output_subdir = ensure_subdir(output_dir, f"hurricane_{vuln_type}")
    elif exposure_type == "landslide":
        # For landslide, ignore scenario in folder name (clean naming)
        output_subdir = ensure_subdir(output_dir, f"landslide_{vuln_type}")
    else:
        output_subdir = ensure_subdir(
            output_dir, f"{exposure_type}_{vuln_type}_{scenario}"
        )

    # Track total start time
    total_start_time = time.time()

    # Create exposure layer with appropriate parameters
    print(f"\n[Exposure Layer: {exposure_type} ({scenario})]")
    exposure_start = time.time()

    if exposure_type == "landslide":
        # Landslide-specific parameters
        exposure = get_exposure_layer(
            exposure_type,
            None,  # hurricane_df
            forecast_time,
            config,
            cache_dir,
            resampling_method="mean",
            resolution_context="landslide_visualization",
            use_cache=use_cache,
        )
    else:
        # Standard parameters for hurricane and flood
        exposure = get_exposure_layer(
            exposure_type, hurricane_df, forecast_time, config, cache_dir, use_cache=use_cache
        )

    # Plot exposure layer
    exposure.plot(output_dir=output_subdir)
    exposure_time = time.time() - exposure_start
    print(f"  ✓ Exposure layer completed in {exposure_time:.2f}s")

    # For flood, also plot the baseline
    if exposure_type == "flood":
        print(f"\n[Baseline Flood Extent]")
        exposure.plot_baseline(output_dir=output_subdir)

    # Create vulnerability layer with appropriate resolution context
    print(f"\n[Vulnerability Layer: {vuln_type}]")
    vuln_start = time.time()

    if exposure_type == "landslide":
        # Use visualization resolution for plotting
        vulnerability = get_vulnerability_layer(
            vuln_type, config, cache_dir, resolution_context="landslide_visualization", use_cache=use_cache
        )
    else:
        # Standard vulnerability layer
        vulnerability = get_vulnerability_layer(vuln_type, config, cache_dir, use_cache=use_cache)

    # Plot vulnerability layer
    vulnerability.plot(output_dir=output_subdir)
    vuln_time = time.time() - vuln_start
    print(f"  ✓ Vulnerability layer completed in {vuln_time:.2f}s")

    # Create impact layer
    print(f"\n[Impact Layer: {exposure_type} x {vuln_type} ({scenario})]")
    impact_start = time.time()
    impact = get_impact_layer(exposure, vulnerability, config)

    # Plot impact layer
    impact.plot(output_dir=output_subdir)
    print(f"\n[Binary Probability: {exposure_type} x {vuln_type} ({scenario})]")
    impact.plot_binary_probability(output_dir=output_subdir)

    # Only hurricanes have best/worst overlays
    if exposure_type == "hurricane":
        print(f"\n[Best/Worst Case Overlay Plots: {exposure_type} x {vuln_type}]")
        impact.plot_best_worst_case_overlay(output_dir=output_subdir)

    impact_time = time.time() - impact_start
    print(f"  ✓ Impact layer completed in {impact_time:.2f}s")

    # Handle ensemble analysis for landslide
    if exposure_type == "landslide":
        print(f"\n[Ensemble Analysis: {exposure_type} x {vuln_type}]")
        ensemble_start = time.time()
        print("Generating 50 ensemble members with spatial correlation...")

        # Create high-res vulnerability layer for computation/statistics
        vulnerability_comp = get_vulnerability_layer(
            vuln_type, config, cache_dir, resolution_context="landslide_computation"
        )

        # Get ensemble statistics
        stats = impact.exposure_layer.get_ensemble_impact(vulnerability_comp)
        ensemble_stats = impact.exposure_layer.get_best_worst_case(vulnerability_comp)
        mean_impact = np.mean(stats)
        std_impact = np.std(stats)
        min_impact = np.min(stats)
        max_impact = np.max(stats)

        # Calculate expected impact using high-res grids
        exposure_comp = get_exposure_layer(
            "landslide",
            None,  # hurricane_df
            forecast_time,
            config,
            cache_dir,
            resampling_method="mean",
            resolution_context="landslide_computation",
        )
        exposure_comp_grid = exposure_comp.compute_grid()
        vulnerability_comp_grid = vulnerability_comp.compute_grid()
        expected_impact = np.sum(
            exposure_comp_grid["probability"].values
            * vulnerability_comp_grid[vulnerability_comp.value_column].values
        )

        print(f"Ensemble Statistics:")
        print(f"  Expected Impact (mean probability): {expected_impact:.2f}")
        print(f"  Best Case (ensemble minimum): {min_impact:.2f}")
        print(f"  Worst Case (ensemble maximum): {max_impact:.2f}")
        print(f"  Ensemble Mean: {mean_impact:.2f}")
        print(f"  Ensemble Std Dev: {std_impact:.2f}")

        ensemble_time = time.time() - ensemble_start
        print(f"  ✓ Ensemble analysis completed in {ensemble_time:.2f}s")
    else:
        # Standard impact statistics for hurricane and flood
        print(f"\nExpected impact: {impact.expected_impact():.2f}")
        print(f"Best case (min): {impact.best_case():.2f}")
        print(f"Worst case (max): {impact.worst_case():.2f}")

    # Save impact summary
    impact.save_impact_summary(output_dir=output_subdir)

    total_time = time.time() - total_start_time
    print(f"\n✓ Total analysis completed in {total_time:.2f}s")

    return impact


def main():
    # Load configuration
    config_path = Path("config/impact_analysis_config.yaml")
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    config = load_config(str(config_path))

    # Get cache setting from config
    use_cache = get_config_value(config, "impact_analysis.cache.use_cache", True)
    
    if not use_cache:
        print("  [Cache disabled via config - will recompute all layers]")

    # Get output and cache directories
    output_dir = get_config_value(
        config, "impact_analysis.output.base_directory", "data/results/impact_analysis"
    )
    cache_dir = get_config_value(
        config,
        "impact_analysis.output.cache_directory",
        "data/results/impact_analysis/cache",
    )
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # Load hurricane data (for now, only hurricane exposure is supported)
    hurricane_data_path = get_config_value(
        config,
        "impact_analysis.input.hurricane_data.synthetic_file",
        "data/preprocessed/weatherlab/synthetic/processed_FNV3_2024_11_04_00_00_ensemble_data_synthetic.csv",
    )
    hurricane_file = get_data_path(hurricane_data_path)
    if not hurricane_file.exists():
        hurricane_data_path = get_config_value(
            config,
            "impact_analysis.input.hurricane_data.original_file",
            "data/preprocessed/weatherlab/processed/processed_FNV3_2024_11_04_00_00_ensemble_data.csv",
        )
        hurricane_file = get_data_path(hurricane_data_path)
    if not hurricane_file.exists():
        print(f"Error: Hurricane data file not found: {hurricane_file}")
        sys.exit(1)
    hurricane_df = pd.read_csv(hurricane_file)
    available_forecasts = hurricane_df["forecast_time"].unique()
    if len(available_forecasts) == 0:
        print("Error: No forecast times found in hurricane data")
        sys.exit(1)
    chosen_forecast = available_forecasts[0]
    print(f"Using forecast time: {chosen_forecast}")

    # Run all combos from config
    runs = config.get("impact_analysis", {}).get("runs", {})
    if not runs:
        print("No runs specified in config under impact_analysis.runs")
        sys.exit(1)

    for exposure_type, vulnerabilities in runs.items():
        for vuln_type in vulnerabilities:
            # Determine appropriate scenario for each exposure type
            if exposure_type == "flood":
                scenario = "ensemble"  # Flood uses ensemble variations
            elif exposure_type == "landslide":
                scenario = "ensemble"  # Landslide uses ensemble for visualization
            else:
                scenario = "mean"  # Hurricane uses mean scenario

            run_analysis(
                config,
                exposure_type,
                vuln_type,
                hurricane_df,
                chosen_forecast,
                output_dir,
                cache_dir,
                scenario=scenario,
                use_cache=use_cache,
            )
    print(f"\nImpact analysis complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()