import os
import sys
import yaml
import pandas as pd
from pathlib import Path

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
):
    # For hurricanes, ignore scenario in folder name
    if exposure_type == "hurricane":
        output_subdir = ensure_subdir(output_dir, f"hurricane_{vuln_type}")
    else:
        output_subdir = ensure_subdir(
            output_dir, f"{exposure_type}_{vuln_type}_{scenario}"
        )

    # Create exposure, vulnerability, and impact objects
    exposure = get_exposure_layer(
        exposure_type, hurricane_df, forecast_time, config, cache_dir, scenario=scenario
    )
    vulnerability = get_vulnerability_layer(vuln_type, config, cache_dir)
    impact = get_impact_layer(exposure, vulnerability, config)

    # Run and plot all layers
    print(f"\n[Exposure Layer: {exposure_type} ({scenario})]")
    exposure.plot(output_dir=output_subdir)
    print(f"\n[Vulnerability Layer: {vuln_type}]")
    vulnerability.plot(output_dir=output_subdir)
    print(f"\n[Impact Layer: {exposure_type} x {vuln_type} ({scenario})]")
    impact.plot(output_dir=output_subdir)
    print(f"\n[Binary Probability: {exposure_type} x {vuln_type} ({scenario})]")
    impact.plot_binary_probability(output_dir=output_subdir)
    # Only hurricanes have best/worst overlays
    if exposure_type == "hurricane":
        print(f"\n[Best/Worst Case Overlay Plots: {exposure_type} x {vuln_type}]")
        impact.plot_best_worst_case_overlay(output_dir=output_subdir)
    print(f"\nExpected impact: {impact.expected_impact():.2f}")
    print(f"Best case (min): {impact.best_case():.2f}")
    print(f"Worst case (max): {impact.worst_case():.2f}")
    impact.save_impact_summary(output_dir=output_subdir)


def run_landslide_analysis(
    config, vuln_type, hurricane_df, forecast_time, output_dir, cache_dir, scenarios
):
    # Output subdir naming
    output_subdir = ensure_subdir(output_dir, f"landslide_{vuln_type}")
    metrics = {}
    exposure_layers = {}
    impact_layers = {}

    # 1. Compute and plot all 3 exposure scenarios
    for scenario in scenarios:
        exposure = get_exposure_layer(
            "landslide",
            hurricane_df,
            forecast_time,
            config,
            cache_dir,
            scenario=scenario,
        )
        exposure_layers[scenario] = exposure
        print(f"\n[Exposure Layer: landslide ({scenario})]")
        exposure.plot(output_dir=output_subdir)

    # 2. Compute and plot vulnerability once
    vulnerability = get_vulnerability_layer(vuln_type, config, cache_dir)
    print(f"\n[Vulnerability Layer: {vuln_type}]")
    vulnerability.plot(output_dir=output_subdir)

    # 3. Compute and plot all 3 impact scenarios
    for scenario in scenarios:
        exposure = exposure_layers[scenario]
        impact = get_impact_layer(exposure, vulnerability, config)
        impact_layers[scenario] = impact
        print(f"\n[Impact Layer: landslide x {vuln_type} ({scenario})]")
        impact.plot(output_dir=output_subdir)
        impact.plot_binary_probability(output_dir=output_subdir)
        # Collect metrics
        metrics[scenario] = impact.expected_impact()

    # 4. Write a single summary txt with avg, best, worst metrics
    summary = (
        f"Impact summary for landslide_{vuln_type}\n"
        f"Vulnerability type: {vuln_type}\n"
        f"Scenarios: {', '.join(scenarios)}\n"
        f"Expected impact (avg): {metrics['mean']:.2f}\n"
        f"Best case (min): {metrics['min']:.2f}\n"
        f"Worst case (max): {metrics['max']:.2f}\n"
    )
    out_path = os.path.join(output_subdir, f"impact_summary_landslide_{vuln_type}.txt")
    with open(out_path, "w") as f:
        f.write(summary)
    print(f"Saved impact summary: {out_path}")


def main():
    # Load configuration
    config_path = Path("config/impact_analysis_config.yaml")
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    config = load_config(str(config_path))

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
    runs = config.get("impact_analysis", {}).get("runs", [])
    if not runs:
        print("No runs specified in config under impact_analysis.runs")
        sys.exit(1)
    for run in runs:
        exposure_type = run["exposure"]
        scenarios = run.get("scenarios", ["mean"])
        for vuln_type in run["vulnerabilities"]:
            if exposure_type == "landslide":
                print(
                    f"\n=== Running landslide analysis: Vulnerability={vuln_type} ==="
                )
                run_landslide_analysis(
                    config,
                    vuln_type,
                    hurricane_df,
                    chosen_forecast,
                    output_dir,
                    cache_dir,
                    scenarios,
                )
            else:
                print(
                    f"\n=== Running analysis: Exposure={exposure_type}, Vulnerability={vuln_type} ==="
                )
                run_analysis(
                    config,
                    exposure_type,
                    vuln_type,
                    hurricane_df,
                    chosen_forecast,
                    output_dir,
                    cache_dir,
                )
    print(f"\nImpact analysis complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
