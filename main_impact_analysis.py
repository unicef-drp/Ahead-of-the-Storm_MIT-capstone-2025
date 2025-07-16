#!/usr/bin/env python3
"""
Main CLI script for running hurricane impact analysis.
This script loads configuration, hurricane data, and runs all vulnerability analyses.
"""

import os
import sys
import yaml
import pandas as pd
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.impact_analysis.main_impact_analysis import main_impact_analysis
from src.impact_analysis.layers.hurricane import HurricaneExposureLayer
from src.utils.config_utils import load_config, get_config_value
from src.utils.path_utils import get_data_path


def main():
    """Main CLI function for hurricane impact analysis."""

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

    # Create directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # Load hurricane data
    print("Loading hurricane data...")
    hurricane_data_path = get_config_value(
        config,
        "impact_analysis.input.hurricane_data.synthetic_file",
        "data/preprocessed/weatherlab/synthetic/processed_FNV3_2024_11_04_00_00_ensemble_data_synthetic.csv",
    )
    hurricane_file = get_data_path(hurricane_data_path)

    if not hurricane_file.exists():
        # Try original file as fallback
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

    # Select a forecast time (use the first available)
    available_forecasts = hurricane_df["forecast_time"].unique()
    if len(available_forecasts) == 0:
        print("Error: No forecast times found in hurricane data")
        sys.exit(1)

    chosen_forecast = available_forecasts[0]
    print(f"Using forecast time: {chosen_forecast}")

    # Create exposure layer
    exposure = HurricaneExposureLayer(hurricane_df, chosen_forecast, config, cache_dir)

    # Run all impact analyses
    print("Running impact analysis...")
    main_impact_analysis(config, exposure, output_dir, cache_dir)

    print(f"\nImpact analysis complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
