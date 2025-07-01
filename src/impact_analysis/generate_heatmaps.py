"""
Generate All Heatmaps CLI
Loads hurricane and school data, then generates hurricane grid, school grid, and combined risk heatmaps using the new modules.
"""

import os
import pandas as pd
from datetime import datetime
from src.impact_analysis.hurricane_grid_heatmap import create_hurricane_heatmap
from src.impact_analysis.school_grid_heatmap import create_school_heatmap
from src.impact_analysis.combined_risk_heatmap import create_risk_heatmap
import yaml


def main():
    # Load config
    with open("config/synthetic_hurricane_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    grid_res = config["synthetic_hurricane"]["grid_heatmap"]["resolution_degrees"]
    output_dir = "data/results/weatherlab/plots"
    os.makedirs(output_dir, exist_ok=True)
    # Input files (match original logic)
    synthetic_file = "data/preprocessed/weatherlab/synthetic/processed_FNV3_2024_11_04_00_00_ensemble_data_synthetic.csv"
    original_file = "data/preprocessed/weatherlab/processed/processed_FNV3_2024_11_04_00_00_ensemble_data.csv"
    if os.path.exists(synthetic_file):
        print(f"📁 Found synthetic hurricane data: {synthetic_file}")
        df = pd.read_csv(synthetic_file)
    elif os.path.exists(original_file):
        print(f"📁 Found original hurricane data: {original_file}")
        df = pd.read_csv(original_file)
    else:
        print("❌ No hurricane data found!")
        return
    df["forecast_time"] = pd.to_datetime(df["forecast_time"])
    df["valid_time"] = pd.to_datetime(df["valid_time"])
    forecast_times = sorted(df["forecast_time"].unique())
    chosen_forecast = forecast_times[0]
    print(f"\n🎯 Using forecast time: {chosen_forecast}")
    hurricane_grid = create_hurricane_heatmap(df, chosen_forecast, output_dir, grid_res)
    school_grid = create_school_heatmap(output_dir, grid_res)
    if hurricane_grid is not None and school_grid is not None:
        risk_grid = create_risk_heatmap(
            hurricane_grid,
            school_grid,
            output_dir,
            grid_res,
            forecast_time=chosen_forecast,
        )
    print("\n✅ All heatmaps generated (config-driven test run)")


if __name__ == "__main__":
    main()
