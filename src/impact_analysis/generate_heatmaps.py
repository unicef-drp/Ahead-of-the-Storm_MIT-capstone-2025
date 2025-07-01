"""
Generate All Heatmaps CLI
Loads hurricane and school data, then generates hurricane grid, school grid, and combined risk heatmaps using the new modules.
"""

import pandas as pd
from datetime import datetime
from src.impact_analysis.hurricane_grid_heatmap import create_hurricane_heatmap
from src.impact_analysis.school_grid_heatmap import create_school_heatmap
from src.impact_analysis.combined_risk_heatmap import create_risk_heatmap
from src.utils.config_utils import load_config, get_config_value
from src.utils.logging_utils import setup_logging, get_logger
from src.utils.path_utils import ensure_directory, get_data_path


def main():
    # Setup logging
    logger = setup_logging(__name__)

    # Load impact analysis config
    config = load_config("config/impact_analysis_config.yaml")

    # Get configuration values
    grid_res = get_config_value(config, "impact_analysis.grid.resolution_degrees", 0.1)

    # Setup output directory
    base_dir = get_config_value(
        config, "impact_analysis.output.base_directory", "data/results/impact_analysis"
    )
    plots_dir = get_config_value(
        config, "impact_analysis.output.plots_directory", "plots"
    )
    output_dir = get_data_path(f"{base_dir}/{plots_dir}")
    ensure_directory(output_dir)

    # Get input file paths from config
    hurricane_config = get_config_value(
        config, "impact_analysis.input.hurricane_data", {}
    )
    synthetic_file = get_data_path(hurricane_config.get("synthetic_file"))
    original_file = get_data_path(hurricane_config.get("original_file"))

    # Load hurricane data (prefer synthetic, fallback to original)
    if synthetic_file.exists():
        logger.info(f"📁 Found synthetic hurricane data: {synthetic_file}")
        df = pd.read_csv(synthetic_file)
    elif original_file.exists():
        logger.info(f"📁 Found original hurricane data: {original_file}")
        df = pd.read_csv(original_file)
    else:
        logger.error("❌ No hurricane data found!")
        return

    # Parse datetime columns from config
    datetime_columns = get_config_value(
        config, "impact_analysis.data_processing.datetime_columns", []
    )
    for col in datetime_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    forecast_times = sorted(df["forecast_time"].unique())
    chosen_forecast = forecast_times[0]
    logger.info(f"\n🎯 Using forecast time: {chosen_forecast}")

    # Generate heatmaps
    hurricane_grid = create_hurricane_heatmap(
        df, chosen_forecast, str(output_dir), grid_res, config
    )
    school_grid = create_school_heatmap(str(output_dir), grid_res, config)

    if hurricane_grid is not None and school_grid is not None:
        risk_grid, expected_affected_schools = create_risk_heatmap(
            hurricane_grid,
            school_grid,
            str(output_dir),
            grid_res,
            forecast_time=chosen_forecast,
            config=config,
        )

        if expected_affected_schools > 0:
            logger.info(
                f"📊 Total Expected Schools Affected: {expected_affected_schools:.2f}"
            )

    logger.info("\n✅ All heatmaps generated (config-driven test run)")


if __name__ == "__main__":
    main()
