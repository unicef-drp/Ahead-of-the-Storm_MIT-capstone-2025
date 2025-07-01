#!/usr/bin/env python3
"""
Plot Forecast-Specific Hurricane Dashboards
Generate separate dashboard visualizations for all 6-hour forecast intervals from Nov 4-10.
Each dashboard shows all ensemble members for that specific forecast time (00:00, 06:00, 12:00, 18:00 UTC).
"""

import logging
from datetime import datetime, timedelta
import pandas as pd
import os

from src.data_visualization.hurricane_helper.hurricane_analyzer import HurricaneAnalyzer
from src.data_visualization.hurricane_helper.hurricane_visualizer import (
    HurricaneVisualizer,
)


def main():
    """Generate forecast-specific dashboard visualizations."""

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    print("=" * 70)
    print("FORECAST-SPECIFIC HURRICANE DASHBOARD VISUALIZATIONS")
    print("Google Weather Lab FNV3 Ensemble Model")
    print("November 4-10, 2024 - All 6-Hour Forecast Intervals")
    print("=" * 70)

    # Initialize analyzer and visualizer
    logger.info("Initializing hurricane analyzer and visualizer...")
    analyzer = HurricaneAnalyzer()
    visualizer = HurricaneVisualizer()

    # Load all processed data
    logger.info("Loading all processed hurricane data...")
    df = analyzer.load_all_data()

    if len(df) == 0:
        logger.error("No data found!")
        return

    # Filter for Hurricane Rafael
    logger.info("Filtering data for Hurricane Rafael (AL182024)...")
    rafael_data = df[df["track_id"] == "AL182024"].copy()

    # Ensure forecast_time is datetime
    if rafael_data["forecast_time"].dtype == "object":
        rafael_data["forecast_time"] = pd.to_datetime(
            rafael_data["forecast_time"], format="mixed"
        )

    if len(rafael_data) == 0:
        logger.error("No data found for Hurricane Rafael (AL182024)")
        return

    print(f"\n🌪️  Hurricane Rafael Data:")
    print(f"   Records: {len(rafael_data):,}")
    print(f"   Forecast dates: {len(rafael_data['forecast_time'].dt.date.unique())}")
    print(f"   Ensemble members: {rafael_data['ensemble_member'].nunique()}")

    # Define all 6-hour forecast times for each day
    forecast_times = []
    for day in range(4, 11):  # November 4-10
        for hour in [0, 6, 12, 18]:  # 00:00, 06:00, 12:00, 18:00 UTC
            forecast_times.append(datetime(2024, 11, day, hour, 0))

    print(f"\n📅 Generating visualizations for {len(forecast_times)} forecast times:")
    for i, ft in enumerate(forecast_times, 1):
        print(f"   {i:2d}. {ft.strftime('%B %d, %Y %H:%M')} UTC")

    # Generate dashboard for each forecast time
    for i, forecast_time in enumerate(forecast_times, 1):
        print(f"\n{'='*60}")
        print(f"📊 GENERATING DASHBOARD {i}/{len(forecast_times)}")
        print(f"📅 Date: {forecast_time.strftime('%B %d, %Y %H:%M')} UTC")
        print(f"{'='*60}")

        # Filter data for this specific forecast time
        daily_data = rafael_data[
            (rafael_data["forecast_time"].dt.date == forecast_time.date())
            & (rafael_data["forecast_time"].dt.time == forecast_time.time())
        ].copy()

        if len(daily_data) == 0:
            print(
                f"⚠️  No data found for {forecast_time.strftime('%B %d, %Y %H:%M')} UTC"
            )
            continue

        print(f"📈 Data points: {len(daily_data):,}")
        print(f"🎲 Ensemble members: {daily_data['ensemble_member'].nunique()}")
        print(
            f"⏰ Lead times: {daily_data['lead_time'].nunique()} (0-{daily_data['lead_time'].max()} hours)"
        )

        # Generate ensemble tracks plot for this forecast time
        print(f"📈 Generating ensemble tracks...")
        tracks_path = visualizer.plot_ensemble_tracks(
            df=daily_data, forecast_date=forecast_time, save_plot=True, show_plot=False
        )

        # Generate intensity curves for this forecast time
        print(f"📊 Generating intensity curves...")
        intensity_path = visualizer.plot_intensity_curves(
            df=daily_data, forecast_date=forecast_time, save_plot=True, show_plot=False
        )

        # Generate ensemble spread analysis for this forecast time
        print(f"📊 Generating ensemble spread analysis...")
        spread_path = visualizer.plot_ensemble_spread(
            df=daily_data, forecast_date=forecast_time, save_plot=True, show_plot=False
        )

        # Generate summary dashboard for this forecast time
        print(f"📋 Generating summary dashboard...")
        dashboard_path = visualizer.create_summary_dashboard(
            df=daily_data, forecast_date=forecast_time, save_plot=True, show_plot=False
        )

        # Show some statistics for this forecast time
        max_wind = daily_data["wind_speed"].max()
        min_pressure = daily_data["pressure"].min()
        avg_spread = daily_data.groupby("lead_time")["wind_speed"].std().mean()

        print(f"\n📊 Forecast Statistics:")
        print(f"   Max wind speed: {max_wind:.1f} knots")
        print(f"   Min pressure: {min_pressure:.1f} hPa")
        print(f"   Average ensemble spread: {avg_spread:.1f} knots")
        print(f"   Files saved:")
        print(f"     • Tracks: {os.path.basename(tracks_path)}")
        print(f"     • Intensity: {os.path.basename(intensity_path)}")
        print(f"     • Spread: {os.path.basename(spread_path)}")
        print(f"     • Dashboard: {os.path.basename(dashboard_path)}")

    print("\n{'='*70}")
    print("✅ ALL FORECAST DASHBOARDS COMPLETED!")
    print(f"{'='*70}")
    print("📁 All plots saved in: data/results/weatherlab/plots/")
    print(f"📊 Generated {len(forecast_times)} forecast dashboard sets")
    print(
        "🎲 Each dashboard shows all ensemble members for that specific forecast time"
    )
    print(
        "⏰ Covers all 6-hour intervals (00:00, 06:00, 12:00, 18:00 UTC) from Nov 4-10, 2024"
    )
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
