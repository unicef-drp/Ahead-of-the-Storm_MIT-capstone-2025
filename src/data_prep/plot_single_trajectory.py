#!/usr/bin/env python3
"""
Plot Single Hurricane Trajectory
Generate a single trajectory plot for one ensemble member from one forecast time.
"""

import sys
import os
import logging
from datetime import datetime
import pandas as pd

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from hurricane_helper.hurricane_analyzer import HurricaneAnalyzer
from hurricane_helper.hurricane_visualizer import HurricaneVisualizer

def main():
    """Generate a single trajectory plot."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    print("=" * 60)
    print("SINGLE HURRICANE TRAJECTORY PLOT")
    print("Google Weather Lab FNV3 Ensemble Model")
    print("=" * 60)
    
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
    rafael_data = df[df['track_id'] == 'AL182024'].copy()
    
    # Ensure forecast_time is datetime
    if rafael_data['forecast_time'].dtype == 'object':
        rafael_data['forecast_time'] = pd.to_datetime(rafael_data['forecast_time'], format='mixed')

    if len(rafael_data) == 0:
        logger.error("No data found for Hurricane Rafael (AL182024)")
        return
    
    print(f"\n🌪️  Hurricane Rafael Data:")
    print(f"   Records: {len(rafael_data):,}")
    print(f"   Forecast dates: {len(rafael_data['forecast_time'].dt.date.unique())}")
    print(f"   Ensemble members: {rafael_data['ensemble_member'].nunique()}")
    
    # Get available forecast times
    forecast_times = rafael_data['forecast_time'].unique()
    forecast_times = sorted(list(forecast_times))
    
    print(f"\n📅 Available forecast times:")
    for i, ft in enumerate(forecast_times[:10]):  # Show first 10
        print(f"   {i+1:2d}. {ft}")
    if len(forecast_times) > 10:
        print(f"   ... and {len(forecast_times) - 10} more")
    
    # Choose a forecast time (Nov 5, 2024, 00:00:00)
    chosen_forecast = pd.Timestamp('2024-11-05 00:00:00')
    ensemble_member = 10  # Use ensemble member 10
    
    print(f"\n🎯 Plotting single trajectory:")
    print(f"   Forecast time: {chosen_forecast}")
    print(f"   Ensemble member: {ensemble_member}")
    
    # Generate the single trajectory plot
    logger.info("Generating single trajectory plot...")
    plot_path = visualizer.plot_single_trajectory(
        df=rafael_data,
        forecast_time=chosen_forecast,
        ensemble_member=ensemble_member,
        save_plot=True,
        show_plot=False
    )
    
    if plot_path:
        print(f"\n✅ Single trajectory plot saved:")
        print(f"   {plot_path}")
        
        # Show some stats about this trajectory
        trajectory_data = rafael_data[
            (rafael_data['forecast_time'] == chosen_forecast) & 
            (rafael_data['ensemble_member'] == ensemble_member)
        ].copy()
        
        if len(trajectory_data) > 0:
            trajectory_data = trajectory_data.sort_values('valid_time')
            start_point = trajectory_data.iloc[0]
            end_point = trajectory_data.iloc[-1]
            
            print(f"\n📊 Trajectory Statistics:")
            print(f"   Duration: {len(trajectory_data)} forecast points")
            print(f"   Start: {start_point['valid_time']} at ({start_point['latitude']:.2f}°N, {start_point['longitude']:.2f}°E)")
            print(f"   End: {end_point['valid_time']} at ({end_point['latitude']:.2f}°N, {end_point['longitude']:.2f}°E)")
            print(f"   Max wind speed: {trajectory_data['wind_speed'].max():.1f} knots")
            print(f"   Min pressure: {trajectory_data['pressure'].min():.1f} hPa")
    else:
        print(f"\n❌ Failed to generate single trajectory plot")
    
    print("\n" + "=" * 60)
    print("✅ SINGLE TRAJECTORY PLOT COMPLETED!")
    print("=" * 60)

if __name__ == "__main__":
    main() 