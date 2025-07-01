#!/usr/bin/env python3
"""
Hurricane Data Analysis and Visualization Script

This script performs comprehensive analysis and visualization of hurricane forecast data
from Google Weather Lab's FNV3 model, including ensemble tracks, intensity curves,
and statistical summaries.
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_prep.hurricane_helper.hurricane_analyzer import HurricaneAnalyzer
from data_prep.hurricane_helper.hurricane_visualizer import HurricaneVisualizer


def setup_logging():
    """Setup logging for the analysis script."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"hurricane_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def main():
    """Main analysis and visualization function."""
    logger = setup_logging()
    
    print("=" * 60)
    print("HURRICANE RAFAEL (2024) ANALYSIS & VISUALIZATION")
    print("Google Weather Lab FNV3 Ensemble Model")
    print("=" * 60)
    
    try:
        # Initialize analyzer and visualizer
        logger.info("Initializing hurricane analyzer and visualizer...")
        analyzer = HurricaneAnalyzer()
        visualizer = HurricaneVisualizer()
        
        # Load all processed data
        logger.info("Loading all processed hurricane data...")
        df = analyzer.load_all_data()
        
        print(f"\n📊 Data Summary:")
        print(f"   Total records: {len(df):,}")
        print(f"   Date range: {df['forecast_time'].min()} to {df['forecast_time'].max()}")
        print(f"   Ensemble members: {len(df['ensemble_member'].unique())}")
        print(f"   Lead times: {len(df['lead_time'].unique())} (0-{df['lead_time'].max():.0f} hours)")
        
        # Get unique forecast dates
        forecast_dates = analyzer.extract_forecast_dates(df)
        print(f"   Forecast dates: {len(forecast_dates)}")
        
        # Filter data for Hurricane Rafael (AL182024)
        logger.info("Filtering data for Hurricane Rafael (AL182024)...")
        rafael_data = df[df['track_id'] == 'AL182024'].copy()
        
        if len(rafael_data) == 0:
            logger.warning("No data found for Hurricane Rafael (AL182024)")
            print("\n⚠️  No data found for Hurricane Rafael (AL182024)")
            print("   This might indicate the hurricane didn't form during the forecast period")
            return
        
        print(f"\n🌪️  Hurricane Rafael Data:")
        print(f"   Records: {len(rafael_data):,}")
        print(f"   Forecast dates: {len(rafael_data['forecast_time'].dt.date.unique())}")
        print(f"   Max wind speed: {rafael_data['wind_speed'].max():.1f} knots")
        print(f"   Min pressure: {rafael_data['pressure'].min():.1f} hPa")
        
        # Generate visualizations
        logger.info("Generating hurricane visualizations...")
        
        # 1. Ensemble tracks plot
        print("\n📈 Generating ensemble tracks plot...")
        tracks_plot = visualizer.plot_ensemble_tracks(
            rafael_data, 
            save_plot=True, 
            show_plot=False
        )
        print(f"   ✅ Saved: {tracks_plot}")
        
        # 2. Intensity curves (wind speed and pressure)
        print("\n📊 Generating intensity curves...")
        intensity_plot = visualizer.plot_intensity_curves(
            rafael_data,
            variable='wind_speed',
            save_plot=True,
            show_plot=False
        )
        print(f"   ✅ Saved: {intensity_plot}")
        
        # 3. Ensemble spread analysis
        print("\n📊 Generating ensemble spread analysis...")
        spread_plot = visualizer.plot_ensemble_spread(
            rafael_data,
            save_plot=True,
            show_plot=False
        )
        print(f"   ✅ Saved: {spread_plot}")
        
        # 4. Summary dashboard
        print("\n📋 Generating summary dashboard...")
        dashboard_plot = visualizer.create_summary_dashboard(
            rafael_data,
            save_plot=True,
            show_plot=False
        )
        print(f"   ✅ Saved: {dashboard_plot}")
        
        # 5. Track statistics
        print("\n📈 Calculating track statistics...")
        track_stats = analyzer.calculate_track_statistics(rafael_data)
        
        print(f"\n📊 Track Statistics:")
        print(f"   Average track length: {track_stats.get('avg_track_length', 0):.1f} hours")
        print(f"   Max wind speed: {track_stats.get('max_wind_speed', 0):.1f} knots")
        print(f"   Min pressure: {track_stats.get('min_pressure', 0):.1f} hPa")
        print(f"   Average ensemble spread: {track_stats.get('avg_ensemble_spread', 0):.1f} km")
        
        # 6. Save analysis results
        print("\n💾 Saving analysis results...")
        analysis_file = analyzer.save_analysis_results(
            rafael_data, 
            "hurricane_rafael_analysis.csv",
            "hurricane_tracks"
        )
        print(f"   ✅ Saved: {analysis_file}")
        
        print("\n" + "=" * 60)
        print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"📁 Plots saved in: {visualizer.output_dir}")
        print(f"📊 Analysis results: {analysis_file}")
        print(f"📈 Generated visualizations:")
        print(f"   • Ensemble tracks")
        print(f"   • Intensity curves")
        print(f"   • Ensemble spread analysis")
        print(f"   • Summary dashboard")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        print(f"\n❌ Error during analysis: {e}")
        raise


if __name__ == "__main__":
    main() 