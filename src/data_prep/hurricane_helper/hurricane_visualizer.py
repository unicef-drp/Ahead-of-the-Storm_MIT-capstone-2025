"""
Hurricane Data Visualizer for FNV3 Ensemble Data

This module provides functionality to create visualizations of hurricane forecast data
from Google Weather Lab's FNV3 model, including ensemble tracks and intensity plots.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime, timedelta
import warnings
import yaml
import matplotlib.dates as mdates

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")


class HurricaneVisualizer:
    """
    Creates visualizations of hurricane forecast data from FNV3 ensemble model.
    
    This class provides methods to create various plots including ensemble tracks,
    intensity curves, and statistical summaries.
    """
    
    def __init__(self, config_path: str = "config/hurricane_config.yaml"):
        """
        Initialize the hurricane visualizer.
        
        Args:
            config_path: Path to the configuration file
        """
        # Get project root (3 levels up from this file: hurricane_helper/data_prep/src/)
        current_file = os.path.abspath(__file__)
        self.project_root = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file)))))
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._setup_directories()
        
        # Set up color schemes
        self._setup_colors()
    
    def _setup_colors(self):
        """Setup color schemes for different plot types."""
        # Color scheme for ensemble members
        self.ensemble_colors = plt.cm.tab20(np.linspace(0, 1, 20))
        
        # Color scheme for intensity (wind speed)
        self.intensity_colors = {
            'tropical_depression': '#00ff00',  # Green
            'tropical_storm': '#ffff00',       # Yellow
            'category_1': '#ff8000',           # Orange
            'category_2': '#ff4000',           # Red-Orange
            'category_3': '#ff0000',           # Red
            'category_4': '#c000c0',           # Purple
            'category_5': '#800080'            # Dark Purple
        }
        
        # Color scheme for lead times
        self.lead_time_colors = plt.cm.viridis(np.linspace(0, 1, 10))
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        config_file = self.project_root / config_path
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = self.project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"hurricane_visualizer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Hurricane Visualizer initialized")
    
    def _setup_directories(self):
        """Create necessary directories if they don't exist."""
        # Create plots directory
        self.output_dir = self.project_root / "data" / "weatherlab" / "plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Output directory: {self.output_dir}")
    
    def plot_ensemble_tracks(self, df: pd.DataFrame, forecast_date: Optional[datetime] = None,
                           save_plot: bool = True, show_plot: bool = False) -> str:
        """
        Plot ensemble hurricane tracks.
        
        Args:
            df: Hurricane data DataFrame
            forecast_date: Optional forecast date to filter by
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Path to saved plot file
        """
        # Filter by forecast date if provided
        if forecast_date:
            df = df[df['forecast_time'].dt.date == forecast_date.date()]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get unique ensemble members
        ensemble_members = df['ensemble_member'].unique()
        
        # Plot each ensemble member track
        for i, member in enumerate(ensemble_members):
            member_data = df[df['ensemble_member'] == member].sort_values('lead_time')
            
            if len(member_data) > 1:
                # Plot track line
                ax.plot(member_data['longitude'], member_data['latitude'], 
                       color=self.ensemble_colors[i % len(self.ensemble_colors)],
                       alpha=0.7, linewidth=1.5, label=f'Member {member}')
                
                # Plot start point
                start_point = member_data.iloc[0]
                ax.scatter(start_point['longitude'], start_point['latitude'],
                          color=self.ensemble_colors[i % len(self.ensemble_colors)],
                          s=50, marker='o', edgecolors='black', linewidth=1)
                
                # Plot end point
                end_point = member_data.iloc[-1]
                ax.scatter(end_point['longitude'], end_point['latitude'],
                          color=self.ensemble_colors[i % len(self.ensemble_colors)],
                          s=50, marker='s', edgecolors='black', linewidth=1)
        
        # Customize plot
        ax.set_xlabel('Longitude (°E)', fontsize=12)
        ax.set_ylabel('Latitude (°N)', fontsize=12)
        
        title = f"FNV3 Ensemble Hurricane Tracks"
        if forecast_date:
            title += f" - Forecast: {forecast_date.strftime('%Y-%m-%d %H:%M UTC')}"
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend (only show first 10 members to avoid clutter)
        if len(ensemble_members) <= 10:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax.text(0.02, 0.98, f'{len(ensemble_members)} ensemble members',
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"ensemble_tracks"
        if forecast_date:
            plot_filename += f"_{forecast_date.strftime('%Y%m%d_%H%M')}"
        plot_filename += ".png"
        plot_path = self.output_dir / plot_filename
        
        if save_plot:
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved ensemble tracks plot: {plot_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return str(plot_path)
    
    def plot_intensity_curves(self, df: pd.DataFrame, forecast_date: Optional[datetime] = None,
                            variable: str = 'wind_speed', save_plot: bool = True, 
                            show_plot: bool = False) -> str:
        """
        Plot intensity curves for wind speed or pressure, using valid_time as x-axis.
        """
        # Filter by forecast date if provided
        if forecast_date:
            df = df[df['forecast_time'].dt.date == forecast_date.date()]
        # Ensure valid_time is datetime
        if df['valid_time'].dtype == 'object':
            df['valid_time'] = pd.to_datetime(df['valid_time'], format='mixed')
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        # Get unique ensemble members
        ensemble_members = df['ensemble_member'].unique()
        # Plot individual ensemble member curves
        for i, member in enumerate(ensemble_members):
            member_data = df[df['ensemble_member'] == member].sort_values('valid_time')
            if len(member_data) > 1:
                color = self.ensemble_colors[i % len(self.ensemble_colors)]
                # Plot wind speed
                if 'wind_speed' in member_data.columns:
                    ax1.plot(member_data['valid_time'], member_data['wind_speed'],
                            color=color, alpha=0.7, linewidth=1.5, label=f'Member {member}')
                # Plot pressure
                if 'pressure' in member_data.columns:
                    ax2.plot(member_data['valid_time'], member_data['pressure'],
                            color=color, alpha=0.7, linewidth=1.5, label=f'Member {member}')
        # Calculate and plot ensemble statistics
        if len(ensemble_members) > 1:
            stats_data = df.groupby('valid_time').agg({
                'wind_speed': ['mean', 'std'],
                'pressure': ['mean', 'std']
            }).reset_index()
            # Flatten column names
            stats_data.columns = ['valid_time', 'wind_mean', 'wind_std', 'pressure_mean', 'pressure_std']
            # Plot ensemble mean and spread for wind speed
            if 'wind_mean' in stats_data.columns:
                ax1.plot(stats_data['valid_time'], stats_data['wind_mean'],
                        color='black', linewidth=3, label='Ensemble Mean')
                ax1.fill_between(stats_data['valid_time'],
                               stats_data['wind_mean'] - stats_data['wind_std'],
                               stats_data['wind_mean'] + stats_data['wind_std'],
                               color='black', alpha=0.2, label='±1σ Spread')
            # Plot ensemble mean and spread for pressure
            if 'pressure_mean' in stats_data.columns:
                ax2.plot(stats_data['valid_time'], stats_data['pressure_mean'],
                        color='black', linewidth=3, label='Ensemble Mean')
                ax2.fill_between(stats_data['valid_time'],
                               stats_data['pressure_mean'] - stats_data['pressure_std'],
                               stats_data['pressure_mean'] + stats_data['pressure_std'],
                               color='black', alpha=0.2, label='±1σ Spread')
        # Customize wind speed plot
        ax1.set_xlabel('Valid Time (UTC)', fontsize=12)
        ax1.set_ylabel('Wind Speed (kt)', fontsize=12)
        ax1.set_title('Hurricane Wind Speed Forecast', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        # Add hurricane category lines
        category_winds = [34, 64, 83, 96, 113, 137]
        category_labels = ['TS', 'Cat 1', 'Cat 2', 'Cat 3', 'Cat 4', 'Cat 5']
        for wind, label in zip(category_winds, category_labels):
            ax1.axhline(y=wind, color='gray', linestyle='--', alpha=0.5)
            ax1.text(ax1.get_xlim()[1], wind, f' {label}', verticalalignment='bottom')
        # Customize pressure plot
        ax2.set_xlabel('Valid Time (UTC)', fontsize=12)
        ax2.set_ylabel('Pressure (mb)', fontsize=12)
        ax2.set_title('Hurricane Pressure Forecast', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.invert_yaxis()  # Lower pressure at top
        # Add title for the whole figure
        title = f"FNV3 Ensemble Intensity Forecast"
        if forecast_date:
            title += f" - Forecast: {forecast_date.strftime('%Y-%m-%d %H:%M UTC')}"
        fig.suptitle(title, fontsize=16, fontweight='bold')
        # Adjust layout
        plt.tight_layout()
        # Save plot
        plot_filename = f"intensity_curves"
        if forecast_date:
            plot_filename += f"_{forecast_date.strftime('%Y%m%d_%H%M')}"
        plot_filename += ".png"
        plot_path = self.output_dir / plot_filename
        if save_plot:
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved intensity curves plot: {plot_path}")
        if show_plot:
            plt.show()
        else:
            plt.close()
        return str(plot_path)
    
    def plot_ensemble_spread(self, df: pd.DataFrame, forecast_date: Optional[datetime] = None,
                           save_plot: bool = True, show_plot: bool = False) -> str:
        """
        Plot ensemble spread statistics using valid_time as x-axis.
        """
        # Filter by forecast date if provided
        if forecast_date:
            df = df[df['forecast_time'].dt.date == forecast_date.date()]
        # Ensure valid_time is datetime
        if df['valid_time'].dtype == 'object':
            df['valid_time'] = pd.to_datetime(df['valid_time'], format='mixed')
        # Calculate ensemble statistics
        stats_data = df.groupby('valid_time').agg({
            'latitude': ['mean', 'std', 'min', 'max'],
            'longitude': ['mean', 'std', 'min', 'max'],
            'wind_speed': ['mean', 'std', 'min', 'max'],
            'pressure': ['mean', 'std', 'min', 'max']
        }).reset_index()
        # Flatten column names
        stats_data.columns = ['valid_time', 'lat_mean', 'lat_std', 'lat_min', 'lat_max',
                             'lon_mean', 'lon_std', 'lon_min', 'lon_max',
                             'wind_mean', 'wind_std', 'wind_min', 'wind_max',
                             'pressure_mean', 'pressure_std', 'pressure_min', 'pressure_max']
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        # Plot latitude spread
        ax1.fill_between(stats_data['valid_time'], 
                        stats_data['lat_mean'] - stats_data['lat_std'],
                        stats_data['lat_mean'] + stats_data['lat_std'],
                        alpha=0.3, color='blue', label='±1σ')
        ax1.fill_between(stats_data['valid_time'],
                        stats_data['lat_min'], stats_data['lat_max'],
                        alpha=0.1, color='blue', label='Min-Max')
        ax1.plot(stats_data['valid_time'], stats_data['lat_mean'], 
                color='blue', linewidth=2, label='Mean')
        ax1.set_xlabel('Valid Time (UTC)')
        ax1.set_ylabel('Latitude (°N)')
        ax1.set_title('Latitude Spread')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        # Plot longitude spread
        ax2.fill_between(stats_data['valid_time'],
                        stats_data['lon_mean'] - stats_data['lon_std'],
                        stats_data['lon_mean'] + stats_data['lon_std'],
                        alpha=0.3, color='red', label='±1σ')
        ax2.fill_between(stats_data['valid_time'],
                        stats_data['lon_min'], stats_data['lon_max'],
                        alpha=0.1, color='red', label='Min-Max')
        ax2.plot(stats_data['valid_time'], stats_data['lon_mean'],
                color='red', linewidth=2, label='Mean')
        ax2.set_xlabel('Valid Time (UTC)')
        ax2.set_ylabel('Longitude (°E)')
        ax2.set_title('Longitude Spread')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        # Plot wind speed spread
        ax3.fill_between(stats_data['valid_time'],
                        stats_data['wind_mean'] - stats_data['wind_std'],
                        stats_data['wind_mean'] + stats_data['wind_std'],
                        alpha=0.3, color='green', label='±1σ')
        ax3.fill_between(stats_data['valid_time'],
                        stats_data['wind_min'], stats_data['wind_max'],
                        alpha=0.1, color='green', label='Min-Max')
        ax3.plot(stats_data['valid_time'], stats_data['wind_mean'],
                color='green', linewidth=2, label='Mean')
        ax3.set_xlabel('Valid Time (UTC)')
        ax3.set_ylabel('Wind Speed (kt)')
        ax3.set_title('Wind Speed Spread')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        # Plot pressure spread
        ax4.fill_between(stats_data['valid_time'],
                        stats_data['pressure_mean'] - stats_data['pressure_std'],
                        stats_data['pressure_mean'] + stats_data['pressure_std'],
                        alpha=0.3, color='purple', label='±1σ')
        ax4.fill_between(stats_data['valid_time'],
                        stats_data['pressure_min'], stats_data['pressure_max'],
                        alpha=0.1, color='purple', label='Min-Max')
        ax4.plot(stats_data['valid_time'], stats_data['pressure_mean'],
                color='purple', linewidth=2, label='Mean')
        ax4.set_xlabel('Valid Time (UTC)')
        ax4.set_ylabel('Pressure (mb)')
        ax4.set_title('Pressure Spread')
        ax4.grid(True, alpha=0.3)
        ax4.invert_yaxis()
        ax4.legend()
        # Add title
        title = f"FNV3 Ensemble Spread Analysis"
        if forecast_date:
            title += f" - Forecast: {forecast_date.strftime('%Y-%m-%d %H:%M UTC')}"
        fig.suptitle(title, fontsize=16, fontweight='bold')
        # Adjust layout
        plt.tight_layout()
        # Save plot
        plot_filename = f"ensemble_spread"
        if forecast_date:
            plot_filename += f"_{forecast_date.strftime('%Y%m%d_%H%M')}"
        plot_filename += ".png"
        plot_path = self.output_dir / plot_filename
        if save_plot:
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved ensemble spread plot: {plot_path}")
        if show_plot:
            plt.show()
        else:
            plt.close()
        return str(plot_path)
    
    def create_summary_dashboard(self, df: pd.DataFrame, forecast_date: Optional[datetime] = None,
                               save_plot: bool = True, show_plot: bool = False) -> str:
        """
        Create a comprehensive dashboard with multiple plots.
        
        Args:
            df: Hurricane data DataFrame
            forecast_date: Optional forecast date to filter by
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Path to saved plot file
        """
        # Filter by forecast date if provided
        if forecast_date:
            df = df[df['forecast_time'].dt.date == forecast_date.date()]
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # Create grid layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # Track plot (top left, spans 2x2)
        ax_track = fig.add_subplot(gs[0:2, 0:2])
        self._plot_tracks_subplot(df, ax_track)
        
        # Intensity plot (top right, spans 2x2)
        ax_intensity = fig.add_subplot(gs[0:2, 2:4])
        self._plot_intensity_subplot(df, ax_intensity)
        
        # Spread plots (bottom row)
        ax_lat = fig.add_subplot(gs[2, 0])
        ax_lon = fig.add_subplot(gs[2, 1])
        ax_wind = fig.add_subplot(gs[2, 2])
        ax_pressure = fig.add_subplot(gs[2, 3])
        
        self._plot_spread_subplots(df, [ax_lat, ax_lon, ax_wind, ax_pressure])
        
        # Statistics table (bottom left, spans 1x2)
        ax_stats = fig.add_subplot(gs[3, 0:2])
        self._plot_statistics_table(df, ax_stats)
        
        # Add title
        title = f"FNV3 Hurricane Forecast Dashboard"
        if forecast_date:
            title += f" - Forecast: {forecast_date.strftime('%Y-%m-%d %H:%M UTC')}"
        fig.suptitle(title, fontsize=18, fontweight='bold')
        
        # Save plot
        plot_filename = f"dashboard"
        if forecast_date:
            plot_filename += f"_{forecast_date.strftime('%Y%m%d_%H%M')}"
        plot_filename += ".png"
        plot_path = self.output_dir / plot_filename
        
        if save_plot:
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved dashboard: {plot_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return str(plot_path)
    
    def _plot_tracks_subplot(self, df: pd.DataFrame, ax):
        """Helper method to plot tracks in a subplot."""
        ensemble_members = df['ensemble_member'].unique()
        
        for i, member in enumerate(ensemble_members):
            member_data = df[df['ensemble_member'] == member].sort_values('lead_time')
            
            if len(member_data) > 1:
                ax.plot(member_data['longitude'], member_data['latitude'],
                       color=self.ensemble_colors[i % len(self.ensemble_colors)],
                       alpha=0.7, linewidth=1)
        
        ax.set_xlabel('Longitude (°E)')
        ax.set_ylabel('Latitude (°N)')
        ax.set_title('Ensemble Tracks')
        ax.grid(True, alpha=0.3)
    
    def _plot_intensity_subplot(self, df: pd.DataFrame, ax):
        """Helper method to plot intensity in a subplot using valid_time as x-axis."""
        if df['valid_time'].dtype == 'object':
            df['valid_time'] = pd.to_datetime(df['valid_time'], format='mixed')
        ensemble_members = df['ensemble_member'].unique()
        for i, member in enumerate(ensemble_members):
            member_data = df[df['ensemble_member'] == member].sort_values('valid_time')
            if len(member_data) > 1 and 'wind_speed' in member_data.columns:
                ax.plot(member_data['valid_time'], member_data['wind_speed'],
                       color=self.ensemble_colors[i % len(self.ensemble_colors)],
                       alpha=0.7, linewidth=1)
        ax.set_xlabel('Valid Time (UTC)')
        ax.set_ylabel('Wind Speed (kt)')
        ax.set_title('Wind Speed Forecast')
        ax.grid(True, alpha=0.3)
    
    def _plot_spread_subplots(self, df: pd.DataFrame, axes):
        """Helper method to plot spread subplots using valid_time as x-axis."""
        if df['valid_time'].dtype == 'object':
            df['valid_time'] = pd.to_datetime(df['valid_time'], format='mixed')
        stats_data = df.groupby('valid_time').agg({
            'latitude': ['mean', 'std'],
            'longitude': ['mean', 'std'],
            'wind_speed': ['mean', 'std'],
            'pressure': ['mean', 'std']
        }).reset_index()
        # Flatten column names
        stats_data.columns = ['valid_time', 'lat_mean', 'lat_std', 'lon_mean', 'lon_std',
                             'wind_mean', 'wind_std', 'pressure_mean', 'pressure_std']
        variables = ['lat', 'lon', 'wind', 'pressure']
        colors = ['blue', 'red', 'green', 'purple']
        titles = ['Latitude', 'Longitude', 'Wind Speed', 'Pressure']
        ylabels = ['Latitude (°N)', 'Longitude (°E)', 'Wind Speed (kt)', 'Pressure (mb)']
        for i, (ax, var, color, title, ylabel) in enumerate(zip(axes, variables, colors, titles, ylabels)):
            mean_col = f'{var}_mean'
            std_col = f'{var}_std'
            if mean_col in stats_data.columns and std_col in stats_data.columns:
                ax.fill_between(stats_data['valid_time'],
                              stats_data[mean_col] - stats_data[std_col],
                              stats_data[mean_col] + stats_data[std_col],
                              alpha=0.3, color=color)
                ax.plot(stats_data['valid_time'], stats_data[mean_col],
                       color=color, linewidth=2)
            ax.set_xlabel('Valid Time (UTC)')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{title} Spread')
            ax.grid(True, alpha=0.3)
            if i == 3:  # Pressure plot
                ax.invert_yaxis()
    
    def _plot_statistics_table(self, df: pd.DataFrame, ax):
        """Helper method to plot statistics table."""
        # Calculate statistics
        stats = {
            'Total Points': len(df),
            'Ensemble Members': df['ensemble_member'].nunique(),
            'Lead Times': df['lead_time'].nunique(),
            'Max Wind Speed': f"{df['wind_speed'].max():.1f} kt",
            'Min Pressure': f"{df['pressure'].min():.1f} mb",
            'Latitude Range': f"{df['latitude'].min():.2f}° - {df['latitude'].max():.2f}°",
            'Longitude Range': f"{df['longitude'].min():.2f}° - {df['longitude'].max():.2f}°"
        }
        
        # Create table
        table_data = [[key, value] for key, value in stats.items()]
        table = ax.table(cellText=table_data, colLabels=['Statistic', 'Value'],
                        cellLoc='left', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        ax.set_title('Data Summary')
        ax.axis('off')

    def plot_single_trajectory(self, df: pd.DataFrame, forecast_time: datetime, 
                              ensemble_member: int = 0, save_plot: bool = True, 
                              show_plot: bool = False) -> str:
        """
        Plot a single trajectory for one ensemble member from one forecast time.
        
        Args:
            df: Hurricane data DataFrame
            forecast_time: Specific forecast time to plot
            ensemble_member: Ensemble member number (default: 0)
            save_plot: Whether to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Path to saved plot file
        """
        # Filter data for specific forecast time and ensemble member
        filtered_df = df[
            (df['forecast_time'].dt.date == forecast_time.date()) & 
            (df['forecast_time'].dt.time == forecast_time.time()) &
            (df['ensemble_member'] == ensemble_member)
        ].copy()
        
        if len(filtered_df) == 0:
            self.logger.warning(f"No data found for forecast {forecast_time} and ensemble member {ensemble_member}")
            return ""
        
        # Ensure valid_time is datetime
        if filtered_df['valid_time'].dtype == 'object':
            filtered_df['valid_time'] = pd.to_datetime(filtered_df['valid_time'], format='mixed')
        
        # Sort by valid_time for proper trajectory
        filtered_df = filtered_df.sort_values('valid_time')
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Track (latitude vs longitude)
        ax1.plot(filtered_df['longitude'], filtered_df['latitude'], 'b-', linewidth=2, alpha=0.8)
        ax1.scatter(filtered_df['longitude'], filtered_df['latitude'], 
                   c=filtered_df['wind_speed'], cmap='viridis', s=50, alpha=0.8)
        
        # Add colorbar for wind speed
        scatter = ax1.scatter([], [], c=[], cmap='viridis')
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Wind Speed (knots)')
        
        # Annotate start and end points
        start_point = filtered_df.iloc[0]
        end_point = filtered_df.iloc[-1]
        
        ax1.annotate(f'Start\n{start_point["valid_time"].strftime("%m/%d %H:%M")}\n{start_point["wind_speed"]:.0f} kt', 
                    xy=(start_point['longitude'], start_point['latitude']), 
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.7),
                    fontsize=8)
        
        ax1.annotate(f'End\n{end_point["valid_time"].strftime("%m/%d %H:%M")}\n{end_point["wind_speed"]:.0f} kt', 
                    xy=(end_point['longitude'], end_point['latitude']), 
                    xytext=(10, -20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7),
                    fontsize=8)
        
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.set_title(f'Single Trajectory - Forecast: {forecast_time.strftime("%Y-%m-%d %H:%M")} UTC, Ensemble: {ensemble_member}')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal', adjustable='datalim')
        
        # Plot 2: Intensity over time
        ax2.plot(filtered_df['valid_time'], filtered_df['wind_speed'], 'r-', linewidth=2, label='Wind Speed')
        ax2_twin = ax2.twinx()
        ax2_twin.plot(filtered_df['valid_time'], filtered_df['pressure'], 'b-', linewidth=2, label='Pressure')
        
        ax2.set_xlabel('Valid Time')
        ax2.set_ylabel('Wind Speed (knots)', color='red')
        ax2_twin.set_ylabel('Pressure (hPa)', color='blue')
        ax2.set_title('Intensity Evolution Over Time')
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # Add legend
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        
        # Save plot
        if save_plot:
            filename = f"single_trajectory_forecast_{forecast_time.strftime('%Y%m%d_%H%M')}_ensemble_{ensemble_member}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved single trajectory plot: {filepath}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return filepath if save_plot else ""


def main():
    """Main function to demonstrate hurricane data visualization."""
    try:
        # Import the analyzer to get data
        from .hurricane_analyzer import HurricaneAnalyzer
        
        # Initialize analyzer and visualizer
        analyzer = HurricaneAnalyzer()
        visualizer = HurricaneVisualizer()
        
        # Load data
        print("Loading hurricane data...")
        df = analyzer.load_all_data()
        
        # Get forecast dates
        forecast_dates = analyzer.extract_forecast_dates(df)
        
        print(f"Creating visualizations for {len(forecast_dates)} forecast dates...")
        
        # Create visualizations for each forecast date
        for i, forecast_date in enumerate(forecast_dates[:3]):  # Limit to first 3 for demo
            print(f"Processing forecast date {i+1}/{min(3, len(forecast_dates))}: {forecast_date}")
            
            # Create individual plots
            visualizer.plot_ensemble_tracks(df, forecast_date)
            visualizer.plot_intensity_curves(df, forecast_date)
            visualizer.plot_ensemble_spread(df, forecast_date)
            
            # Create dashboard
            visualizer.create_summary_dashboard(df, forecast_date)
        
        print(f"\n✅ Visualizations completed successfully!")
        print(f"Plots saved to: {visualizer.output_dir}")
        
    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        raise


if __name__ == "__main__":
    main() 