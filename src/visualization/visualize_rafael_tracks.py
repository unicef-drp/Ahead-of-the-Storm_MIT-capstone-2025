#!/usr/bin/env python3
"""
Hurricane Rafael Track Visualization
Original Ensemble and Observed Tracks Only

This script visualizes the WeatherLab ensemble forecasts for Hurricane Rafael (2024) 
and overlays the observed (historical) track, with no artificial modification.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def load_data(date="2024-11-04", time="12:00"):
    """Load original track data."""
    original_file = f"data/weatherlab/raw_ensemble/rafael_{date}_{time.replace(':', '_')}.csv"
    print(f"Loading data for {date} {time} UTC...")
    print(f"Original file: {original_file}")
    df = pd.read_csv(original_file)
    print(f"\nData loaded successfully! Shape: {df.shape}")
    return df

def separate_tracks(df):
    """Separate observed and ensemble tracks."""
    # Observed track: sample == 0
    observed = df[df['sample'] == 0]
    ensemble = df[df['sample'] > 0]
    print(f"Observed track points: {len(observed)}")
    print(f"Ensemble track points: {len(ensemble)}")
    print(f"Ensemble members: {ensemble['sample'].nunique()}")
    return observed, ensemble

def plot_tracks(observed, ensemble, date, time):
    """Plot the original ensemble and observed tracks."""
    nicaragua_bounds = {
        'lat_min': 10.7, 'lat_max': 15.0,
        'lon_min': -87.7, 'lon_max': -82.7
    }
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title(f'WeatherLab Ensemble & Observed Tracks\nHurricane Rafael {date} {time} UTC', fontsize=16, fontweight='bold')
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    ax.grid(True, alpha=0.3)
    # Plot ensemble members
    for sample in ensemble['sample'].unique():
        track = ensemble[ensemble['sample'] == sample]
        ax.plot(track['lon'], track['lat'], 'b-', alpha=0.3, linewidth=0.8)
    # Plot observed track
    if len(observed) > 0:
        ax.plot(observed['lon'], observed['lat'], 'r-', linewidth=3, label='Observed')
        ax.scatter(observed['lon'].iloc[0], observed['lat'].iloc[0], c='red', s=100, marker='o', label='Start')
        ax.scatter(observed['lon'].iloc[-1], observed['lat'].iloc[-1], c='red', s=100, marker='s', label='End')
    # Add Nicaragua outline
    nicaragua_rect = plt.Rectangle((nicaragua_bounds['lon_min'], nicaragua_bounds['lat_min']), 
                                  nicaragua_bounds['lon_max'] - nicaragua_bounds['lon_min'],
                                  nicaragua_bounds['lat_max'] - nicaragua_bounds['lat_min'],
                                  fill=False, edgecolor='green', linewidth=2, linestyle='--', label='Nicaragua')
    ax.add_patch(nicaragua_rect)
    ax.legend()
    # Zoom to region of interest
    ax.set_xlim(-110, -60)
    ax.set_ylim(5, 35)
    plt.tight_layout()
    plt.show()

def plot_pressure(observed, ensemble):
    """Plot minimum pressure evolution for ensemble and observed tracks."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title('Minimum Sea Level Pressure Evolution', fontsize=15)
    ax.set_xlabel('Lead Time (hours)')
    ax.set_ylabel('Minimum Sea Level Pressure (hPa)')
    ax.grid(True, alpha=0.3)
    # Convert lead_time to hours
    ensemble['lead_time_hours'] = pd.to_timedelta(ensemble['lead_time']).dt.total_seconds() / 3600
    observed['lead_time_hours'] = pd.to_timedelta(observed['lead_time']).dt.total_seconds() / 3600
    # Plot ensemble mean
    ens_mean = ensemble.groupby('lead_time_hours')['minimum_sea_level_pressure_hpa'].mean()
    ax.plot(ens_mean.index, ens_mean.values, 'b-', linewidth=2, label='Ensemble Mean')
    # Plot observed
    if len(observed) > 0:
        ax.plot(observed['lead_time_hours'], observed['minimum_sea_level_pressure_hpa'], 'r-', linewidth=3, label='Observed')
    ax.legend()
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()

def main():
    date = "2024-11-04"
    time = "12:00"
    df = load_data(date, time)
    observed, ensemble = separate_tracks(df)
    plot_tracks(observed, ensemble, date, time)
    plot_pressure(observed, ensemble)

if __name__ == "__main__":
    main() 