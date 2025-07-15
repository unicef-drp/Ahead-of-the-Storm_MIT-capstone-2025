"""
Rainfall Ensemble Impact Analysis and Heatmap Generation

This script loads processed rainfall ensemble data (NetCDF), computes ensemble statistics (mean, min, max, etc.),
and generates heatmaps/overlays for the Nicaragua region. Plots and summary statistics are saved to the results directory
specified in config/rainfall_config.yaml.
"""
import os
import yaml
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from src.utils.path_utils import ensure_directory

def load_config(config_path='config/rainfall_config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_processed_rainfall(config):
    processed_path = os.path.join(config['processed_data_dir'], 'rainfall_ensemble.nc')
    return xr.open_dataarray(processed_path)

def compute_ensemble_stats(precip_data):
    stats = {
        'mean': precip_data.mean(dim='number'),
        'min': precip_data.min(dim='number'),
        'max': precip_data.max(dim='number'),
        'std': precip_data.std(dim='number'),
        'median': precip_data.median(dim='number'),
    }
    return stats

def plot_ensemble_heatmaps(precip_data, stats, config):
    results_dir = config['results_dir']
    ensure_directory(results_dir)
    projection = ccrs.PlateCarree()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), subplot_kw={'projection': projection})
    axes = axes.flatten()
    # Mean
    im0 = axes[0].pcolormesh(precip_data.longitude, precip_data.latitude, stats['mean'],
                             cmap='Blues', transform=projection)
    axes[0].set_title('Ensemble Mean Rainfall (mm)')
    fig.colorbar(im0, ax=axes[0], orientation='vertical')
    # Min
    im1 = axes[1].pcolormesh(precip_data.longitude, precip_data.latitude, stats['min'],
                             cmap='Greens', transform=projection)
    axes[1].set_title('Ensemble Min Rainfall (mm)')
    fig.colorbar(im1, ax=axes[1], orientation='vertical')
    # Max
    im2 = axes[2].pcolormesh(precip_data.longitude, precip_data.latitude, stats['max'],
                             cmap='Reds', transform=projection)
    axes[2].set_title('Ensemble Max Rainfall (mm)')
    fig.colorbar(im2, ax=axes[2], orientation='vertical')
    # Std
    im3 = axes[3].pcolormesh(precip_data.longitude, precip_data.latitude, stats['std'],
                             cmap='Purples', transform=projection)
    axes[3].set_title('Ensemble Std Dev Rainfall (mm)')
    fig.colorbar(im3, ax=axes[3], orientation='vertical')
    for ax in axes:
        ax.coastlines(resolution='50m')
        ax.add_feature(cfeature.BORDERS, alpha=0.5)
        ax.set_extent([
            config['region']['lon_min'], config['region']['lon_max'],
            config['region']['lat_min'], config['region']['lat_max']
        ], crs=projection)
    plt.tight_layout()
    out_path = os.path.join(results_dir, 'rainfall_ensemble_heatmaps.png')
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved rainfall ensemble heatmaps to: {out_path}")

def main(config_path='config/rainfall_config.yaml'):
    config = load_config(config_path)
    precip_data = load_processed_rainfall(config)
    stats = compute_ensemble_stats(precip_data)
    plot_ensemble_heatmaps(precip_data, stats, config)

if __name__ == "__main__":
    main() 