#!/usr/bin/env python3
"""
Test script to show the corrected surge visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.utils.hurricane_geom import get_nicaragua_boundary

def test_visualization():
    """Test the corrected surge visualization."""
    
    # Load the corrected interpolated data
    interpolated_file = Path("data/preprocessed/surge/nicaragua_surge_interpolated_heights.npy")
    if not interpolated_file.exists():
        print("Interpolated surge file not found")
        return
    
    print("Loading corrected interpolated surge data...")
    interpolated_surge = np.load(interpolated_file)
    print(f"Interpolated surge shape: {interpolated_surge.shape}")
    print(f"Interpolated surge range: {np.min(interpolated_surge):.3f} to {np.max(interpolated_surge):.3f}")
    print(f"Cells with surge > 0: {np.sum(interpolated_surge > 0)}")
    print(f"Cells with surge > 0.1: {np.sum(interpolated_surge > 0.1)}")
    
    # Get grid parameters
    nicaragua_gdf = get_nicaragua_boundary()
    minx, miny, maxx, maxy = nicaragua_gdf.total_bounds
    
    resolution = 0.1
    lons = np.arange(minx, maxx + resolution, resolution)
    lats = np.arange(miny, maxy + resolution, resolution)
    
    # Create meshgrid for visualization
    lon_mesh, lat_mesh = np.meshgrid(lons[:-1], lats[:-1])
    
    print(f"Grid bounds: {minx:.3f} to {maxx:.3f} W, {miny:.3f} to {maxy:.3f} N")
    print(f"Grid shape: {len(lats)-1} x {len(lons)-1}")
    
    # Create meshgrid using grid cell centers (same as interpolation)
    lon_centers = [(lons[i] + lons[i+1]) / 2 for i in range(len(lons) - 1)]
    lat_centers = [(lats[j] + lats[j+1]) / 2 for j in range(len(lats) - 1)]
    
    lon_mesh, lat_mesh = np.meshgrid(lon_centers, lat_centers)
    
    print(f"Grid cell centers:")
    print(f"  Lon centers: {len(lon_centers)} points ({lon_centers[0]:.3f} to {lon_centers[-1]:.3f})")
    print(f"  Lat centers: {len(lat_centers)} points ({lat_centers[0]:.3f} to {lat_centers[-1]:.3f})")
    
    # Find high surge locations
    high_surge_indices = np.where(interpolated_surge > 0.1)
    print(f"\nHigh surge locations (>0.1m):")
    for i in range(min(10, len(high_surge_indices[0]))):
        lat_idx, lon_idx = high_surge_indices[0][i], high_surge_indices[1][i]
        lat_val = lat_mesh[lat_idx, lon_idx]
        lon_val = lon_mesh[lat_idx, lon_idx]
        surge_val = interpolated_surge[lat_idx, lon_idx]
        print(f"  ({lat_val:.3f}, {lon_val:.3f}): {surge_val:.3f}m")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Linear scale
    im1 = ax1.pcolormesh(lon_mesh, lat_mesh, interpolated_surge, 
                         cmap='Blues', shading='auto', alpha=0.8)
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Surge Height (m)', fontsize=12)
    
    # Add Nicaragua boundary
    nicaragua_gdf.plot(ax=ax1, edgecolor='black', facecolor='none', linewidth=2)
    ax1.set_title('Corrected Surge Heights - Linear Scale', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Longitude', fontsize=12)
    ax1.set_ylabel('Latitude', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Log scale
    log_surge = np.log10(interpolated_surge + 0.01)
    im2 = ax2.pcolormesh(lon_mesh, lat_mesh, log_surge, 
                         cmap='Blues', shading='auto', alpha=0.8)
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Log10(Surge Height + 0.01) (m)', fontsize=12)
    
    # Add Nicaragua boundary
    nicaragua_gdf.plot(ax=ax2, edgecolor='black', facecolor='none', linewidth=2)
    ax2.set_title('Corrected Surge Heights - Log Scale', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Longitude', fontsize=12)
    ax2.set_ylabel('Latitude', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Set consistent bounds
    for ax in [ax1, ax2]:
        ax.set_xlim(-87.5, -82.5)
        ax.set_ylim(10.5, 15.5)
    
    # Mark specific test points
    test_coords = [
        (10.913, -86.686),  # Pacific coast
        (11.013, -83.386),  # Caribbean coast
        (14.213, -85.886),  # Inland
    ]
    
    for ax in [ax1, ax2]:
        for lat_val, lon_val in test_coords:
            # Find closest grid cell
            lat_idx = np.argmin(np.abs(lats[:-1] - lat_val))
            lon_idx = np.argmin(np.abs(lons[:-1] - lon_val))
            
            if 0 <= lat_idx < len(lats)-1 and 0 <= lon_idx < len(lons)-1:
                mesh_lat = lat_mesh[lat_idx, lon_idx]
                mesh_lon = lon_mesh[lat_idx, lon_idx]
                surge_val = interpolated_surge[lat_idx, lon_idx]
                
                ax.plot(mesh_lon, mesh_lat, 'ro', markersize=8)
                ax.annotate(f'{surge_val:.2f}m', (mesh_lon, mesh_lat), 
                           xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    # Save the plot
    output_file = Path("data/preprocessed/surge/corrected_surge_visualization.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nCorrected surge visualization saved to: {output_file}")
    
    # Summary
    print(f"\nSummary:")
    print(f"  Total grid cells: {interpolated_surge.size}")
    print(f"  Cells with surge > 0: {np.sum(interpolated_surge > 0)} ({100*np.sum(interpolated_surge > 0)/interpolated_surge.size:.1f}%)")
    print(f"  Cells with surge > 0.1: {np.sum(interpolated_surge > 0.1)} ({100*np.sum(interpolated_surge > 0.1)/interpolated_surge.size:.1f}%)")
    print(f"  Maximum surge: {np.max(interpolated_surge):.3f}m")
    
    # Check coastal distribution
    pacific_mask = (lon_mesh < -85.0) & (lat_mesh < 12.0)
    caribbean_mask = (lon_mesh > -84.0) & (lat_mesh < 12.0)
    
    pacific_surge = interpolated_surge[pacific_mask]
    caribbean_surge = interpolated_surge[caribbean_mask]
    
    print(f"\nCoastal distribution:")
    print(f"  Pacific coast cells with surge > 0.1: {np.sum(pacific_surge > 0.1)}")
    print(f"  Caribbean coast cells with surge > 0.1: {np.sum(caribbean_surge > 0.1)}")
    print(f"  Pacific max surge: {np.max(pacific_surge):.3f}m")
    print(f"  Caribbean max surge: {np.max(caribbean_surge):.3f}m")

if __name__ == "__main__":
    test_visualization()
