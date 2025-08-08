#!/usr/bin/env python3
"""
Visualize surge data with inland buffer clearly marked.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.utils.hurricane_geom import get_nicaragua_boundary
from shapely.geometry import Point

def visualize_surge_with_buffer():
    """Visualize surge data with inland buffer."""
    
    # Load the interpolated surge data
    surge_file = Path("data/preprocessed/surge/nicaragua_surge_interpolated_heights.npy")
    if not surge_file.exists():
        print("Interpolated surge file not found")
        return
    
    print("Loading interpolated surge data...")
    interpolated_surge = np.load(surge_file)
    print(f"Interpolated surge shape: {interpolated_surge.shape}")
    print(f"Surge range: {np.min(interpolated_surge):.3f} to {np.max(interpolated_surge):.3f}")
    
    # Get Nicaragua bounds for grid
    nicaragua_gdf = get_nicaragua_boundary()
    minx, miny, maxx, maxy = nicaragua_gdf.total_bounds
    
    resolution = 0.1
    lons = np.arange(minx, maxx + resolution, resolution)
    lats = np.arange(miny, maxy + resolution, resolution)
    
    # Create grid coordinates for visualization
    lon_centers = [(lons[i] + lons[i+1]) / 2 for i in range(len(lons) - 1)]
    lat_centers = [(lats[j] + lats[j+1]) / 2 for j in range(len(lats) - 1)]
    lon_mesh, lat_mesh = np.meshgrid(lon_centers, lat_centers)
    
    # Load original surge data to show buffer relationship
    original_surge_file = Path("data/preprocessed/surge/nicaragua_surge_ensemble_mean.npy")
    if original_surge_file.exists():
        original_surge = np.load(original_surge_file)
        
        # Create computational grid coordinates
        comp_minx, comp_maxx = -89.93, -73.83
        comp_miny, comp_maxy = 7.78, 18.97
        
        comp_lons = np.linspace(comp_minx, comp_maxx, 349)
        comp_lats = np.linspace(comp_miny, comp_maxy, 249)
        
        comp_lon_mesh, comp_lat_mesh = np.meshgrid(comp_lons, comp_lats)
        
        # Find significant surge points
        surge_threshold = 0.5
        valid_surge_mask = original_surge > surge_threshold
        comp_lons_surge = comp_lon_mesh[valid_surge_mask]
        comp_lats_surge = comp_lat_mesh[valid_surge_mask]
        comp_surge_surge = original_surge[valid_surge_mask]
        
        print(f"Original significant surge points: {len(comp_lons_surge)}")
    else:
        comp_lons_surge = []
        comp_lats_surge = []
        comp_surge_surge = []
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Interpolated surge heights
    im1 = ax1.pcolormesh(lon_mesh, lat_mesh, interpolated_surge, 
                         cmap='Blues', shading='auto', alpha=0.8)
    plt.colorbar(im1, ax=ax1, label='Surge Height (m)')
    nicaragua_gdf.plot(ax=ax1, edgecolor='black', facecolor='none', linewidth=2)
    ax1.set_title('Interpolated Surge Heights', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Longitude', fontsize=12)
    ax1.set_ylabel('Latitude', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Mark high surge locations
    high_surge_indices = np.where(interpolated_surge > 0.1)
    if len(high_surge_indices[0]) > 0:
        high_surge_lats = lat_mesh[high_surge_indices]
        high_surge_lons = lon_mesh[high_surge_indices]
        ax1.scatter(high_surge_lons, high_surge_lats, c='red', s=30, alpha=0.8, label='High Surge (>0.1m)')
        ax1.legend()
    
    # Plot 2: Distance from original surge points
    distance_from_surge = np.zeros_like(interpolated_surge)
    for i in range(interpolated_surge.shape[0]):
        for j in range(interpolated_surge.shape[1]):
            lat_val = lat_mesh[i, j]
            lon_val = lon_mesh[i, j]
            
            # Calculate distance to nearest original surge point
            if len(comp_lons_surge) > 0:
                distances = np.sqrt((lon_val - comp_lons_surge)**2 + (lat_val - comp_lats_surge)**2) * 111.0
                distance_from_surge[i, j] = np.min(distances)
            else:
                distance_from_surge[i, j] = 999.0
    
    im2 = ax2.pcolormesh(lon_mesh, lat_mesh, distance_from_surge, 
                         cmap='Reds', shading='auto', alpha=0.8)
    plt.colorbar(im2, ax=ax2, label='Distance from Original Surge Points (km)')
    nicaragua_gdf.plot(ax=ax2, edgecolor='black', facecolor='none', linewidth=2)
    # Plot the original surge points
    if len(comp_lons_surge) > 0:
        ax2.scatter(comp_lons_surge, comp_lats_surge, c='yellow', s=10, alpha=0.8, label='Original Surge Points')
    ax2.set_title('Distance from Original Surge Points', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Longitude', fontsize=12)
    ax2.set_ylabel('Latitude', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: 5km buffer mask
    buffer_mask = np.zeros_like(interpolated_surge)
    max_extension_distance_km = 5.0
    
    for i in range(interpolated_surge.shape[0]):
        for j in range(interpolated_surge.shape[1]):
            lat_val = lat_mesh[i, j]
            lon_val = lon_mesh[i, j]
            
            # Calculate distance to nearest original surge point
            if len(comp_lons_surge) > 0:
                distances = np.sqrt((lon_val - comp_lons_surge)**2 + (lat_val - comp_lats_surge)**2) * 111.0
                min_distance = np.min(distances)
                
                if min_distance <= max_extension_distance_km:
                    buffer_mask[i, j] = 1.0
    
    im3 = ax3.pcolormesh(lon_mesh, lat_mesh, buffer_mask, 
                         cmap='Greens', shading='auto', alpha=0.8)
    plt.colorbar(im3, ax=ax3, label='5km Buffer Mask')
    nicaragua_gdf.plot(ax=ax3, edgecolor='black', facecolor='none', linewidth=2)
    # Plot the original surge points
    if len(comp_lons_surge) > 0:
        ax3.scatter(comp_lons_surge, comp_lats_surge, c='yellow', s=10, alpha=0.8, label='Original Surge Points')
    ax3.set_title('5km Inland Buffer Mask', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Longitude', fontsize=12)
    ax3.set_ylabel('Latitude', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Surge with buffer overlay
    im4 = ax4.pcolormesh(lon_mesh, lat_mesh, interpolated_surge, 
                         cmap='Blues', shading='auto', alpha=0.8)
    plt.colorbar(im4, ax=ax4, label='Surge Height (m)')
    nicaragua_gdf.plot(ax=ax4, edgecolor='black', facecolor='none', linewidth=2)
    
    # Overlay buffer boundary
    buffer_boundary = np.where(buffer_mask == 1, 0, np.nan)
    ax4.pcolormesh(lon_mesh, lat_mesh, buffer_boundary, 
                   cmap='Reds', shading='auto', alpha=0.3)
    
    # Plot the original surge points
    if len(comp_lons_surge) > 0:
        ax4.scatter(comp_lons_surge, comp_lats_surge, c='yellow', s=10, alpha=0.8, label='Original Surge Points')
    
    # Mark high surge locations
    if len(high_surge_indices[0]) > 0:
        ax4.scatter(high_surge_lons, high_surge_lats, c='red', s=30, alpha=0.8, label='High Surge (>0.1m)')
    
    ax4.set_title('Surge Heights with 5km Buffer Overlay', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Longitude', fontsize=12)
    ax4.set_ylabel('Latitude', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Save the plot
    output_file = Path("data/preprocessed/surge/surge_with_buffer_visualization.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSurge with buffer visualization saved to: {output_file}")
    
    # Summary statistics
    print(f"\nSummary:")
    print(f"  Interpolated surge shape: {interpolated_surge.shape}")
    print(f"  Surge range: {np.min(interpolated_surge):.3f} to {np.max(interpolated_surge):.3f}")
    print(f"  Cells with surge > 0: {np.sum(interpolated_surge > 0)}")
    print(f"  Cells with surge > 0.1: {np.sum(interpolated_surge > 0.1)}")
    print(f"  Cells in 5km buffer: {np.sum(buffer_mask > 0)}")
    print(f"  Original surge points: {len(comp_lons_surge)}")
    
    # Debug: Check some specific points
    print(f"\nDebug - Surge values at specific points:")
    test_points = [
        (-85.0, 12.0),  # Center of Nicaragua
        (-83.5, 12.0),  # Caribbean coast
        (-86.5, 12.0),  # Pacific coast
    ]
    for lon, lat in test_points:
        # Find closest grid cell
        lon_centers_array = np.array(lon_centers)
        lat_centers_array = np.array(lat_centers)
        lon_idx = np.argmin(np.abs(lon_centers_array - lon))
        lat_idx = np.argmin(np.abs(lat_centers_array - lat))
        surge_val = interpolated_surge[lat_idx, lon_idx]
        buffer_val = buffer_mask[lat_idx, lon_idx]
        print(f"  Point ({lon}, {lat}): Surge={surge_val:.3f}m, InBuffer={buffer_val}")

if __name__ == "__main__":
    visualize_surge_with_buffer()
