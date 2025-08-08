#!/usr/bin/env python3
"""
Check surge coverage and determine optimal buffer distance.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.utils.hurricane_geom import get_nicaragua_boundary
from shapely.geometry import Point

def check_surge_coverage():
    """Check surge coverage and determine optimal buffer distance."""
    
    # Load the original surge data
    original_surge_file = Path("data/preprocessed/surge/nicaragua_surge_ensemble_mean.npy")
    if not original_surge_file.exists():
        print("Original surge file not found")
        return
    
    print("Loading original surge data...")
    original_surge = np.load(original_surge_file)
    print(f"Original surge shape: {original_surge.shape}")
    print(f"Original surge range: {np.min(original_surge):.3f} to {np.max(original_surge):.3f}")
    
    # Create computational grid coordinates
    comp_minx, comp_maxx = -89.93, -73.83
    comp_miny, comp_maxy = 7.78, 18.97
    
    comp_lons = np.linspace(comp_minx, comp_maxx, 349)
    comp_lats = np.linspace(comp_miny, comp_maxy, 249)
    
    comp_lon_mesh, comp_lat_mesh = np.meshgrid(comp_lons, comp_lats)
    
    # Get Nicaragua bounds for target grid
    nicaragua_gdf = get_nicaragua_boundary()
    minx, miny, maxx, maxy = nicaragua_gdf.total_bounds
    
    resolution = 0.1
    lons = np.arange(minx, maxx + resolution, resolution)
    lats = np.arange(miny, maxy + resolution, resolution)
    
    # Create target grid points
    grid_lons = []
    grid_lats = []
    for j in range(len(lats) - 1):  # latitude (rows)
        for i in range(len(lons) - 1):  # longitude (columns)
            grid_lons.append((lons[i] + lons[i+1]) / 2)
            grid_lats.append((lats[j] + lats[j+1]) / 2)
    
    # Find significant surge points
    surge_threshold = 0.5
    valid_surge_mask = original_surge > surge_threshold
    comp_lons_surge = comp_lon_mesh[valid_surge_mask]
    comp_lats_surge = comp_lat_mesh[valid_surge_mask]
    comp_surge_surge = original_surge[valid_surge_mask]
    
    print(f"Significant surge points (> {surge_threshold}m): {len(comp_lons_surge)}")
    
    # Test different buffer distances
    buffer_distances = [2.0, 5.0, 10.0, 15.0, 20.0]
    
    print(f"\nTesting different buffer distances:")
    for buffer_km in buffer_distances:
        # Count cells that would be covered
        covered_cells = 0
        for lon, lat in zip(grid_lons, grid_lats):
            if len(comp_lons_surge) > 0:
                distances = np.sqrt((lon - comp_lons_surge)**2 + (lat - comp_lats_surge)**2) * 111.0
                min_distance = np.min(distances)
                if min_distance <= buffer_km:
                    covered_cells += 1
        
        print(f"  {buffer_km}km buffer: {covered_cells} cells covered")
    
    # Check distance from Nicaragua coastline to nearest surge points
    print(f"\nDistance from Nicaraguan coastline to nearest surge points:")
    
    # Sample points along Nicaraguan coastline
    coastline_points = [
        (-83.5, 12.0),  # Caribbean coast
        (-83.5, 13.0),  # Caribbean coast
        (-83.5, 14.0),  # Caribbean coast
        (-86.5, 12.0),  # Pacific coast
        (-86.5, 13.0),  # Pacific coast
        (-86.5, 14.0),  # Pacific coast
    ]
    
    for lon, lat in coastline_points:
        if len(comp_lons_surge) > 0:
            distances = np.sqrt((lon - comp_lons_surge)**2 + (lat - comp_lats_surge)**2) * 111.0
            min_distance = np.min(distances)
            print(f"  Coastline point ({lon}, {lat}): {min_distance:.1f}km to nearest surge")
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Original surge points
    im1 = ax1.pcolormesh(comp_lon_mesh, comp_lat_mesh, original_surge, 
                         cmap='Blues', shading='auto', alpha=0.8)
    plt.colorbar(im1, ax=ax1, label='Original Surge Height (m)')
    nicaragua_gdf.plot(ax=ax1, edgecolor='black', facecolor='none', linewidth=2)
    # Plot significant surge points
    ax1.scatter(comp_lons_surge, comp_lats_surge, c='yellow', s=10, alpha=0.8, label='Significant Surge Points')
    ax1.set_title('Original Surge Data with Significant Points', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Longitude', fontsize=12)
    ax1.set_ylabel('Latitude', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Distance from coastline to surge points
    distance_from_coastline = np.zeros_like(original_surge)
    for i in range(original_surge.shape[0]):
        for j in range(original_surge.shape[1]):
            lat_val = comp_lat_mesh[i, j]
            lon_val = comp_lon_mesh[i, j]
            
            # Calculate distance to Nicaraguan coastline
            grid_point = Point(lon_val, lat_val)
            distance_from_coastline[i, j] = grid_point.distance(nicaragua_gdf.geometry.iloc[0]) * 111.0
    
    im2 = ax2.pcolormesh(comp_lon_mesh, comp_lat_mesh, distance_from_coastline, 
                         cmap='Reds', shading='auto', alpha=0.8)
    plt.colorbar(im2, ax=ax2, label='Distance from Nicaraguan Coastline (km)')
    nicaragua_gdf.plot(ax=ax2, edgecolor='black', facecolor='none', linewidth=2)
    ax2.set_title('Distance from Nicaraguan Coastline', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Longitude', fontsize=12)
    ax2.set_ylabel('Latitude', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Current 5km buffer coverage
    current_buffer = np.zeros(len(grid_lons))
    max_extension_distance_km = 5.0
    
    for i, (lon, lat) in enumerate(zip(grid_lons, grid_lats)):
        if len(comp_lons_surge) > 0:
            distances = np.sqrt((lon - comp_lons_surge)**2 + (lat - comp_lats_surge)**2) * 111.0
            min_distance = np.min(distances)
            if min_distance <= max_extension_distance_km:
                current_buffer[i] = 1.0
    
    current_buffer_grid = current_buffer.reshape(len(lats)-1, len(lons)-1)
    lon_centers = [(lons[i] + lons[i+1]) / 2 for i in range(len(lons) - 1)]
    lat_centers = [(lats[j] + lats[j+1]) / 2 for j in range(len(lats) - 1)]
    lon_mesh, lat_mesh = np.meshgrid(lon_centers, lat_centers)
    
    im3 = ax3.pcolormesh(lon_mesh, lat_mesh, current_buffer_grid, 
                         cmap='Greens', shading='auto', alpha=0.8)
    plt.colorbar(im3, ax=ax3, label='Current 5km Buffer')
    nicaragua_gdf.plot(ax=ax3, edgecolor='black', facecolor='none', linewidth=2)
    ax3.set_title('Current 5km Buffer Coverage', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Longitude', fontsize=12)
    ax3.set_ylabel('Latitude', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: 15km buffer coverage (suggested increase)
    extended_buffer = np.zeros(len(grid_lons))
    max_extension_distance_km = 15.0
    
    for i, (lon, lat) in enumerate(zip(grid_lons, grid_lats)):
        if len(comp_lons_surge) > 0:
            distances = np.sqrt((lon - comp_lons_surge)**2 + (lat - comp_lats_surge)**2) * 111.0
            min_distance = np.min(distances)
            if min_distance <= max_extension_distance_km:
                extended_buffer[i] = 1.0
    
    extended_buffer_grid = extended_buffer.reshape(len(lats)-1, len(lons)-1)
    
    im4 = ax4.pcolormesh(lon_mesh, lat_mesh, extended_buffer_grid, 
                         cmap='Greens', shading='auto', alpha=0.8)
    plt.colorbar(im4, ax=ax4, label='Extended 15km Buffer')
    nicaragua_gdf.plot(ax=ax4, edgecolor='black', facecolor='none', linewidth=2)
    ax4.set_title('Extended 15km Buffer Coverage', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Longitude', fontsize=12)
    ax4.set_ylabel('Latitude', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    # Save the plot
    output_file = Path("data/preprocessed/surge/surge_coverage_analysis.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSurge coverage analysis saved to: {output_file}")
    
    # Summary
    print(f"\nSummary:")
    print(f"  Original surge points: {len(comp_lons_surge)}")
    print(f"  Current 5km buffer cells: {np.sum(current_buffer > 0)}")
    print(f"  Extended 15km buffer cells: {np.sum(extended_buffer > 0)}")
    print(f"  Coverage increase: {np.sum(extended_buffer > 0) - np.sum(current_buffer > 0)} cells")

if __name__ == "__main__":
    check_surge_coverage()
