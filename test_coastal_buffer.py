#!/usr/bin/env python3
"""
Test implementing a coastal buffer approach with distance-based decay from the coast.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.utils.hurricane_geom import get_nicaragua_boundary

def test_coastal_buffer():
    """Test implementing a coastal buffer approach."""
    
    # Load the computational grid surge data
    surge_file = Path("data/preprocessed/surge/nicaragua_surge_ensemble_mean.npy")
    if not surge_file.exists():
        print("Surge file not found")
        return
    
    print("Loading computational grid surge data...")
    surge_data = np.load(surge_file)
    print(f"Computational surge shape: {surge_data.shape}")
    
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
    
    # Find coastal surge points (highest values at the coast)
    valid_surge_mask = surge_data > 0
    comp_lons_all = comp_lon_mesh[valid_surge_mask]
    comp_lats_all = comp_lat_mesh[valid_surge_mask]
    comp_surge_all = surge_data[valid_surge_mask]
    
    print(f"Total surge points: {len(comp_lons_all)}")
    print(f"Target grid points: {len(grid_lons)}")
    
    # Define coastal boundaries
    caribbean_coast_lon = -83.5  # Caribbean coast (east)
    pacific_coast_lon = -86.5    # Pacific coast (west)
    
    # Find coastal surge points (within 5km of coast)
    coastal_threshold_km = 5.0
    coastal_mask = np.zeros_like(comp_lons_all, dtype=bool)
    
    for i, (lon, lat) in enumerate(zip(comp_lons_all, comp_lats_all)):
        # Calculate distance to nearest coast
        dist_to_caribbean = abs(lon - caribbean_coast_lon) * 111.0  # Convert to km
        dist_to_pacific = abs(lon - pacific_coast_lon) * 111.0  # Convert to km
        min_dist_to_coast = min(dist_to_caribbean, dist_to_pacific)
        
        if min_dist_to_coast <= coastal_threshold_km:
            coastal_mask[i] = True
    
    coastal_lons = comp_lons_all[coastal_mask]
    coastal_lats = comp_lats_all[coastal_mask]
    coastal_surge = comp_surge_all[coastal_mask]
    
    print(f"Coastal surge points (within {coastal_threshold_km}km): {len(coastal_lons)}")
    print(f"Coastal surge range: {np.min(coastal_surge):.3f} to {np.max(coastal_surge):.3f}")
    
    # Create the coastal buffer approach
    max_buffer_distance_km = 2.0  # 2km buffer from coast
    buffer_surge = np.zeros(len(grid_lons))
    
    print(f"\nApplying coastal buffer approach:")
    print(f"  Max buffer distance: {max_buffer_distance_km}km")
    print(f"  Coastal points: {len(coastal_lons)}")
    
    # For each target grid point, find the nearest coastal surge point and apply distance decay
    for i, (lon, lat) in enumerate(zip(grid_lons, grid_lats)):
        if len(coastal_lons) > 0:
            # Calculate distance to all coastal points
            distances = np.sqrt((lon - coastal_lons)**2 + (lat - coastal_lats)**2) * 111.0  # Convert to km
            
            # Find nearest coastal point
            nearest_idx = np.argmin(distances)
            nearest_distance_km = distances[nearest_idx]
            nearest_surge = coastal_surge[nearest_idx]
            
            # Apply distance-based decay
            if nearest_distance_km <= max_buffer_distance_km:
                # Linear decay from coast to max distance
                decay_factor = 1.0 - (nearest_distance_km / max_buffer_distance_km)
                buffer_surge[i] = nearest_surge * decay_factor
            else:
                # Beyond max distance, no surge
                buffer_surge[i] = 0.0
    
    print(f"Buffer surge range: {np.min(buffer_surge):.3f} to {np.max(buffer_surge):.3f}")
    print(f"Buffer cells > 0: {np.sum(buffer_surge > 0)}")
    print(f"Buffer cells > 0.1: {np.sum(buffer_surge > 0.1)}")
    
    # Reshape to grid
    reshaped_surge = buffer_surge.reshape(len(lats)-1, len(lons)-1)
    print(f"Reshaped surge shape: {reshaped_surge.shape}")
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Coastal surge points (source data)
    ax1.scatter(coastal_lons, coastal_lats, c=coastal_surge, 
               cmap='Blues', s=20, alpha=0.8)
    nicaragua_gdf.plot(ax=ax1, edgecolor='black', facecolor='none', linewidth=2)
    ax1.set_title(f'Coastal Surge Points (Within {coastal_threshold_km}km)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Longitude', fontsize=12)
    ax1.set_ylabel('Latitude', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Add coast lines
    ax1.axvline(x=caribbean_coast_lon, color='red', linestyle='--', alpha=0.7, label='Caribbean Coast')
    ax1.axvline(x=pacific_coast_lon, color='red', linestyle='--', alpha=0.7, label='Pacific Coast')
    ax1.legend()
    
    # Plot 2: Distance from coast for target grid
    lon_centers = [(lons[i] + lons[i+1]) / 2 for i in range(len(lons) - 1)]
    lat_centers = [(lats[j] + lats[j+1]) / 2 for j in range(len(lats) - 1)]
    lon_mesh, lat_mesh = np.meshgrid(lon_centers, lat_centers)
    
    # Calculate distance from coast for visualization
    distance_from_coast = np.zeros_like(reshaped_surge)
    for i in range(reshaped_surge.shape[0]):
        for j in range(reshaped_surge.shape[1]):
            lat_val = lat_mesh[i, j]
            lon_val = lon_mesh[i, j]
            
            dist_to_caribbean = abs(lon_val - caribbean_coast_lon) * 111.0
            dist_to_pacific = abs(lon_val - pacific_coast_lon) * 111.0
            min_dist = min(dist_to_caribbean, dist_to_pacific)
            
            distance_from_coast[i, j] = min_dist
    
    im2 = ax2.pcolormesh(lon_mesh, lat_mesh, distance_from_coast, 
                         cmap='Reds', shading='auto', alpha=0.8)
    plt.colorbar(im2, ax=ax2, label='Distance from Coast (km)')
    nicaragua_gdf.plot(ax=ax2, edgecolor='black', facecolor='none', linewidth=2)
    ax2.set_title('Distance from Coast', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Longitude', fontsize=12)
    ax2.set_ylabel('Latitude', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add coast lines
    ax2.axvline(x=caribbean_coast_lon, color='blue', linestyle='--', alpha=0.7, label='Caribbean Coast')
    ax2.axvline(x=pacific_coast_lon, color='blue', linestyle='--', alpha=0.7, label='Pacific Coast')
    ax2.legend()
    
    # Plot 3: Coastal buffer result
    im3 = ax3.pcolormesh(lon_mesh, lat_mesh, reshaped_surge, 
                         cmap='Blues', shading='auto', alpha=0.8)
    plt.colorbar(im3, ax=ax3, label='Surge Height (m)')
    nicaragua_gdf.plot(ax=ax3, edgecolor='black', facecolor='none', linewidth=2)
    ax3.set_title(f'Coastal Buffer Surge ({max_buffer_distance_km}km)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Longitude', fontsize=12)
    ax3.set_ylabel('Latitude', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Mark high surge locations
    high_surge_indices = np.where(reshaped_surge > 0.1)
    if len(high_surge_indices[0]) > 0:
        high_surge_lats = lat_mesh[high_surge_indices]
        high_surge_lons = lon_mesh[high_surge_indices]
        ax3.scatter(high_surge_lons, high_surge_lats, c='red', s=30, alpha=0.8, label='High Surge (>0.1m)')
        ax3.legend()
    
    # Add coast lines
    ax3.axvline(x=caribbean_coast_lon, color='red', linestyle='--', alpha=0.7, label='Caribbean Coast')
    ax3.axvline(x=pacific_coast_lon, color='red', linestyle='--', alpha=0.7, label='Pacific Coast')
    
    # Plot 4: Decay factor visualization
    decay_factor_grid = np.zeros_like(reshaped_surge)
    for i in range(reshaped_surge.shape[0]):
        for j in range(reshaped_surge.shape[1]):
            lat_val = lat_mesh[i, j]
            lon_val = lon_mesh[i, j]
            
            dist_to_caribbean = abs(lon_val - caribbean_coast_lon) * 111.0
            dist_to_pacific = abs(lon_val - pacific_coast_lon) * 111.0
            min_dist = min(dist_to_caribbean, dist_to_pacific)
            
            if min_dist <= max_buffer_distance_km:
                decay_factor_grid[i, j] = 1.0 - (min_dist / max_buffer_distance_km)
            else:
                decay_factor_grid[i, j] = 0.0
    
    im4 = ax4.pcolormesh(lon_mesh, lat_mesh, decay_factor_grid, 
                         cmap='Greens', shading='auto', alpha=0.8)
    plt.colorbar(im4, ax=ax4, label='Decay Factor')
    nicaragua_gdf.plot(ax=ax4, edgecolor='black', facecolor='none', linewidth=2)
    ax4.set_title('Distance Decay Factor', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Longitude', fontsize=12)
    ax4.set_ylabel('Latitude', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    # Add coast lines
    ax4.axvline(x=caribbean_coast_lon, color='blue', linestyle='--', alpha=0.7, label='Caribbean Coast')
    ax4.axvline(x=pacific_coast_lon, color='blue', linestyle='--', alpha=0.7, label='Pacific Coast')
    ax4.legend()
    
    # Save the plot
    output_file = Path("data/preprocessed/surge/coastal_buffer_test.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nCoastal buffer test visualization saved to: {output_file}")
    
    # Save the coastal buffer data
    output_data_file = Path("data/preprocessed/surge/nicaragua_surge_coastal_buffer.npy")
    np.save(output_data_file, reshaped_surge)
    print(f"Coastal buffer data saved to: {output_data_file}")
    
    # Summary
    print(f"\nSummary:")
    print(f"  Coastal surge points: {len(coastal_lons)}")
    print(f"  Buffer cells > 0: {np.sum(buffer_surge > 0)}")
    print(f"  Buffer cells > 0.1: {np.sum(buffer_surge > 0.1)}")
    print(f"  Max surge in buffer: {np.max(buffer_surge):.3f}m")

if __name__ == "__main__":
    test_coastal_buffer()
