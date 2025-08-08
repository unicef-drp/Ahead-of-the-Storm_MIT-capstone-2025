#!/usr/bin/env python3
"""
Test that shows exactly how the interpolation is mapping coordinates.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.utils.hurricane_geom import get_nicaragua_boundary

def test_interpolation_mapping():
    """Test exactly how the interpolation is mapping coordinates."""
    
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
    
    # Get grid parameters for interpolated data
    nicaragua_gdf = get_nicaragua_boundary()
    minx, miny, maxx, maxy = nicaragua_gdf.total_bounds
    
    resolution = 0.1
    lons = np.arange(minx, maxx + resolution, resolution)
    lats = np.arange(miny, maxy + resolution, resolution)
    
    # Create grid cell centers for interpolation (same as in surge layer)
    grid_lons = []
    grid_lats = []
    for j in range(len(lats) - 1):  # latitude (rows)
        for i in range(len(lons) - 1):  # longitude (columns)
            grid_lons.append((lons[i] + lons[i+1]) / 2)
            grid_lats.append((lats[j] + lats[j+1]) / 2)
    
    print(f"Grid parameters:")
    print(f"  Computational bounds: {comp_minx:.3f} to {comp_maxx:.3f} W, {comp_miny:.3f} to {comp_maxy:.3f} N")
    print(f"  Interpolation bounds: {minx:.3f} to {maxx:.3f} W, {miny:.3f} to {maxy:.3f} N")
    print(f"  Interpolation grid cells: {len(grid_lons)}")
    
    # Find high surge locations in computational grid
    high_surge_indices = np.where(surge_data > 0.5)
    print(f"\nHigh surge locations (>0.5m) in computational grid:")
    for i in range(min(5, len(high_surge_indices[0]))):
        lat_idx, lon_idx = high_surge_indices[0][i], high_surge_indices[1][i]
        lat_val = comp_lat_mesh[lat_idx, lon_idx]
        lon_val = comp_lon_mesh[lat_idx, lon_idx]
        surge_val = surge_data[lat_idx, lon_idx]
        print(f"  ({lat_val:.3f}, {lon_val:.3f}): {surge_val:.3f}m")
    
    # Test the interpolation step by step
    print(f"\nTesting interpolation step by step:")
    
    # Step 1: Filter to Nicaragua region
    nicaragua_lon_min, nicaragua_lon_max = -87.7, -82.7
    nicaragua_lat_min, nicaragua_lat_max = 10.7, 15.0
    
    nicaragua_mask = (
        (comp_lon_mesh >= nicaragua_lon_min) & (comp_lon_mesh <= nicaragua_lon_max) &
        (comp_lat_mesh >= nicaragua_lat_min) & (comp_lat_mesh <= nicaragua_lat_max)
    )
    
    valid_surge_mask = (surge_data > 0) & nicaragua_mask
    print(f"  Valid surge mask sum: {np.sum(valid_surge_mask)}")
    
    if np.any(valid_surge_mask):
        comp_lons_valid = comp_lon_mesh[valid_surge_mask]
        comp_lats_valid = comp_lat_mesh[valid_surge_mask]
        comp_surge_valid = surge_data[valid_surge_mask]
        
        print(f"  Valid interpolation points: {len(comp_lons_valid)}")
        print(f"  Valid surge range: {np.min(comp_surge_valid):.3f} to {np.max(comp_surge_valid):.3f}")
        
        # Show some sample coordinates
        print(f"  Sample computational coordinates:")
        for i in range(min(5, len(comp_lons_valid))):
            print(f"    ({comp_lats_valid[i]:.3f}, {comp_lons_valid[i]:.3f}): {comp_surge_valid[i]:.3f}m")
        
        # Step 2: Create interpolation points
        points = np.column_stack([comp_lons_valid, comp_lats_valid])
        values = comp_surge_valid
        
        # Step 3: Create target points
        xi = np.column_stack([grid_lons, grid_lats])
        
        print(f"  Target grid points: {len(grid_lons)}")
        print(f"  Sample target coordinates:")
        for i in range(min(5, len(grid_lons))):
            print(f"    ({grid_lats[i]:.3f}, {grid_lons[i]:.3f})")
        
        # Step 4: Interpolate
        from scipy.interpolate import griddata
        interpolated_surge = griddata(points, values, xi, method='nearest', fill_value=0.0)
        
        print(f"  Interpolated surge range: {np.min(interpolated_surge):.3f} to {np.max(interpolated_surge):.3f}")
        print(f"  Interpolated cells > 0: {np.sum(interpolated_surge > 0)}")
        
        # Step 5: Reshape to grid
        reshaped_surge = interpolated_surge.reshape(len(lats)-1, len(lons)-1)
        print(f"  Reshaped surge shape: {reshaped_surge.shape}")
        
        # Find high surge locations in interpolated data
        interp_high_indices = np.where(reshaped_surge > 0.1)
        print(f"  High surge locations (>0.1m) in interpolated data:")
        for i in range(min(5, len(interp_high_indices[0]))):
            lat_idx, lon_idx = interp_high_indices[0][i], interp_high_indices[1][i]
            # Get the actual coordinates for this grid cell
            lat_center = (lats[lat_idx] + lats[lat_idx+1]) / 2
            lon_center = (lons[lon_idx] + lons[lon_idx+1]) / 2
            surge_val = reshaped_surge[lat_idx, lon_idx]
            print(f"    Grid[{lat_idx},{lon_idx}] ({lat_center:.3f}, {lon_center:.3f}): {surge_val:.3f}m")
        
        # Create visualization showing the mapping
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Computational grid with high surge
        im1 = ax1.pcolormesh(comp_lon_mesh, comp_lat_mesh, surge_data, 
                             cmap='Blues', shading='auto', alpha=0.8)
        plt.colorbar(im1, ax=ax1, label='Surge Height (m)')
        
        # Add Nicaragua boundary
        nicaragua_gdf.plot(ax=ax1, edgecolor='black', facecolor='none', linewidth=2)
        ax1.set_title('Computational Grid - High Surge Locations', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Longitude', fontsize=12)
        ax1.set_ylabel('Latitude', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Mark high surge locations
        if len(high_surge_indices[0]) > 0:
            comp_high_lats = comp_lat_mesh[high_surge_indices]
            comp_high_lons = comp_lon_mesh[high_surge_indices]
            ax1.scatter(comp_high_lons, comp_high_lats, c='red', s=20, alpha=0.8, label='High Surge (>0.5m)')
            ax1.legend()
        
        # Plot 2: Valid interpolation points
        ax2.scatter(comp_lons_valid, comp_lats_valid, c=comp_surge_valid, 
                   cmap='Blues', s=20, alpha=0.8)
        nicaragua_gdf.plot(ax=ax2, edgecolor='black', facecolor='none', linewidth=2)
        ax2.set_title('Valid Interpolation Points', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Longitude', fontsize=12)
        ax2.set_ylabel('Latitude', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Target grid points
        ax3.scatter(grid_lons, grid_lats, c='green', s=10, alpha=0.6)
        nicaragua_gdf.plot(ax=ax3, edgecolor='black', facecolor='none', linewidth=2)
        ax3.set_title('Target Grid Points', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Longitude', fontsize=12)
        ax3.set_ylabel('Latitude', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Interpolated result
        lon_centers = [(lons[i] + lons[i+1]) / 2 for i in range(len(lons) - 1)]
        lat_centers = [(lats[j] + lats[j+1]) / 2 for j in range(len(lats) - 1)]
        interp_lon_mesh, interp_lat_mesh = np.meshgrid(lon_centers, lat_centers)
        
        im4 = ax4.pcolormesh(interp_lon_mesh, interp_lat_mesh, reshaped_surge, 
                             cmap='Blues', shading='auto', alpha=0.8)
        plt.colorbar(im4, ax=ax4, label='Surge Height (m)')
        
        nicaragua_gdf.plot(ax=ax4, edgecolor='black', facecolor='none', linewidth=2)
        ax4.set_title('Interpolated Result', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Longitude', fontsize=12)
        ax4.set_ylabel('Latitude', fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        # Mark high surge locations in interpolated result
        if len(interp_high_indices[0]) > 0:
            interp_high_lats = interp_lat_mesh[interp_high_indices]
            interp_high_lons = interp_lon_mesh[interp_high_indices]
            interp_high_surge = reshaped_surge[interp_high_indices]
            
            ax4.scatter(interp_high_lons, interp_high_lats, c='red', s=30, alpha=0.8, label='High Surge (>0.1m)')
            ax4.legend()
        
        # Save the plot
        output_file = Path("data/preprocessed/surge/interpolation_mapping_debug.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nInterpolation mapping debug visualization saved to: {output_file}")
        
        # Summary
        print(f"\nSummary:")
        print(f"  Computational high surge locations: {len(high_surge_indices[0])}")
        print(f"  Valid interpolation points: {len(comp_lons_valid)}")
        print(f"  Target grid points: {len(grid_lons)}")
        print(f"  Interpolated high surge locations: {len(interp_high_indices[0])}")

if __name__ == "__main__":
    test_interpolation_mapping()
