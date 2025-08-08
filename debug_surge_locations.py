#!/usr/bin/env python3
"""
Debug script to check actual surge locations in the computational grid.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def debug_surge_locations():
    """Debug the actual surge locations in the computational grid."""
    
    # Load existing surge data
    surge_file = Path("data/preprocessed/surge/nicaragua_surge_ensemble_mean.npy")
    if not surge_file.exists():
        print("Surge file not found")
        return
    
    print("Loading surge data...")
    surge_data = np.load(surge_file)
    print(f"Raw surge shape: {surge_data.shape}")
    print(f"Raw surge range: {np.min(surge_data):.3f} to {np.max(surge_data):.3f}")
    
    # Create computational grid coordinates (same as in surge layer)
    comp_minx, comp_maxx = -89.93, -73.83
    comp_miny, comp_maxy = 7.78, 18.97
    
    # Calculate the correct number of points to match surge data shape
    comp_lons = np.linspace(comp_minx, comp_maxx, 349)
    comp_lats = np.linspace(comp_miny, comp_maxy, 249)
    
    # Create computational grid meshgrid
    comp_lon_mesh, comp_lat_mesh = np.meshgrid(comp_lons, comp_lats)
    
    print(f"Computational grid shape: {comp_lon_mesh.shape}")
    print(f"Computational bounds: {np.min(comp_lon_mesh):.3f} to {np.max(comp_lon_mesh):.3f} W, {np.min(comp_lat_mesh):.3f} to {np.max(comp_lat_mesh):.3f} N")
    
    # Find high surge locations
    high_surge_mask = surge_data > 0.5
    high_surge_indices = np.where(high_surge_mask)
    
    print(f"\nHigh surge locations (>0.5m) in computational grid:")
    for i in range(min(20, len(high_surge_indices[0]))):
        lat_idx, lon_idx = high_surge_indices[0][i], high_surge_indices[1][i]
        lat_val = comp_lat_mesh[lat_idx, lon_idx]
        lon_val = comp_lon_mesh[lat_idx, lon_idx]
        surge_val = surge_data[lat_idx, lon_idx]
        print(f"  ({lat_val:.3f}, {lon_val:.3f}): {surge_val:.3f}m")
    
    # Find surge locations near Nicaragua
    nicaragua_lon_min, nicaragua_lon_max = -87.7, -82.7
    nicaragua_lat_min, nicaragua_lat_max = 10.7, 15.0
    
    nicaragua_mask = (
        (comp_lon_mesh >= nicaragua_lon_min) & (comp_lon_mesh <= nicaragua_lon_max) &
        (comp_lat_mesh >= nicaragua_lat_min) & (comp_lat_mesh <= nicaragua_lat_max)
    )
    
    nicaragua_surge_mask = (surge_data > 0.1) & nicaragua_mask
    nicaragua_surge_indices = np.where(nicaragua_surge_mask)
    
    print(f"\nNicaragua region surge locations (>0.1m):")
    for i in range(min(20, len(nicaragua_surge_indices[0]))):
        lat_idx, lon_idx = nicaragua_surge_indices[0][i], nicaragua_surge_indices[1][i]
        lat_val = comp_lat_mesh[lat_idx, lon_idx]
        lon_val = comp_lon_mesh[lat_idx, lon_idx]
        surge_val = surge_data[lat_idx, lon_idx]
        print(f"  ({lat_val:.3f}, {lon_val:.3f}): {surge_val:.3f}m")
    
    # Create a simple visualization of the computational grid surge
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot surge data
    im = ax.pcolormesh(comp_lon_mesh, comp_lat_mesh, surge_data, 
                       cmap='Blues', shading='auto', alpha=0.8)
    plt.colorbar(im, ax=ax, label='Surge Height (m)')
    
    # Add Nicaragua boundary
    from src.utils.hurricane_geom import get_nicaragua_boundary
    nicaragua_gdf = get_nicaragua_boundary()
    nicaragua_gdf.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=2)
    
    # Mark high surge locations
    if len(high_surge_indices[0]) > 0:
        high_lats = comp_lat_mesh[high_surge_indices]
        high_lons = comp_lon_mesh[high_surge_indices]
        high_surge = surge_data[high_surge_indices]
        
        ax.scatter(high_lons, high_lats, c='red', s=50, alpha=0.8, label='High Surge (>0.5m)')
        
        # Add some labels
        for i in range(min(5, len(high_lons))):
            ax.annotate(f'{high_surge[i]:.2f}m', 
                       (high_lons[i], high_lats[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    ax.set_title('Computational Grid Surge Heights', fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Set bounds to focus on Nicaragua region
    ax.set_xlim(-88, -82)
    ax.set_ylim(10.5, 15.5)
    
    # Save the plot
    output_file = Path("data/preprocessed/surge/computational_grid_surge_debug.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nComputational grid surge visualization saved to: {output_file}")

if __name__ == "__main__":
    debug_surge_locations()
