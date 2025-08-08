#!/usr/bin/env python3
"""
Test coordinate mapping during interpolation.
"""

import numpy as np
from pathlib import Path

def test_coordinate_mapping():
    """Test coordinate mapping during interpolation."""
    
    # Load existing surge data
    surge_file = Path("data/preprocessed/surge/nicaragua_surge_ensemble_mean.npy")
    if not surge_file.exists():
        print("Surge file not found")
        return
    
    print("Loading surge data...")
    surge_data = np.load(surge_file)
    
    # Create computational grid coordinates
    comp_minx, comp_maxx = -89.93, -73.83
    comp_miny, comp_maxy = 7.78, 18.97
    
    comp_lons = np.linspace(comp_minx, comp_maxx, 349)
    comp_lats = np.linspace(comp_miny, comp_maxy, 249)
    
    comp_lon_mesh, comp_lat_mesh = np.meshgrid(comp_lons, comp_lats)
    
    # Find some specific surge locations in Nicaragua region
    nicaragua_lon_min, nicaragua_lon_max = -87.7, -82.7
    nicaragua_lat_min, nicaragua_lat_max = 10.7, 15.0
    
    nicaragua_mask = (
        (comp_lon_mesh >= nicaragua_lon_min) & (comp_lon_mesh <= nicaragua_lon_max) &
        (comp_lat_mesh >= nicaragua_lat_min) & (comp_lat_mesh <= nicaragua_lat_max)
    )
    
    nicaragua_surge_mask = (surge_data > 0.1) & nicaragua_mask
    nicaragua_surge_indices = np.where(nicaragua_surge_mask)
    
    print("Sample surge locations in computational grid:")
    for i in range(min(5, len(nicaragua_surge_indices[0]))):
        lat_idx, lon_idx = nicaragua_surge_indices[0][i], nicaragua_surge_indices[1][i]
        lat_val = comp_lat_mesh[lat_idx, lon_idx]
        lon_val = comp_lon_mesh[lat_idx, lon_idx]
        surge_val = surge_data[lat_idx, lon_idx]
        print(f"  ({lat_val:.3f}, {lon_val:.3f}): {surge_val:.3f}m")
    
    # Now test the interpolation step
    from scipy.interpolate import griddata
    
    # Get the surge data for interpolation
    comp_lons_valid = comp_lon_mesh[nicaragua_surge_mask]
    comp_lats_valid = comp_lat_mesh[nicaragua_surge_mask]
    comp_surge_valid = surge_data[nicaragua_surge_mask]
    
    print(f"\nInterpolation test:")
    print(f"  Source points: {len(comp_lons_valid)}")
    print(f"  Source coordinate range: {np.min(comp_lons_valid):.3f} to {np.max(comp_lons_valid):.3f} W")
    print(f"  Source coordinate range: {np.min(comp_lats_valid):.3f} to {np.max(comp_lats_valid):.3f} N")
    
    # Create some test target points
    test_lons = [-85.7, -83.5, -84.5]  # Pacific, Caribbean, inland
    test_lats = [10.7, 10.7, 14.2]
    
    print(f"\nTest target points:")
    for i, (lon, lat) in enumerate(zip(test_lons, test_lats)):
        print(f"  Target {i}: ({lat:.3f}, {lon:.3f})")
    
    # Create interpolation points
    points = np.column_stack([comp_lons_valid, comp_lats_valid])
    values = comp_surge_valid
    
    # Create target points
    xi = np.column_stack([test_lons, test_lats])
    
    # Interpolate
    interpolated_values = griddata(points, values, xi, method='nearest', fill_value=0.0)
    
    print(f"\nInterpolation results:")
    for i, (lon, lat) in enumerate(zip(test_lons, test_lats)):
        print(f"  Target {i}: ({lat:.3f}, {lon:.3f}) -> {interpolated_values[i]:.3f}m")
    
    # Check if the coordinates are being interpreted correctly
    print(f"\nCoordinate interpretation check:")
    print(f"  Points shape: {points.shape}")
    print(f"  Xi shape: {xi.shape}")
    print(f"  Sample points (first 3):")
    for i in range(min(3, len(points))):
        print(f"    ({points[i, 1]:.3f}, {points[i, 0]:.3f})")  # lat, lon
    print(f"  Sample xi (first 3):")
    for i in range(min(3, len(xi))):
        print(f"    ({xi[i, 1]:.3f}, {xi[i, 0]:.3f})")  # lat, lon

if __name__ == "__main__":
    test_coordinate_mapping()
