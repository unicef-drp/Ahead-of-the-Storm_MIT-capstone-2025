#!/usr/bin/env python3
"""
Debug the reshape operation for interpolated surge data.
"""

import numpy as np
from pathlib import Path

def test_reshape_debug():
    """Debug the reshape operation."""
    
    # Load the interpolated data
    interpolated_file = Path("data/preprocessed/surge/nicaragua_surge_interpolated_heights.npy")
    if not interpolated_file.exists():
        print("Interpolated surge file not found")
        return
    
    interpolated_surge = np.load(interpolated_file)
    print(f"Interpolated surge shape: {interpolated_surge.shape}")
    print(f"Interpolated surge range: {np.min(interpolated_surge):.3f} to {np.max(interpolated_surge):.3f}")
    
    # Get the grid parameters
    from src.utils.hurricane_geom import get_nicaragua_boundary
    nicaragua_gdf = get_nicaragua_boundary()
    minx, miny, maxx, maxy = nicaragua_gdf.total_bounds
    
    resolution = 0.1
    lons = np.arange(minx, maxx + resolution, resolution)
    lats = np.arange(miny, maxy + resolution, resolution)
    
    print(f"Grid parameters:")
    print(f"  Nicaragua bounds: {minx:.3f} to {maxx:.3f} W, {miny:.3f} to {maxy:.3f} N")
    print(f"  Lons: {len(lons)} points ({lons[0]:.3f} to {lons[-1]:.3f})")
    print(f"  Lats: {len(lats)} points ({lats[0]:.3f} to {lats[-1]:.3f})")
    print(f"  Expected grid shape: {len(lats)-1} x {len(lons)-1}")
    print(f"  Expected total points: {(len(lats)-1) * (len(lons)-1)}")
    
    # Check if the reshape is correct
    expected_shape = (len(lats)-1, len(lons)-1)
    actual_shape = interpolated_surge.shape
    
    print(f"\nShape comparison:")
    print(f"  Expected: {expected_shape}")
    print(f"  Actual: {actual_shape}")
    print(f"  Match: {expected_shape == actual_shape}")
    
    # Check the coordinate mapping
    print(f"\nCoordinate mapping check:")
    print(f"  Grid cell centers:")
    for i in range(min(5, len(lats)-1)):
        for j in range(min(5, len(lons)-1)):
            lat_center = (lats[i] + lats[i+1]) / 2
            lon_center = (lons[j] + lons[j+1]) / 2
            surge_val = interpolated_surge[i, j]
            if surge_val > 0.1:
                print(f"    Grid[{i},{j}] ({lat_center:.3f}, {lon_center:.3f}): {surge_val:.3f}m")
    
    # Check if the coordinates are in the right order
    print(f"\nCoordinate order check:")
    print(f"  First few lat centers: {[(lats[i] + lats[i+1]) / 2 for i in range(5)]}")
    print(f"  First few lon centers: {[(lons[i] + lons[i+1]) / 2 for i in range(5)]}")
    
    # Check if the issue is with the coordinate order in the meshgrid
    lon_mesh, lat_mesh = np.meshgrid(lons[:-1], lats[:-1])
    print(f"\nMeshgrid check:")
    print(f"  Lon mesh shape: {lon_mesh.shape}")
    print(f"  Lat mesh shape: {lat_mesh.shape}")
    print(f"  Lon mesh range: {np.min(lon_mesh):.3f} to {np.max(lon_mesh):.3f}")
    print(f"  Lat mesh range: {np.min(lat_mesh):.3f} to {np.max(lat_mesh):.3f}")

if __name__ == "__main__":
    test_reshape_debug()
