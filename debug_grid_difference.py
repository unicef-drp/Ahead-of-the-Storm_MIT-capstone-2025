#!/usr/bin/env python3
"""
Detailed debug script to check for differences in grid coordinates, data ordering, 
or coordinate systems between compute_grid() and _get_interpolated_surge_data().
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.impact_analysis.layers.surge import SurgeLayer
import numpy as np

def debug_grid_difference():
    """Debug grid coordinate and data ordering differences."""
    
    print("=== Debugging Grid Coordinate Differences ===\n")
    
    # Initialize surge layer
    surge_layer = SurgeLayer(
        hurricane_file="data/preprocessed/weatherlab/synthetic/processed_FNV3_2024_11_04_00_00_ensemble_data_synthetic.csv",
        bathymetry_file="data/raw/bathymetry/nic_bathymetry_2024_gebco_large.tif"
    )
    
    print("1. Checking _get_interpolated_surge_data() grid...")
    
    # Get interpolated surge data
    interpolated_surge = surge_layer._get_interpolated_surge_data()
    print(f"   Interpolated surge shape: {interpolated_surge.shape}")
    print(f"   Interpolated surge size: {interpolated_surge.size}")
    print(f"   Interpolated cells > 0: {np.sum(interpolated_surge > 0)}")
    
    # Check the grid coordinates used in interpolation
    from src.utils.hurricane_geom import get_nicaragua_boundary
    nicaragua_gdf = get_nicaragua_boundary()
    minx, miny, maxx, maxy = nicaragua_gdf.total_bounds
    
    resolution = 0.1
    lons = np.arange(minx, maxx + resolution, resolution)
    lats = np.arange(miny, maxy + resolution, resolution)
    
    print(f"   Grid bounds: {minx:.3f} to {maxx:.3f} W, {miny:.3f} to {maxy:.3f} N")
    print(f"   Grid dimensions: {len(lons)-1} x {len(lats)-1} = {(len(lons)-1) * (len(lats)-1)}")
    
    print("\n2. Checking compute_grid() grid...")
    
    # Get exposure grid
    grid_gdf = surge_layer.compute_grid()
    if grid_gdf is not None and len(grid_gdf) > 0:
        exposure_data = grid_gdf['probability'].values
        print(f"   Exposure data shape: {exposure_data.shape}")
        print(f"   Exposure data size: {exposure_data.size}")
        print(f"   Exposure cells > 0: {np.sum(exposure_data > 0)}")
        
        # Check if the grid dimensions match
        expected_size = (len(lons)-1) * (len(lats)-1)
        print(f"   Expected grid size: {expected_size}")
        print(f"   Actual exposure size: {len(exposure_data)}")
        
        if len(exposure_data) == expected_size:
            print("   ✓ Grid dimensions match")
        else:
            print("   ⚠ Grid dimensions do NOT match!")
            
        # Check if the data can be reshaped to match interpolated surge
        if len(exposure_data) == interpolated_surge.size:
            exposure_reshaped = exposure_data.reshape(interpolated_surge.shape)
            print(f"   ✓ Can reshape exposure data to {exposure_reshaped.shape}")
            
            # Check for any differences in the data
            if np.array_equal(interpolated_surge, exposure_reshaped):
                print("   ✓ Data is identical after reshaping")
            else:
                print("   ⚠ Data is different after reshaping")
                
                # Check for differences in non-zero values
                interpolated_nonzero = interpolated_surge > 0
                exposure_nonzero = exposure_reshaped > 0
                
                print(f"   Interpolated nonzero positions: {np.sum(interpolated_nonzero)}")
                print(f"   Exposure nonzero positions: {np.sum(exposure_nonzero)}")
                
                # Check if the difference is in the pattern
                pattern_diff = np.sum(interpolated_nonzero != exposure_nonzero)
                print(f"   Pattern differences: {pattern_diff}")
                
                if pattern_diff == 0:
                    print("   ✓ Patterns are identical")
                else:
                    print("   ⚠ Patterns are different")
                    
                    # Show some examples of differences
                    diff_positions = np.where(interpolated_nonzero != exposure_nonzero)
                    if len(diff_positions[0]) > 0:
                        print(f"   First 5 differences:")
                        for i in range(min(5, len(diff_positions[0]))):
                            row, col = diff_positions[0][i], diff_positions[1][i]
                            print(f"     Position ({row}, {col}): interpolated={interpolated_surge[row, col]:.3f}, exposure={exposure_reshaped[row, col]:.3f}")
        else:
            print(f"   ⚠ Cannot reshape: {len(exposure_data)} vs {interpolated_surge.size}")
    else:
        print("   ✗ Failed to get exposure grid")
    
    print("\n3. Checking coordinate systems...")
    
    # Check if there are any differences in how coordinates are handled
    print("   Interpolated surge uses:")
    print("     - Nicaragua bounds from get_nicaragua_boundary()")
    print("     - 0.1 degree resolution")
    print("     - Grid cell centers for visualization")
    
    print("   Exposure grid uses:")
    print("     - Same Nicaragua bounds")
    print("     - Same 0.1 degree resolution")
    print("     - Grid cell geometries for GeoDataFrame")
    
    print("\n4. Testing data consistency...")
    
    # Test if the issue is in the data processing
    try:
        # Get the raw interpolated data again
        raw_interpolated = surge_layer._get_interpolated_surge_data()
        
        # Get the exposure data again
        exposure_gdf = surge_layer.compute_grid()
        exposure_values = exposure_gdf['probability'].values
        
        # Reshape exposure to match interpolated
        if len(exposure_values) == raw_interpolated.size:
            exposure_reshaped = exposure_values.reshape(raw_interpolated.shape)
            
            # Check for any numerical differences
            max_diff = np.max(np.abs(raw_interpolated - exposure_reshaped))
            mean_diff = np.mean(np.abs(raw_interpolated - exposure_reshaped))
            
            print(f"   Maximum difference: {max_diff:.6f}")
            print(f"   Mean difference: {mean_diff:.6f}")
            
            if max_diff < 1e-10:
                print("   ✓ Data is numerically identical")
            else:
                print("   ⚠ Data has numerical differences")
                
        else:
            print("   ⚠ Cannot compare data due to size mismatch")
            
    except Exception as e:
        print(f"   ✗ Error in data consistency test: {e}")
    
    print("\n=== Debug Summary ===")
    print("The issue might be:")
    print("1. Different coordinate systems or grid ordering")
    print("2. Different data processing in the two methods")
    print("3. Different visualization parameters")

if __name__ == "__main__":
    debug_grid_difference()
