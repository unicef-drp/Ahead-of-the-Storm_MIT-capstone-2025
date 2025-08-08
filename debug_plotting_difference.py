#!/usr/bin/env python3
"""
Debug script to compare data used by compute_grid() vs _get_interpolated_surge_data()
to identify why main exposure plotting differs from mean/max visualizations.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.impact_analysis.layers.surge import SurgeLayer
import numpy as np

def debug_plotting_difference():
    """Debug the difference between compute_grid() and _get_interpolated_surge_data()."""
    
    print("=== Debugging Plotting Data Difference ===\n")
    
    # Initialize surge layer
    surge_layer = SurgeLayer(
        hurricane_file="data/preprocessed/weatherlab/synthetic/processed_FNV3_2024_11_04_00_00_ensemble_data_synthetic.csv",
        bathymetry_file="data/raw/bathymetry/nic_bathymetry_2024_gebco_large.tif"
    )
    
    print("1. Getting data from _get_interpolated_surge_data() (used by mean/max)...")
    
    # Get data used by mean/max visualizations
    interpolated_surge = surge_layer._get_interpolated_surge_data()
    print(f"   Interpolated surge shape: {interpolated_surge.shape}")
    print(f"   Interpolated surge range: {interpolated_surge.min():.3f} to {interpolated_surge.max():.3f}")
    print(f"   Interpolated cells > 0: {np.sum(interpolated_surge > 0)}")
    print(f"   Interpolated cells > 0.1: {np.sum(interpolated_surge > 0.1)}")
    print(f"   Interpolated cells > 0.5: {np.sum(interpolated_surge > 0.5)}")
    
    print("\n2. Getting data from compute_grid() (used by main exposure plotting)...")
    
    # Get data used by main exposure plotting
    grid_gdf = surge_layer.compute_grid()
    if grid_gdf is not None and len(grid_gdf) > 0:
        exposure_data = grid_gdf['probability'].values
        print(f"   Exposure data shape: {exposure_data.shape}")
        print(f"   Exposure data range: {exposure_data.min():.3f} to {exposure_data.max():.3f}")
        print(f"   Exposure cells > 0: {np.sum(exposure_data > 0)}")
        print(f"   Exposure cells > 0.1: {np.sum(exposure_data > 0.1)}")
        print(f"   Exposure cells > 0.5: {np.sum(exposure_data > 0.5)}")
        
        # Reshape exposure data to compare with interpolated surge
        if len(exposure_data) == interpolated_surge.size:
            exposure_reshaped = exposure_data.reshape(interpolated_surge.shape)
            print(f"   Exposure reshaped shape: {exposure_reshaped.shape}")
            
            # Compare the two datasets
            if np.array_equal(interpolated_surge, exposure_reshaped):
                print("   ✓ Exposure data exactly matches interpolated surge data!")
            else:
                print("   ⚠ Exposure data does NOT match interpolated surge data")
                print(f"   Difference in cells > 0: {np.sum(interpolated_surge > 0) - np.sum(exposure_reshaped > 0)}")
                print(f"   Difference in cells > 0.1: {np.sum(interpolated_surge > 0.1) - np.sum(exposure_reshaped > 0.1)}")
                
                # Check if it's a threshold issue
                if np.sum(exposure_reshaped > 0) < np.sum(interpolated_surge > 0):
                    print("   ⚠ Exposure data has fewer non-zero cells - may be applying threshold")
                else:
                    print("   ⚠ Exposure data has more non-zero cells - unexpected")
        else:
            print(f"   ⚠ Cannot reshape exposure data: {len(exposure_data)} vs {interpolated_surge.size}")
    else:
        print("   ✗ Failed to get exposure grid data")
    
    print("\n3. Testing get_plot_data() method...")
    
    # Test the get_plot_data method that's used by universal plotting
    try:
        data_column, data_values = surge_layer.get_plot_data()
        print(f"   Plot data column: {data_column}")
        print(f"   Plot data shape: {data_values.shape}")
        print(f"   Plot data range: {data_values.min():.3f} to {data_values.max():.3f}")
        print(f"   Plot cells > 0: {np.sum(data_values > 0)}")
        
        # Compare with exposure data
        if np.array_equal(data_values, exposure_data):
            print("   ✓ Plot data matches exposure data")
        else:
            print("   ⚠ Plot data does NOT match exposure data")
            
    except Exception as e:
        print(f"   ✗ Error in get_plot_data(): {e}")
    
    print("\n4. Testing mean/max visualization data...")
    
    # Test mean/max visualization data
    try:
        mean_surge = surge_layer._get_interpolated_surge_data()
        max_surge = surge_layer._get_max_surge_data()
        
        print(f"   Mean surge cells > 0: {np.sum(mean_surge > 0)}")
        print(f"   Max surge cells > 0: {np.sum(max_surge > 0) if max_surge is not None else 'N/A'}")
        
        # Check if mean/max use same data as interpolated
        if np.array_equal(mean_surge, interpolated_surge):
            print("   ✓ Mean visualization uses same data as interpolated surge")
        else:
            print("   ⚠ Mean visualization uses different data than interpolated surge")
            
    except Exception as e:
        print(f"   ✗ Error in mean/max visualization: {e}")
    
    print("\n=== Debug Summary ===")
    print("The issue is likely that:")
    print("1. Main exposure plotting uses compute_grid() data")
    print("2. Mean/max visualizations use _get_interpolated_surge_data() data")
    print("3. These two methods may be using different data sources or processing")

if __name__ == "__main__":
    debug_plotting_difference()
