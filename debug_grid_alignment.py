#!/usr/bin/env python3
"""
Debug script to check grid alignment between surge exposure and population vulnerability.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.impact_analysis.layers.surge import SurgeLayer
from src.impact_analysis.layers.population import PopulationVulnerabilityLayer
from src.impact_analysis.analysis.surge_impact import SurgeImpactAnalysis
from src.utils.config_utils import load_config
import geopandas as gpd
import numpy as np

def debug_grid_alignment():
    """Debug grid alignment between surge exposure and population vulnerability."""
    
    print("=== Debugging Grid Alignment ===\n")
    
    # Load configuration
    config = load_config("config/impact_analysis_config.yaml")
    
    # Initialize layers
    print("1. Initializing surge layer...")
    surge_layer = SurgeLayer(
        hurricane_file="data/preprocessed/weatherlab/synthetic/processed_FNV3_2024_11_04_00_00_ensemble_data_synthetic.csv",
        bathymetry_file="data/raw/bathymetry/nic_bathymetry_2024_gebco_large.tif"
    )
    
    print("2. Initializing population vulnerability layer...")
    population_layer = PopulationVulnerabilityLayer(config)
    
    print("3. Getting grids...")
    exposure_grid = surge_layer.compute_grid()
    vulnerability_grid = population_layer.compute_grid()
    
    print(f"\n4. Grid Properties:")
    print(f"   Exposure grid: {len(exposure_grid)} cells")
    print(f"   Vulnerability grid: {len(vulnerability_grid)} cells")
    print(f"   Exposure grid bounds: {exposure_grid.total_bounds}")
    print(f"   Vulnerability grid bounds: {vulnerability_grid.total_bounds}")
    
    print(f"\n5. Exposure Grid Details:")
    print(f"   Probability range: {exposure_grid['probability'].min():.3f} to {exposure_grid['probability'].max():.3f}")
    print(f"   Cells with surge > 0: {np.sum(exposure_grid['probability'] > 0)}")
    print(f"   Cells with surge > 0.1: {np.sum(exposure_grid['probability'] > 0.1)}")
    print(f"   Cells with surge > 0.5: {np.sum(exposure_grid['probability'] > 0.5)}")
    
    print(f"\n6. Vulnerability Grid Details:")
    value_col = getattr(population_layer, "value_column", "population_count")
    print(f"   Value column: {value_col}")
    print(f"   Population range: {vulnerability_grid[value_col].min():.0f} to {vulnerability_grid[value_col].max():.0f}")
    print(f"   Cells with population > 0: {np.sum(vulnerability_grid[value_col] > 0)}")
    print(f"   Total population: {vulnerability_grid[value_col].sum():.0f}")
    
    print(f"\n7. Testing Impact Calculation...")
    surge_impact = SurgeImpactAnalysis(surge_layer, population_layer, config)
    impact_gdf = surge_impact.compute_impact()
    
    print(f"   Impact grid: {len(impact_gdf)} cells")
    print(f"   Impact range: {impact_gdf['expected_impact'].min():.3f} to {impact_gdf['expected_impact'].max():.3f}")
    print(f"   Cells with impact > 0: {np.sum(impact_gdf['expected_impact'] > 0)}")
    print(f"   Total expected impact: {impact_gdf['expected_impact'].sum():.3f}")
    
    # Check if grids have same geometry
    print(f"\n8. Grid Geometry Check:")
    if len(exposure_grid) == len(vulnerability_grid):
        print(f"   ✓ Grids have same number of cells: {len(exposure_grid)}")
        
        # Check if geometries match
        geom_match = 0
        for i in range(min(10, len(exposure_grid))):  # Check first 10 cells
            if exposure_grid.iloc[i].geometry.equals(vulnerability_grid.iloc[i].geometry):
                geom_match += 1
        
        print(f"   Geometry match (first 10 cells): {geom_match}/10")
        
        if geom_match == 10:
            print(f"   ✓ Grids appear to have matching geometry")
        else:
            print(f"   ✗ Grids have different geometry")
    else:
        print(f"   ✗ Grids have different number of cells")
        print(f"      This is the problem! Grids must have same geometry for standard impact calculation.")
    
    print(f"\n=== Debug Complete ===")

if __name__ == "__main__":
    debug_grid_alignment()
