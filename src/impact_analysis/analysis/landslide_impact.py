import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path

from src.impact_analysis.layers.base import ImpactLayer
from src.utils.config_utils import get_config_value
from src.utils.path_utils import get_results_path


class LandslideImpactLayer(ImpactLayer):
    """
    Landslide impact layer that computes impact metrics for landslide exposure scenarios.
    Supports mean, min, and max scenarios from the exposure layer.
    """
    
    def __init__(self, exposure_layer, vulnerability_layer, config):
        super().__init__(exposure_layer, vulnerability_layer, config)
        self.scenario = exposure_layer.resampling_method  # "mean", "min", or "max"
        self.vulnerability_type = vulnerability_layer.vulnerability_type
        
    def expected_impact(self):
        """Compute expected impact for landslide scenario."""
        if self._impact_grid is None:
            self._compute_impact()
        
        # Sum all impact values
        total_impact = np.sum(self._impact_grid)
        return total_impact
    
    def _compute_impact(self):
        """Compute impact grid by multiplying exposure and vulnerability."""
        if self._impact_grid is not None:
            return self._impact_grid
            
        # Get exposure and vulnerability grids
        exposure_grid = self.exposure_layer.get_probability_grid()
        vulnerability_grid = self.vulnerability_layer.get_vulnerability_grid()
        
        # Ensure grids have same shape
        if exposure_grid.shape != vulnerability_grid.shape:
            raise ValueError(f"Exposure grid shape {exposure_grid.shape} != vulnerability grid shape {vulnerability_grid.shape}")
        
        # Compute impact: probability * vulnerability
        self._impact_grid = exposure_grid * vulnerability_grid
        
        return self._impact_grid
    
    def plot(self, ax=None, output_dir="data/results/impact_analysis/"):
        """Plot landslide impact with scenario-specific naming."""
        if self._impact_grid is None:
            self._compute_impact()
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with scenario
        filename = f"landslide_impact_{self.vulnerability_type}_{self.scenario}.png"
        filepath = output_path / filename
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Get grid geometry for plotting
        grid_gdf = self.exposure_layer.get_grid_cells()
        
        # Create impact GeoDataFrame
        impact_gdf = grid_gdf.copy()
        impact_gdf['impact'] = self._impact_grid
        
        # Plot impact values
        impact_gdf.plot(
            column='impact',
            ax=ax,
            cmap='Reds',
            legend=True,
            legend_kwds={'label': 'Expected Impact', 'shrink': 0.8}
        )
        
        # Add title with scenario
        scenario_title = self.scenario.capitalize()
        ax.set_title(f'Landslide Impact on {self.vulnerability_type.replace("_", " ").title()} ({scenario_title} Scenario)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved landslide impact plot: {filepath}")
        return filepath
    
    def plot_binary_probability(self, output_dir="data/results/impact_analysis/"):
        """Plot binary probability of impact (impact > 0) with scenario-specific naming."""
        if self._impact_grid is None:
            self._compute_impact()
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with scenario
        filename = f"binary_probability_{self.vulnerability_type}_{self.scenario}.png"
        filepath = output_path / filename
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Get grid geometry for plotting
        grid_gdf = self.exposure_layer.get_grid_cells()
        
        # Create binary impact GeoDataFrame
        binary_gdf = grid_gdf.copy()
        binary_gdf['has_impact'] = (self._impact_grid > 0).astype(int)
        
        # Plot binary values
        binary_gdf.plot(
            column='has_impact',
            ax=ax,
            cmap='RdYlBu_r',
            legend=True,
            legend_kwds={'label': 'Has Impact (1=Yes, 0=No)', 'shrink': 0.8}
        )
        
        # Add title with scenario
        scenario_title = self.scenario.capitalize()
        ax.set_title(f'Binary Impact Probability - {self.vulnerability_type.replace("_", " ").title()} ({scenario_title} Scenario)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved binary probability plot: {filepath}")
        return filepath 