"""
Storm surge impact analysis module.

This module provides functionality to analyze storm surge impacts
on various vulnerability layers (population, schools, etc.).
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
import warnings

from src.impact_analysis.layers.surge import SurgeLayer
from src.utils.config_utils import load_config, get_config_value
from src.utils.path_utils import get_data_path, ensure_directory

warnings.filterwarnings('ignore')


class SurgeImpactAnalysis:
    """Storm surge impact analysis class."""

    def __init__(self, exposure_layer, vulnerability_layer, config):
        """Initialize the surge impact analysis."""
        self.exposure_layer = exposure_layer
        self.vulnerability_layer = vulnerability_layer
        self.config = config
        
    def load_surge_data(self, hurricane_file: str, bathymetry_file: Optional[str] = None) -> bool:
        """Load surge data for analysis."""
        return self.surge_layer.load_data(hurricane_file, bathymetry_file)
        
    def run_surge_analysis(self, 
                          hurricane_file: str,
                          bathymetry_file: Optional[str] = None,
                          n_members: Optional[int] = None,
                          max_hours: Optional[int] = None) -> Dict:
        """Run complete surge analysis."""
        print("Running storm surge impact analysis...")
        print("=" * 50)
        
        # Load data
        if not self.load_surge_data(hurricane_file, bathymetry_file):
            print("Failed to load surge data")
            return {}
            
        # Run ensemble simulation
        results = self.surge_layer.run_ensemble_simulation(n_members, max_hours)
        
        if not results:
            print("No surge results generated")
            return {}
            
        # Analyze results
        analysis_results = self._analyze_results(results)
        
        # Generate outputs
        self._generate_outputs(results, analysis_results)
        
        return {
            'simulation_results': results,
            'analysis_results': analysis_results
        }
        
    def _analyze_results(self, results: Dict) -> Dict:
        """Analyze surge ensemble results."""
        print("Analyzing surge results...")
        
        analysis = {}
        
        # Extract statistics
        max_surges = [r['max_surge'] for r in results.values()]
        coverages = [r['coverage_percent'] for r in results.values()]
        storm_winds = [r['storm_max_wind'] for r in results.values()]
        
        analysis['ensemble_stats'] = {
            'num_members': len(results),
            'max_surge_range': [min(max_surges), max(max_surges)],
            'mean_max_surge': np.mean(max_surges),
            'std_max_surge': np.std(max_surges),
            'coverage_range': [min(coverages), max(coverages)],
            'mean_coverage': np.mean(coverages),
            'storm_wind_range': [min(storm_winds), max(storm_winds)],
            'mean_storm_wind': np.mean(storm_winds)
        }
        
        # Calculate ensemble mean exposure
        all_surge_fields = [r['final_surge_field'] for r in results.values()]
        mean_surge = np.mean(all_surge_fields, axis=0)
        std_surge = np.std(all_surge_fields, axis=0)
        
        analysis['exposure'] = {
            'mean_surge': mean_surge,
            'std_surge': std_surge,
            'max_surge': np.maximum.reduce(all_surge_fields),
            'probability_50cm': np.mean([field > 0.5 for field in all_surge_fields], axis=0),
            'probability_100cm': np.mean([field > 1.0 for field in all_surge_fields], axis=0)
        }
        
        print(f"Analysis complete:")
        print(f"  Members: {len(results)}")
        print(f"  Surge range: {min(max_surges):.2f} - {max(max_surges):.2f}m")
        print(f"  Mean surge: {np.mean(max_surges):.2f} ± {np.std(max_surges):.2f}m")
        print(f"  Coverage range: {min(coverages):.1f}% - {max(coverages):.1f}%")
        
        return analysis
        
    def _generate_outputs(self, results: Dict, analysis: Dict):
        """Generate output files and visualizations."""
        print("Generating outputs...")
        
        # Get output directory
        output_dir = get_config_value(
            self.config, "impact_analysis.output.base_directory", "data/results/impact_analysis"
        )
        surge_output_dir = get_data_path(output_dir) / "surge_ensemble"
        ensure_directory(surge_output_dir)
        
        # Save ensemble summary
        self._save_ensemble_summary(results, surge_output_dir)
        
        # Generate plots
        self._generate_plots(results, analysis, surge_output_dir)
        
        # Save exposure data
        self._save_exposure_data(analysis, surge_output_dir)
        
        print(f"Outputs saved to: {surge_output_dir}")
        
    def _save_ensemble_summary(self, results: Dict, output_dir: Path):
        """Save ensemble summary to CSV."""
        summary_data = []
        for member, data in results.items():
            summary_data.append({
                'ensemble_member': member,
                'max_surge_m': data['max_surge'],
                'coverage_percent': data['coverage_percent'],
                'surge_area_10cm_km2': data['surge_10cm_area'] * (3.0 ** 2),  # Assuming 3km grid
                'surge_area_50cm_km2': data['surge_50cm_area'] * (3.0 ** 2),
                'surge_area_100cm_km2': data['surge_100cm_area'] * (3.0 ** 2),
                'max_wind_ms': data['max_wind'],
                'storm_max_wind_kt': data['storm_max_wind'],
                'storm_min_pressure_mb': data['storm_min_pressure']
            })

        summary_df = pd.DataFrame(summary_data)
        summary_path = output_dir / 'ensemble_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        print(f"  Ensemble summary saved to: {summary_path}")
        
    def _generate_plots(self, results: Dict, analysis: Dict, output_dir: Path):
        """Generate visualization plots."""
        # Generate main surge plot
        plot_path = self.surge_layer.plot(str(output_dir), results)
        print(f"  Main surge plot saved to: {plot_path}")
        
        # Generate additional analysis plots
        self._plot_ensemble_statistics(results, analysis, output_dir)
        
    def _plot_ensemble_statistics(self, results: Dict, analysis: Dict, output_dir: Path):
        """Plot ensemble statistics."""
        import matplotlib.pyplot as plt
        
        # Extract data
        max_surges = [r['max_surge'] for r in results.values()]
        coverages = [r['coverage_percent'] for r in results.values()]
        storm_winds = [r['storm_max_wind'] for r in results.values()]
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Surge distribution
        ax1.hist(max_surges, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Maximum Surge (m)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Surge Distribution')
        ax1.grid(True, alpha=0.3)
        
        # 2. Coverage distribution
        ax2.hist(coverages, bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.set_xlabel('Coverage (%)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Spatial Coverage Distribution')
        ax2.grid(True, alpha=0.3)
        
        # 3. Wind-Surge correlation
        ax3.scatter(storm_winds, max_surges, s=80, alpha=0.7, c='red')
        ax3.set_xlabel('Storm Max Wind (kt)')
        ax3.set_ylabel('Max Surge (m)')
        ax3.set_title('Wind-Surge Relationship')
        ax3.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        if len(storm_winds) > 2:
            correlation = np.corrcoef(storm_winds, max_surges)[0, 1]
            ax3.text(0.05, 0.95, f'r = {correlation:.2f}', transform=ax3.transAxes,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 4. Coverage vs Surge
        ax4.scatter(max_surges, coverages, s=80, alpha=0.7, c='purple')
        ax4.set_xlabel('Max Surge (m)')
        ax4.set_ylabel('Coverage (%)')
        ax4.set_title('Surge vs Coverage')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        stats_path = output_dir / 'ensemble_statistics.png'
        plt.savefig(stats_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  Ensemble statistics plot saved to: {stats_path}")
        
    def _save_exposure_data(self, analysis: Dict, output_dir: Path):
        """Save exposure data to files."""
        # Save mean surge as numpy array
        mean_surge = analysis['exposure']['mean_surge']
        mean_surge_path = output_dir / 'mean_surge.npy'
        np.save(mean_surge_path, mean_surge)
        
        # Save probability maps
        prob_50cm = analysis['exposure']['probability_50cm']
        prob_100cm = analysis['exposure']['probability_100cm']
        
        prob_50cm_path = output_dir / 'probability_50cm.npy'
        prob_100cm_path = output_dir / 'probability_100cm.npy'
        
        np.save(prob_50cm_path, prob_50cm)
        np.save(prob_100cm_path, prob_100cm)
        
        print(f"  Exposure data saved to: {output_dir}")
        
    def get_surge_exposure(self) -> np.ndarray:
        """Get the surge exposure field."""
        return self.surge_layer.get_surge_exposure(self.surge_layer.results if hasattr(self.surge_layer, 'results') else {})
        
    def get_surge_probability(self, threshold: float = 0.5) -> np.ndarray:
        """Get surge probability map for given threshold."""
        if not hasattr(self.surge_layer, 'results') or not self.surge_layer.results:
            return np.zeros_like(self.surge_layer.grid_lats) if self.surge_layer.grid_lats is not None else np.array([])
            
        all_surge_fields = [r['final_surge_field'] for r in self.surge_layer.results.values()]
        return np.mean([field > threshold for field in all_surge_fields], axis=0)
        
    # Required methods for ImpactLayer interface
    def compute_impact(self):
        """Compute the impact grid."""
        if not hasattr(self, 'exposure_layer') or not hasattr(self.exposure_layer, 'results'):
            # Return empty impact grid
            from shapely.geometry import Point
            return gpd.GeoDataFrame({'expected_impact': [0.0]}, geometry=[Point(0, 0)], crs="EPSG:4326")
        
        # Get exposure grid (this is the properly gridded exposure data)
        exposure_grid = self.exposure_layer.compute_grid()
        
        # Get vulnerability grid
        vulnerability_grid = self.vulnerability_layer.compute_grid()
        
        # Calculate impact
        if hasattr(vulnerability_grid, 'geometry') and len(vulnerability_grid) > 0:
            value_col = getattr(self.vulnerability_layer, 'value_column', 'value')
            if value_col in vulnerability_grid.columns and 'probability' in exposure_grid.columns:
                vulnerability_values = vulnerability_grid[value_col].values
                exposure_values = exposure_grid['probability'].values
                
                # Calculate impact as exposure * vulnerability
                impact_values = vulnerability_values * exposure_values
                
                # Create impact GeoDataFrame
                impact_gdf = vulnerability_grid.copy()
                impact_gdf['expected_impact'] = impact_values
                
                return impact_gdf
        
        # Fallback to empty grid
        from shapely.geometry import Point
        return gpd.GeoDataFrame({'expected_impact': [0.0]}, geometry=[Point(0, 0)], crs="EPSG:4326")
        
    def get_plot_metadata(self):
        """Return metadata for plotting."""
        return {
            "layer_type": "impact",
            "hazard_type": "Surge",
            "vulnerability_type": "population",  # This should be dynamic
            "data_column": "expected_impact",
            "colormap": "Reds",
            "title_template": "Number of {vulnerability_type} to be Impacted by Storm Surge",
            "legend_template": "Expected Impact per Cell",
            "filename_template": "surge_impact_{vulnerability_type}_{parameters}",
            "special_features": []
        }
        
    def get_plot_data(self) -> Tuple[str, np.ndarray]:
        """Return data column name and values for plotting."""
        impact_gdf = self.compute_impact()
        return "expected_impact", impact_gdf["expected_impact"].values
        
    def plot(self, ax=None, output_dir="data/results/impact_analysis/"):
        """Plot the impact layer using universal plotting function."""
        from src.impact_analysis.utils.plotting_utils import plot_layer_with_scales
        
        # Use universal plotting function
        plot_layer_with_scales(self, output_dir=output_dir)
        
    def plot_binary_probability(self, output_dir="data/results/impact_analysis/"):
        """Plot binary probability map using universal plotting function."""
        # Create a temporary layer for binary probability plotting
        # This follows the same pattern as other impact layers
        import matplotlib.pyplot as plt
        import os
        from src.utils.hurricane_geom import get_nicaragua_boundary

        # Get exposure and vulnerability grids
        exposure_grid = self.exposure_layer.compute_grid()
        vulnerability_grid = self.vulnerability_layer.compute_grid()
        value_col = getattr(self.vulnerability_layer, "value_column", "value")

        # Create binary vulnerability mask (1 if any entities present, 0 otherwise)
        binary_vulnerability = (vulnerability_grid[value_col] > 0).astype(int)

        # Combine exposure probability with binary vulnerability
        binary_probability = exposure_grid["probability"] * binary_vulnerability

        # Create plot
        nicaragua_gdf = get_nicaragua_boundary()

        # Get vulnerability name
        if hasattr(self.vulnerability_layer, "__class__"):
            vuln_name = (
                self.vulnerability_layer.__class__.__name__.replace(
                    "VulnerabilityLayer", ""
                ).lower()
                or "vulnerability"
            )
        else:
            vuln_name = "vulnerability"

        # Get vulnerability parameters for more detailed naming
        vuln_params = ""
        if hasattr(self.vulnerability_layer, "age_groups") and hasattr(
            self.vulnerability_layer, "gender"
        ):
            age_str = "_".join(map(str, self.vulnerability_layer.age_groups))
            vuln_params = f"_{self.vulnerability_layer.gender}_ages_{age_str}"

        fig, ax = plt.subplots(figsize=(12, 10))

        # Create GeoDataFrame for plotting
        plot_gdf = exposure_grid.copy()
        plot_gdf["binary_probability"] = binary_probability

        plot_gdf.plot(
            ax=ax,
            column="binary_probability",
            cmap="YlOrRd",
            linewidth=0.1,
            edgecolor="grey",
            alpha=0.7,
            legend=True,
            legend_kwds={
                "label": f"Probability of At Least One {vuln_name.title()} Affected"
            },
        )

        if nicaragua_gdf is not None:
            nicaragua_gdf.plot(
                ax=ax, color="none", edgecolor="black", linewidth=3, alpha=1.0
            )

        ax.set_title(
            f"Binary Probability of {vuln_name.title()} Impact\n(At Least One Affected per Grid Cell)"
        )
        plt.tight_layout()

        out_path = os.path.join(
            output_dir, f"binary_probability_{vuln_name}{vuln_params}.png"
        )
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved binary probability plot: {out_path}")
        plt.close(fig)

        # Print summary statistics
        total_cells_with_vulnerability = binary_vulnerability.sum()
        cells_with_risk = (binary_probability > 0).sum()
        max_probability = binary_probability.max()
        mean_probability = binary_probability.mean()

        print(f"\nBinary Probability Summary for {vuln_name}:")
        print(f"  Total grid cells with {vuln_name}s: {total_cells_with_vulnerability}")
        print(f"  Grid cells at risk (probability > 0): {cells_with_risk}")
        print(f"  Maximum probability in any cell: {max_probability:.3f}")
        print(f"  Mean probability across all cells: {mean_probability:.3f}")
        
    def expected_impact(self) -> float:
        """Compute the expected number of affected entities."""
        if not hasattr(self, 'exposure_layer') or not hasattr(self.exposure_layer, 'results'):
            return 0.0
            
        # Get exposure and vulnerability grids
        exposure_grid = self.exposure_layer.compute_grid()
        vulnerability_grid = self.vulnerability_layer.compute_grid()
        
        # Calculate expected impact
        if hasattr(vulnerability_grid, 'geometry') and len(vulnerability_grid) > 0:
            value_col = getattr(self.vulnerability_layer, 'value_column', 'value')
            if value_col in vulnerability_grid.columns and 'probability' in exposure_grid.columns:
                vulnerability_values = vulnerability_grid[value_col].values
                exposure_values = exposure_grid['probability'].values
                
                # Calculate impact as sum of exposure * vulnerability
                impact_values = vulnerability_values * exposure_values
                return np.sum(impact_values)
        
        return 0.0
        
    def best_case(self) -> float:
        """Compute the best-case number of affected entities."""
        # For surge, best case is typically lower impact
        expected = self.expected_impact()
        return expected * 0.5  # 50% of expected impact
        
    def worst_case(self) -> float:
        """Compute the worst-case number of affected entities."""
        # For surge, worst case is typically higher impact
        expected = self.expected_impact()
        return expected * 2.0  # 200% of expected impact
        
    def save_impact_summary(self, output_dir="data/results/impact_analysis/"):
        """Save impact analysis summary to file."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create summary file
        summary_file = output_path / "surge_impact_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("Storm Surge Impact Analysis Summary\n")
            f.write("=" * 40 + "\n\n")
            
            # Basic statistics
            f.write(f"Expected Impact: {self.expected_impact():.2f}\n")
            f.write(f"Best Case (min): {self.best_case():.2f}\n")
            f.write(f"Worst Case (max): {self.worst_case():.2f}\n\n")
            
            # Add surge-specific information if available
            if hasattr(self, 'exposure_layer') and hasattr(self.exposure_layer, 'results'):
                results = self.exposure_layer.results
                if results:
                    max_surges = [r['max_surge'] for r in results.values()]
                    coverages = [r['coverage_percent'] for r in results.values()]
                    
                    f.write("Surge Ensemble Statistics:\n")
                    f.write(f"  Number of ensemble members: {len(results)}\n")
                    f.write(f"  Max surge range: {min(max_surges):.3f} - {max(max_surges):.3f} m\n")
                    f.write(f"  Mean max surge: {np.mean(max_surges):.3f} m\n")
                    f.write(f"  Coverage range: {min(coverages):.1f} - {max(coverages):.1f}%\n")
                    f.write(f"  Mean coverage: {np.mean(coverages):.1f}%\n")
        
        print(f"Impact summary saved to: {summary_file}")
        return str(summary_file)


def run_surge_impact_analysis(hurricane_file: str,
                             bathymetry_file: Optional[str] = None,
                             n_members: Optional[int] = None,
                             max_hours: Optional[int] = None) -> Dict:
    """
    Run storm surge impact analysis.
    
    Parameters:
    - hurricane_file: Path to hurricane ensemble CSV file
    - bathymetry_file: Path to bathymetry file (optional, will auto-detect)
    - n_members: Number of ensemble members to process
    - max_hours: Maximum simulation hours
    
    Returns:
    - Dictionary containing analysis results
    """
    analyzer = SurgeImpactAnalysis()
    return analyzer.run_surge_analysis(hurricane_file, bathymetry_file, n_members, max_hours)
