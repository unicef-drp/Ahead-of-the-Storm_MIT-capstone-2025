#!/usr/bin/env python3
"""
Landslide impact analysis layer.
"""

import os
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from src.impact_analysis.analysis.hurricane_impact import HurricaneImpactLayer
from src.utils.config_utils import get_config_value
from src.utils.hurricane_geom import get_nicaragua_boundary


class LandslideImpactLayer(HurricaneImpactLayer):
    """Landslide impact layer that inherits from hurricane impact layer."""
    
    def _get_vulnerability_name(self):
        """Get vulnerability name for factory function."""
        class_name = self.vulnerability_layer.__class__.__name__
        # Map class names to factory names
        name_mapping = {
            "SchoolVulnerabilityLayer": "schools",
            "SchoolPopulationVulnerabilityLayer": "school_population",
            "PopulationVulnerabilityLayer": "population",
            "PovertyVulnerabilityLayer": "poverty",
            "SeverePovertyVulnerabilityLayer": "severe_poverty",
            "ShelterVulnerabilityLayer": "shelters",
            "HealthFacilityVulnerabilityLayer": "health_facilities"
        }
        
        # Special handling for HealthFacilityVulnerabilityLayer
        if class_name == "HealthFacilityVulnerabilityLayer":
            if hasattr(self.vulnerability_layer, 'weighted_by_population') and self.vulnerability_layer.weighted_by_population:
                return "health_facilities_population"
            else:
                return "health_facilities"
        
        # Special handling for ShelterVulnerabilityLayer
        if class_name == "ShelterVulnerabilityLayer":
            if hasattr(self.vulnerability_layer, 'weighted_by_capacity') and self.vulnerability_layer.weighted_by_capacity:
                return "shelters_population"
            else:
                return "shelters"
        
        # Special handling for PopulationVulnerabilityLayer
        if class_name == "PopulationVulnerabilityLayer":
            if hasattr(self.vulnerability_layer, 'age_groups'):
                # Check if it's children (ages 0, 5, 10, 15) or total population
                if self.vulnerability_layer.age_groups == [0, 5, 10, 15]:
                    return "children"
                else:
                    return "population"
        
        return name_mapping.get(class_name, "schools")

    def compute_impact(self):
        """Override to use high-resolution computation grid."""
        # Use high-resolution grids for computation
        exposure_grid = self.exposure_layer.get_computation_grid()
        vulnerability_grid = self.vulnerability_layer.compute_grid()
        
        # Ensure grids have same length
        if len(exposure_grid) != len(vulnerability_grid):
            raise ValueError(f"Grid length mismatch: exposure={len(exposure_grid)}, vulnerability={len(vulnerability_grid)}")
        
        # Get vulnerability values
        value_col = getattr(self.vulnerability_layer, "value_column", "school_count")
        vulnerability = vulnerability_grid[value_col].values
        
        # For ensemble-based approach, use mean probability for expected impact
        probability = exposure_grid["probability"].values
        
        # Create impact grid
        impact_gdf = exposure_grid.copy()
        impact_gdf["vulnerability"] = vulnerability
        impact_gdf["expected_impact"] = probability * vulnerability
        
        return impact_gdf

    def plot(self, ax=None, output_dir="data/results/impact_analysis/"):
        """Override plot method to use landslide-specific naming and visualization grid."""
        # Use visualization grid for plotting
        exposure_grid = self.exposure_layer.get_visualization_grid()
        
        # Use the existing vulnerability layer but get visualization grid
        vulnerability_grid = self.vulnerability_layer.get_visualization_grid()
        value_col = getattr(self.vulnerability_layer, "value_column", "school_count")
        vulnerability = vulnerability_grid[value_col].values
        
        # Create visualization impact grid
        impact_gdf = exposure_grid.copy()
        impact_gdf["vulnerability"] = vulnerability
        impact_gdf["expected_impact"] = (
            impact_gdf["probability"] * impact_gdf["vulnerability"]
        )
        
        nicaragua_gdf = get_nicaragua_boundary()
        show_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 10))
            show_fig = True
        
        # Ensure ax is a single Axes object
        if isinstance(ax, (list, tuple)):
            ax = ax[0]
        
        impact_gdf.plot(
            ax=ax,
            column="expected_impact",
            cmap="OrRd",
            linewidth=0.1,
            edgecolor="grey",
            alpha=0.7,
            legend=True,
            legend_kwds={"label": "Expected Impact per Cell"},
        )
        
        if nicaragua_gdf is not None:
            nicaragua_gdf.plot(
                ax=ax, color="none", edgecolor="black", linewidth=3, alpha=1.0
            )
        
        # Get vulnerability name for file naming
        if hasattr(self.vulnerability_layer, "__class__"):
            vuln_name = (
                self.vulnerability_layer.__class__.__name__.replace(
                    "VulnerabilityLayer", ""
                ).lower()
                or "vulnerability"
            )
        else:
            vuln_name = "vulnerability"
        
        ax.set_title(f"Landslide Impact Heatmap - Ensemble Approach (Expected Affected Entities)")
        plt.tight_layout()
        
        # Create landslide-specific filename
        out_path = os.path.join(
            output_dir, f"landslide_impact_{vuln_name}_ensemble.png"
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved landslide impact plot: {out_path}")
        if show_fig:
            plt.close(fig)

    def plot_binary_probability(self, output_dir="data/results/impact_analysis/"):
        """Override binary probability method to use landslide-specific naming and visualization grid."""
        import numpy as np
        
        # Use visualization grid for plotting
        exposure_grid = self.exposure_layer.get_visualization_grid()
        
        # Use the existing vulnerability layer but get visualization grid
        vulnerability_grid = self.vulnerability_layer.get_visualization_grid()
        value_col = getattr(self.vulnerability_layer, "value_column", "school_count")

        # Create binary vulnerability mask (1 if any entities present, 0 otherwise)
        binary_vulnerability = (vulnerability_grid[value_col] > 0).astype(int)
        
        # Create binary probability grid - show landslide probability only where schools exist
        binary_gdf = exposure_grid.copy()
        # Set probability to 0 for cells without schools, keep actual probability for cells with schools
        binary_gdf["binary_probability"] = np.where(
            vulnerability_grid[value_col] > 0, 
            exposure_grid["probability"], 
            0.0
        )
        
        nicaragua_gdf = get_nicaragua_boundary()
        fig, ax = plt.subplots(figsize=(12, 10))
        
        binary_gdf.plot(
            ax=ax,
            column="binary_probability",
            cmap="YlOrRd",
            linewidth=0.1,
            edgecolor="grey",
            alpha=0.7,
            legend=True,
            legend_kwds={"label": "Landslide Probability (cells with schools only)"},
        )
        
        if nicaragua_gdf is not None:
            nicaragua_gdf.plot(
                ax=ax, color="none", edgecolor="black", linewidth=3, alpha=1.0
            )
        
        # Get vulnerability name for file naming
        if hasattr(self.vulnerability_layer, "__class__"):
            vuln_name = (
                self.vulnerability_layer.__class__.__name__.replace(
                    "VulnerabilityLayer", ""
                ).lower()
                or "vulnerability"
            )
        else:
            vuln_name = "vulnerability"
        
        ax.set_title(f"Landslide Probability for {vuln_name.title()} - Ensemble Approach")
        plt.tight_layout()
        
        # Create landslide-specific filename
        out_path = os.path.join(
            output_dir, f"binary_probability_{vuln_name}_ensemble.png"
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved binary probability plot: {out_path}")
        plt.close(fig)
        
        # Print summary statistics
        total_cells_with_vulnerability = binary_vulnerability.sum()
        cells_with_risk = (binary_vulnerability & (exposure_grid["probability"] > 0)).sum()
        # Calculate statistics only for cells with schools
        cells_with_schools = vulnerability_grid[value_col] > 0
        if cells_with_schools.sum() > 0:
            max_probability = exposure_grid.loc[cells_with_schools, "probability"].max()
            mean_probability = exposure_grid.loc[cells_with_schools, "probability"].mean()
        else:
            max_probability = 0.0
            mean_probability = 0.0
        
        print(f"\nLandslide Probability Summary for {vuln_name} (ensemble):")
        print(f"  Total grid cells with {vuln_name}s: {total_cells_with_vulnerability}")
        print(f"  Grid cells with {vuln_name}s at risk (probability > 0): {cells_with_risk}")
        print(f"  Maximum probability in cells with {vuln_name}s: {max_probability:.3f}")
        print(f"  Mean probability in cells with {vuln_name}s: {mean_probability:.3f}")

    def best_case(self):
        """Get best case impact from ensemble."""
        from src.impact_analysis.helper.factories import get_vulnerability_layer
        
        vuln_name = self._get_vulnerability_name()
        high_res_vulnerability = get_vulnerability_layer(
            vuln_name,
            self.config,
            self.vulnerability_layer.cache_dir,
            resolution_context="landslide_computation"
        )
        return self.exposure_layer.get_best_worst_case(high_res_vulnerability)[0]

    def worst_case(self):
        """Get worst case impact from ensemble."""
        from src.impact_analysis.helper.factories import get_vulnerability_layer
        
        vuln_name = self._get_vulnerability_name()
        high_res_vulnerability = get_vulnerability_layer(
            vuln_name,
            self.config,
            self.vulnerability_layer.cache_dir,
            resolution_context="landslide_computation"
        )
        return self.exposure_layer.get_best_worst_case(high_res_vulnerability)[1]

    def expected_impact(self):
        """Get expected impact using mean probability."""
        # Use high-resolution vulnerability layer for accurate computation
        from src.impact_analysis.helper.factories import get_vulnerability_layer
        
        vuln_name = self._get_vulnerability_name()
        high_res_vulnerability = get_vulnerability_layer(
            vuln_name,
            self.config,
            self.vulnerability_layer.cache_dir,
            resolution_context="landslide_computation"
        )
        
        # Create temporary exposure layer with computation context
        temp_exposure = type(self.exposure_layer)(
            self.exposure_layer.landslide_file,
            self.config,
            self.exposure_layer.cache_dir,
            self.exposure_layer.resampling_method,
            resolution_context="landslide_computation"
        )
        
        exposure_grid = temp_exposure.compute_grid()
        vulnerability_grid = high_res_vulnerability.compute_grid()
        
        # Calculate expected impact
        expected_impact = np.sum(
            exposure_grid["probability"].values * 
            vulnerability_grid[high_res_vulnerability.value_column].values
        )
        
        return expected_impact

    def get_ensemble_statistics(self):
        """Get ensemble statistics."""
        # Use high-resolution vulnerability layer for ensemble calculations
        from src.impact_analysis.helper.factories import get_vulnerability_layer
        
        vuln_name = self._get_vulnerability_name()
        high_res_vulnerability = get_vulnerability_layer(
            vuln_name,
            self.config,
            self.vulnerability_layer.cache_dir,
            resolution_context="landslide_computation"
        )
        
        ensemble_impacts = self.exposure_layer.get_ensemble_impact(high_res_vulnerability)
        
        return {
            "mean_impact": np.mean(ensemble_impacts),
            "median_impact": np.median(ensemble_impacts),
            "std_impact": np.std(ensemble_impacts),
            "min_impact": np.min(ensemble_impacts),
            "max_impact": np.max(ensemble_impacts),
            "ensemble_impacts": ensemble_impacts
        }

    def save_impact_summary(self, output_dir="data/results/impact_analysis/"):
        """Save impact summary with ensemble statistics."""
        # Get ensemble statistics
        stats = self.get_ensemble_statistics()
        
        # Get vulnerability name
        vuln_name = self._get_vulnerability_name()
        
        # Create summary text
        summary = f"""Landslide Impact Analysis Summary
{'='*50}

Vulnerability Type: {vuln_name}
Exposure Method: Ensemble (mean probability with spatial correlation)

Ensemble Statistics (50 members):
  Expected Impact (mean probability): {self.expected_impact():.2f}
  Best Case (ensemble minimum): {stats['min_impact']:.2f}
  Worst Case (ensemble maximum): {stats['max_impact']:.2f}
  Ensemble Mean: {stats['mean_impact']:.2f}
  Ensemble Median: {stats['median_impact']:.2f}
  Ensemble Std Dev: {stats['std_impact']:.2f}

Ensemble Distribution:
  {stats['min_impact']:.2f} (min) - {stats['max_impact']:.2f} (max)
  {stats['mean_impact']:.2f} ± {stats['std_impact']:.2f} (mean ± std)

Analysis completed with spatial correlation and ensemble sampling.
"""
        
        # Save to file
        os.makedirs(output_dir, exist_ok=True)
        summary_path = os.path.join(output_dir, f"impact_summary_landslide_{vuln_name}.txt")
        
        with open(summary_path, "w") as f:
            f.write(summary)
        
        print(f"Saved impact summary: {summary_path}")
        return summary_path
