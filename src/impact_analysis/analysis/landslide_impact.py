from src.impact_analysis.analysis.hurricane_impact import HurricaneImpactLayer
import os
import matplotlib.pyplot as plt
from src.utils.hurricane_geom import get_nicaragua_boundary


class LandslideImpactLayer(HurricaneImpactLayer):
    """Landslide-specific impact layer that fixes file naming and functionality."""
    
    def _get_vulnerability_name(self):
        """Helper method to get vulnerability name for factory."""
        if hasattr(self.vulnerability_layer, "__class__"):
            class_name = self.vulnerability_layer.__class__.__name__
            # Map class names to factory names
            vuln_name_map = {
                "SchoolVulnerabilityLayer": "schools",
                "SchoolPopulationVulnerabilityLayer": "school_population",
                "HealthFacilityVulnerabilityLayer": "health_facilities",
                "PopulationVulnerabilityLayer": "population",
                "PovertyVulnerabilityLayer": "poverty",
                "SeverePovertyVulnerabilityLayer": "severe_poverty",
                "ShelterVulnerabilityLayer": "shelters",
            }
            return vuln_name_map.get(class_name, "schools")  # Default to schools
        else:
            return "schools"  # Default for schools
    
    def compute_impact(self):
        """Override to use high-resolution computation."""
        if self.impact_gdf is not None:
            return self.impact_gdf
        
        # Use high-resolution grids for computation
        exposure_grid = self.exposure_layer.get_computation_grid()
        vulnerability_grid = self.vulnerability_layer.compute_grid()
        
        # Use the correct value column for this vulnerability layer
        value_col = getattr(self.vulnerability_layer, "value_column", "school_count")
        vulnerability = vulnerability_grid[value_col].values
        
        # Compute impact at high resolution
        impact_gdf = exposure_grid.copy()
        impact_gdf["vulnerability"] = vulnerability
        impact_gdf["expected_impact"] = (
            impact_gdf["probability"] * impact_gdf["vulnerability"]
        )
        
        # Store high-res result for accurate calculations
        self.impact_gdf = impact_gdf
        return impact_gdf
    
    def plot(self, ax=None, output_dir="data/results/impact_analysis/"):
        """Override plot method to use landslide-specific naming and visualization grid."""
        # Use visualization grid for plotting
        exposure_grid = self.exposure_layer.get_visualization_grid()
        
        # Create a temporary vulnerability layer with visualization resolution
        from src.impact_analysis.helper.factories import get_vulnerability_layer
        
        vuln_name = self._get_vulnerability_name()
        
        viz_vulnerability = get_vulnerability_layer(
            vuln_name,
            self.config,
            self.vulnerability_layer.cache_dir,
            resolution_context="landslide_visualization"
        )
        vulnerability_grid = viz_vulnerability.compute_grid()
        
        # Use the correct value column for this vulnerability layer
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
        
        # Get scenario from exposure layer
        scenario = getattr(self.exposure_layer, "resampling_method", "unknown")
        
        ax.set_title(f"Landslide Impact Heatmap - {scenario.title()} Scenario (Expected Affected Entities)")
        plt.tight_layout()
        
        # Create landslide-specific filename
        out_path = os.path.join(
            output_dir, f"landslide_impact_{vuln_name}_{scenario}.png"
        )
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved landslide impact plot: {out_path}")
        if show_fig:
            plt.close(fig)
    
    def plot_binary_probability(self, output_dir="data/results/impact_analysis/"):
        """Override binary probability method to use landslide-specific naming and visualization grid."""
        import numpy as np
        
        # Use visualization grid for plotting
        exposure_grid = self.exposure_layer.get_visualization_grid()
        
        # Create a temporary vulnerability layer with visualization resolution
        from src.impact_analysis.helper.factories import get_vulnerability_layer
        
        vuln_name = self._get_vulnerability_name()
        
        viz_vulnerability = get_vulnerability_layer(
            vuln_name,
            self.config,
            self.vulnerability_layer.cache_dir,
            resolution_context="landslide_visualization"
        )
        vulnerability_grid = viz_vulnerability.compute_grid()
        value_col = getattr(self.vulnerability_layer, "value_column", "school_count")

        # Create binary vulnerability mask (1 if any entities present, 0 otherwise)
        binary_vulnerability = (vulnerability_grid[value_col] > 0).astype(int)

        # Combine exposure probability with binary vulnerability
        binary_probability = exposure_grid["probability"] * binary_vulnerability

        # Create plot
        nicaragua_gdf = get_nicaragua_boundary()

        if hasattr(self.vulnerability_layer, "__class__"):
            vuln_name = (
                self.vulnerability_layer.__class__.__name__.replace(
                    "VulnerabilityLayer", ""
                ).lower()
                or "vulnerability"
            )
        else:
            vuln_name = "vulnerability"

        # Get scenario from exposure layer
        scenario = getattr(self.exposure_layer, "resampling_method", "unknown")

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
            f"Landslide Binary Probability - {scenario.title()} Scenario\n(At Least One {vuln_name.title()} Affected per Grid Cell)"
        )
        plt.tight_layout()

        out_path = os.path.join(
            output_dir, f"binary_probability_{vuln_name}_{scenario}.png"
        )
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved binary probability plot: {out_path}")
        plt.close(fig)

        # Print summary statistics
        total_cells_with_vulnerability = binary_vulnerability.sum()
        cells_with_risk = (binary_probability > 0).sum()
        max_probability = binary_probability.max()
        mean_probability = binary_probability.mean()

        print(f"\nBinary Probability Summary for {vuln_name} ({scenario}):")
        print(f"  Total grid cells with {vuln_name}s: {total_cells_with_vulnerability}")
        print(f"  Grid cells at risk (probability > 0): {cells_with_risk}")
        print(f"  Maximum probability in any cell: {max_probability:.3f}")
        print(f"  Mean probability across all cells: {mean_probability:.3f}")
    
    def best_case(self):
        """Override to return the expected impact for landslide (no ensemble)."""
        return self.expected_impact()
    
    def worst_case(self):
        """Override to return the expected impact for landslide (no ensemble)."""
        return self.expected_impact()
    
    def _per_member_impacts(self):
        """Override to return single value for landslide (no ensemble)."""
        return [self.expected_impact()]
