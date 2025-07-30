import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from typing import Dict, Any
from src.impact_analysis.layers.base import ImpactLayer
from src.utils.hurricane_geom import get_nicaragua_boundary
import os


class HurricaneImpactLayer(ImpactLayer):
    def __init__(self, exposure_layer, vulnerability_layer, config):
        super().__init__(exposure_layer, vulnerability_layer, config)
        self.impact_gdf = None

    def _get_vulnerability_name(self):
        """Get vulnerability name for factory function."""
        class_name = self.vulnerability_layer.__class__.__name__
        # Map class names to factory names
        name_mapping = {
            "SchoolVulnerabilityLayer": "Schools",
            "SchoolPopulationVulnerabilityLayer": "School Population",
            "PopulationVulnerabilityLayer": "Population",
            "PovertyVulnerabilityLayer": "People in Poverty",
            "SeverePovertyVulnerabilityLayer": "People in Severe Poverty",
            "ShelterVulnerabilityLayer": "Shelters",
            "HealthFacilityVulnerabilityLayer": "Health Facilities"
        }
        
        # Special handling for HealthFacilityVulnerabilityLayer
        if class_name == "HealthFacilityVulnerabilityLayer":
            if hasattr(self.vulnerability_layer, 'weighted_by_population') and self.vulnerability_layer.weighted_by_population:
                return "Health Facility Population"
            else:
                return "Health Facilities"
        
        # Special handling for ShelterVulnerabilityLayer
        if class_name == "ShelterVulnerabilityLayer":
            if hasattr(self.vulnerability_layer, 'weighted_by_capacity') and self.vulnerability_layer.weighted_by_capacity:
                return "Shelter Population"
            else:
                return "Shelters"
        
        # Special handling for PopulationVulnerabilityLayer
        if class_name == "PopulationVulnerabilityLayer":
            if hasattr(self.vulnerability_layer, 'age_groups'):
                # Check if it's children (ages 0, 5, 10, 15) or total population
                if self.vulnerability_layer.age_groups == [0, 5, 10, 15]:
                    return "Children"
                else:
                    return "Population"
        
        # Special handling for PovertyVulnerabilityLayer
        if class_name == "PovertyVulnerabilityLayer":
            if hasattr(self.vulnerability_layer, 'age_groups'):
                # Check if it's children (ages 0, 5, 10, 15) or total population
                if self.vulnerability_layer.age_groups == [0, 5, 10, 15]:
                    return "Children in Poverty"
                else:
                    return "People in Poverty"
        
        # Special handling for SeverePovertyVulnerabilityLayer
        if class_name == "SeverePovertyVulnerabilityLayer":
            if hasattr(self.vulnerability_layer, 'age_groups'):
                # Check if it's children (ages 0, 5, 10, 15) or total population
                if self.vulnerability_layer.age_groups == [0, 5, 10, 15]:
                    return "Children in Severe Poverty"
                else:
                    return "People in Severe Poverty"
        
        return name_mapping.get(class_name, "Schools")

    def compute_impact(self):
        if self.impact_gdf is not None:
            return self.impact_gdf
        exposure_grid = self.exposure_layer.compute_grid()
        vulnerability_grid = self.vulnerability_layer.compute_grid()
        # Use the correct value column for this vulnerability layer
        value_col = getattr(self.vulnerability_layer, "value_column", "school_count")
        vulnerability = vulnerability_grid[value_col].values
        impact_gdf = exposure_grid.copy()
        impact_gdf["vulnerability"] = vulnerability
        impact_gdf["expected_impact"] = (
            impact_gdf["probability"] * impact_gdf["vulnerability"]
        )
        self.impact_gdf = impact_gdf
        return impact_gdf

    def get_plot_metadata(self) -> Dict[str, Any]:
        """Return metadata for plotting this hurricane impact layer."""
        vuln_name = self._get_vulnerability_name()
        return {
            "layer_type": "impact",
            "hazard_type": "Hurricane",
            "vulnerability_type": vuln_name,
            "data_column": "expected_impact",
            "colormap": "Reds",
            "title_template": "Number of {vulnerability_type} to be Impacted by Forecasted {hazard_type}",
            "legend_template": "Expected Impact per Cell",
            "filename_template": "{hazard_type}_impact_{vulnerability_type}_{parameters}",
            "special_features": []
        }

    def plot(self, ax=None, output_dir="data/results/impact_analysis/"):
        """Plot the hurricane impact layer using universal plotting function."""
        from src.impact_analysis.utils.plotting_utils import plot_layer_with_scales
        plot_layer_with_scales(self, output_dir=output_dir)

    def expected_impact(self):
        # Use ensemble-based calculation (mean of individual member impacts)
        impacts = self._per_member_impacts()
        return np.mean(impacts) if impacts else 0.0

    def _per_member_impacts(self):
        grid_cells = self.exposure_layer.get_grid_cells()
        member_regions = self.exposure_layer.get_member_regions()
        vulnerability_grid = self.vulnerability_layer.compute_grid()
        value_col = getattr(self.vulnerability_layer, "value_column", "school_count")
        vulnerability = vulnerability_grid[value_col].values
        impacts = []
        for i, region in enumerate(member_regions):
            hit = np.array(
                [
                    int(
                        region is not None
                        and region.is_valid
                        and not region.is_empty
                        and region.intersects(cell)
                    )
                    for cell in grid_cells.geometry
                ]
            )
            impact = (hit * vulnerability).sum()
            impacts.append(impact)
        print("All per-track impacts (first 10):", impacts[:10])
        print("Type of vulnerability array:", vulnerability.dtype)
        print("Type of hit array:", hit.dtype)
        return impacts

    def best_case(self):
        impacts = self._per_member_impacts()
        return min(impacts) if impacts else 0.0

    def worst_case(self):
        impacts = self._per_member_impacts()
        return max(impacts) if impacts else 0.0

    def plot_best_worst_case_overlay(self, output_dir="data/results/impact_analysis/"):
        import matplotlib.pyplot as plt
        import os

        grid_cells = self.exposure_layer.get_grid_cells()
        member_regions = self.exposure_layer.get_member_regions()
        vulnerability_grid = self.vulnerability_layer.compute_grid()
        value_col = getattr(self.vulnerability_layer, "value_column", "school_count")
        vulnerability = vulnerability_grid[value_col].values
        impacts = []
        for region in member_regions:
            hit = np.array(
                [
                    int(
                        region is not None
                        and region.is_valid
                        and not region.is_empty
                        and region.intersects(cell)
                    )
                    for cell in grid_cells.geometry
                ]
            )
            impact = (hit * vulnerability).sum()
            impacts.append(impact)
        if not impacts:
            print("No member regions to plot.")
            return
        best_idx = int(np.argmin(impacts))
        worst_idx = int(np.argmax(impacts))
        nicaragua_gdf = get_nicaragua_boundary()
        forecast_time = getattr(self.exposure_layer, "chosen_forecast", "unknown")
        if hasattr(self.vulnerability_layer, "__class__"):
            vuln_name = (
                self.vulnerability_layer.__class__.__name__.replace(
                    "VulnerabilityLayer", ""
                ).lower()
                or "vulnerability"
            )
        else:
            vuln_name = "vulnerability"
        date_str = str(forecast_time).replace(":", "-").replace(" ", "_")
        for case, idx in zip(["best", "worst"], [best_idx, worst_idx]):
            region = member_regions[idx]
            impact_val = impacts[idx]
            if np.isnan(impact_val):
                affected_str = "N/A"
                print(f"{case.title()} case affected {vuln_name}s: N/A (NaN impact)")
            else:
                affected = int(round(impact_val))
                affected_str = str(affected)
                print(f"{case.title()} case affected {vuln_name}s: {affected}")
            fig, ax = plt.subplots(figsize=(12, 10))
            # Plot vulnerability grid (not clipped)
            if value_col in vulnerability_grid.columns:
                log_counts = [
                    np.log10(count + 1) for count in vulnerability_grid[value_col]
                ]
                log_col = f"log_{value_col}"
                vulnerability_grid[log_col] = log_counts
                cmap = "Blues" if value_col == "school_count" else "Oranges"
                label = f"Log10({'Schools' if value_col == 'school_count' else 'People'} + 1) per Cell"
                vulnerability_grid.plot(
                    ax=ax,
                    column=log_col,
                    cmap=cmap,
                    linewidth=0.1,
                    edgecolor="grey",
                    alpha=0.7,
                    legend=True,
                    legend_kwds={"label": label},
                )
            if region is not None and region.is_valid and not region.is_empty:
                gpd.GeoSeries([region], crs="EPSG:4326").plot(
                    ax=ax,
                    color="red" if case == "worst" else "green",
                    alpha=0.3,
                    edgecolor="black",
                    linewidth=2,
                    label=f"{case.title()} Case Wind Region",
                )
            if nicaragua_gdf is not None:
                nicaragua_gdf.plot(
                    ax=ax, color="none", edgecolor="black", linewidth=3, alpha=1.0
                )
                minx, miny, maxx, maxy = nicaragua_gdf.total_bounds
                ax.set_xlim(minx, maxx)
                ax.set_ylim(miny, maxy)

            ax.set_title(
                f"{case.title()} Case Hurricane Wind Region Overlay with Vulnerability (Boxed)"
            )
            plt.tight_layout()
            # Get vulnerability parameters for more detailed naming
            vuln_params = ""
            if hasattr(self.vulnerability_layer, "age_groups") and hasattr(
                self.vulnerability_layer, "gender"
            ):
                age_str = "_".join(map(str, self.vulnerability_layer.age_groups))
                vuln_params = f"_{self.vulnerability_layer.gender}_ages_{age_str}"

            out_path = os.path.join(
                output_dir,
                f"{case}_case_hurricane_{vuln_name}{vuln_params}_{date_str}_overlay.png",
            )
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            print(f"Saved {case} case hurricane overlay plot: {out_path}")
            plt.close(fig)

    def plot_binary_probability(self, output_dir="data/results/impact_analysis/"):
        """
        Plot binary probability of at least one entity being affected in each grid cell.
        This shows the probability that at least one school/person/etc. is affected,
        rather than the expected number affected.
        """
        import matplotlib.pyplot as plt
        import os

        exposure_grid = self.exposure_layer.compute_grid()
        vulnerability_grid = self.vulnerability_layer.compute_grid()
        value_col = getattr(self.vulnerability_layer, "value_column", "school_count")

        # Create binary vulnerability mask (1 if any entities present, 0 otherwise)
        binary_vulnerability = (vulnerability_grid[value_col] > 0).astype(int)

        # Combine exposure probability with binary vulnerability
        binary_probability = exposure_grid["probability"] * binary_vulnerability

        # Create plot
        nicaragua_gdf = get_nicaragua_boundary()
        forecast_time = getattr(self.exposure_layer, "chosen_forecast", "unknown")

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

        date_str = str(forecast_time).replace(":", "-").replace(" ", "_")

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
            output_dir, f"binary_probability_{vuln_name}{vuln_params}_{date_str}.png"
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

    def save_impact_summary(self, output_dir="data/results/impact_analysis/"):
        import os

        forecast_time = getattr(self.exposure_layer, "chosen_forecast", "unknown")
        if hasattr(self.vulnerability_layer, "__class__"):
            vuln_name = (
                self.vulnerability_layer.__class__.__name__.replace(
                    "VulnerabilityLayer", ""
                ).lower()
                or "vulnerability"
            )
        else:
            vuln_name = "vulnerability"
        date_str = str(forecast_time).replace(":", "-").replace(" ", "_")
        # Compute all metrics
        expected = self.expected_impact()
        impacts = self._per_member_impacts()
        if not impacts:
            print("No member regions to summarize.")
            return
        best_idx = int(np.argmin(impacts))
        worst_idx = int(np.argmax(impacts))
        best_val = impacts[best_idx]
        worst_val = impacts[worst_idx]
        member_ids = (
            self.exposure_layer.get_member_ids()
            if hasattr(self.exposure_layer, "get_member_ids")
            else None
        )
        best_id = member_ids[best_idx] if member_ids else best_idx
        worst_id = member_ids[worst_idx] if member_ids else worst_idx
        # Get vulnerability parameters for more detailed naming
        vuln_params = ""
        if hasattr(self.vulnerability_layer, "age_groups") and hasattr(
            self.vulnerability_layer, "gender"
        ):
            age_str = "_".join(map(str, self.vulnerability_layer.age_groups))
            vuln_params = f"_{self.vulnerability_layer.gender}_ages_{age_str}"

        summary = (
            f"Impact summary for hurricane_{vuln_name}{vuln_params}_{date_str}\n"
            f"Forecast time: {forecast_time}\n"
            f"Vulnerability type: {vuln_name}\n"
        )

        # Add vulnerability-specific parameters
        if hasattr(self.vulnerability_layer, "age_groups"):
            summary += f"Age groups: {self.vulnerability_layer.age_groups}\n"
        if hasattr(self.vulnerability_layer, "gender"):
            summary += f"Gender: {self.vulnerability_layer.gender}\n"

        summary += (
            f"\nResults:\n"
            f"Expected affected: {expected:.2f}\n"
            f"Best case (min): {best_val:.2f} (ensemble member {best_id})\n"
            f"Worst case (max): {worst_val:.2f} (ensemble member {worst_id})\n"
        )
        print(summary)
        summary_path = os.path.join(
            output_dir,
            f"impact_summary_hurricane_{vuln_name}{vuln_params}_{date_str}.txt",
        )
        with open(summary_path, "w") as f:
            f.write(summary)
        print(f"Saved impact summary: {summary_path}")
