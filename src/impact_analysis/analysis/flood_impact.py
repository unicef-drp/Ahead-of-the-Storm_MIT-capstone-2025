import numpy as np
from src.impact_analysis.analysis.hurricane_impact import HurricaneImpactLayer
from typing import Dict, Any


class FloodImpactLayer(HurricaneImpactLayer):
    def get_plot_metadata(self) -> Dict[str, Any]:
        """Return metadata for plotting this flood impact layer."""
        vuln_name = self._get_vulnerability_name()
        return {
            "layer_type": "impact",
            "hazard_type": "Flood",
            "vulnerability_type": vuln_name,
            "data_column": "expected_impact",
            "colormap": "Reds",
            "title_template": "Number of {vulnerability_type} to be Impacted by Forecasted {hazard_type}",
            "legend_template": "Expected Impact per Cell",
            "filename_template": "{hazard_type}_impact_{vulnerability_type}_{parameters}",
            "special_features": []
        }

    def expected_impact(self):
        """Use ensemble-based calculation for flood impact."""
        # Use ensemble-based calculation (mean of individual member impacts)
        impacts = self._per_member_impacts()
        return np.mean(impacts) if impacts else 0.0

    def save_impact_summary(self, output_dir="data/results/impact_analysis/"):
        """Override to use flood-specific naming."""
        import os
        import numpy as np

        forecast_time = getattr(self.exposure_layer, "chosen_forecast", None)
        if hasattr(self.vulnerability_layer, "__class__"):
            vuln_name = (
                self.vulnerability_layer.__class__.__name__.replace(
                    "VulnerabilityLayer", ""
                ).lower()
                or "vulnerability"
            )
        else:
            vuln_name = "vulnerability"
        
        # Handle forecast time - use empty string if no forecast time available
        if forecast_time is None:
            date_str = ""
        else:
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
            f"Impact summary for flood_{vuln_name}{vuln_params}{'_' + date_str if date_str else ''}\n"
            f"Forecast time: {forecast_time if forecast_time else 'N/A'}\n"
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
            f"impact_summary_flood_{vuln_name}{vuln_params}{'_' + date_str if date_str else ''}.txt",
        )
        with open(summary_path, "w") as f:
            f.write(summary)
        print(f"Saved impact summary: {summary_path}")

    def plot_binary_probability(self, output_dir="data/results/impact_analysis/"):
        """
        Plot binary probability of at least one entity being affected in each grid cell.
        Override to handle flood-specific naming without forecast time.
        """
        import matplotlib.pyplot as plt
        import os
        from src.utils.hurricane_geom import get_nicaragua_boundary

        exposure_grid = self.exposure_layer.compute_grid()
        vulnerability_grid = self.vulnerability_layer.compute_grid()
        value_col = getattr(self.vulnerability_layer, "value_column", "school_count")

        # Create binary vulnerability mask (1 if any entities present, 0 otherwise)
        binary_vulnerability = (vulnerability_grid[value_col] > 0).astype(int)

        # Combine exposure probability with binary vulnerability
        binary_probability = exposure_grid["probability"] * binary_vulnerability

        # Create plot
        nicaragua_gdf = get_nicaragua_boundary()
        forecast_time = getattr(self.exposure_layer, "chosen_forecast", None)

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

        # Handle forecast time - use empty string if no forecast time available
        if forecast_time is None:
            date_str = ""
        else:
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
            output_dir, f"binary_probability_{vuln_name}{vuln_params}{'_' + date_str if date_str else ''}.png"
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
