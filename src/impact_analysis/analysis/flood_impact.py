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
            f"Impact summary for flood_{vuln_name}{vuln_params}_{date_str}\n"
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
            f"impact_summary_flood_{vuln_name}{vuln_params}_{date_str}.txt",
        )
        with open(summary_path, "w") as f:
            f.write(summary)
        print(f"Saved impact summary: {summary_path}")
