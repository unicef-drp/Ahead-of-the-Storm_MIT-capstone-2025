"""
Combined Risk Heatmap
This module contains logic to create a risk heatmap by multiplying hurricane probability by school count.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from src.impact_analysis.hurricane_grid_heatmap import get_nicaragua_boundary
from src.utils.config_utils import get_config_value


def create_risk_heatmap(
    hurricane_grid_gdf,
    school_grid_gdf,
    output_dir,
    grid_res=0.1,
    forecast_time=None,
    config=None,
):
    print(
        "\n🎯 Generating risk heatmap (hurricane probability x schools) over Nicaragua..."
    )

    # Get configuration values with defaults
    if config is None:
        config = {}

    # Risk analysis parameters
    use_log_scale = get_config_value(
        config, "impact_analysis.risk_analysis.use_log_scale", True
    )
    calculation_method = get_config_value(
        config, "impact_analysis.risk_analysis.calculation_method", "multiplication"
    )

    # Visualization settings
    figure_size = get_config_value(
        config, "impact_analysis.heatmaps.figure_size", [12, 10]
    )
    if isinstance(figure_size, list) and len(figure_size) == 2:
        fig_size = tuple(figure_size)
    else:
        fig_size = (12, 10)
    dpi = get_config_value(config, "impact_analysis.heatmaps.dpi", 300)
    color_map = get_config_value(
        config, "impact_analysis.heatmaps.color_maps.risk", "OrRd"
    )
    alpha = get_config_value(config, "impact_analysis.heatmaps.alpha", 0.7)
    line_width = get_config_value(config, "impact_analysis.heatmaps.line_width", 0.1)
    edge_color = get_config_value(config, "impact_analysis.heatmaps.edge_color", "grey")

    if hurricane_grid_gdf is not None and school_grid_gdf is not None:
        merged_gdf = hurricane_grid_gdf.copy()
        merged_gdf["school_count"] = school_grid_gdf["school_count"]

        # Calculate total number of ensemble members for probability calculation
        total_ensemble_members = merged_gdf["track_count"].max()
        print(f"Total ensemble members: {total_ensemble_members}")

        # Calculate hurricane probability for each grid cell
        merged_gdf["hurricane_probability"] = (
            merged_gdf["track_count"] / total_ensemble_members
        )

        # Calculate expected affected schools for each grid cell
        merged_gdf["expected_affected_schools"] = (
            merged_gdf["hurricane_probability"] * merged_gdf["school_count"]
        )

        # Calculate total expected affected schools
        total_expected_affected = merged_gdf["expected_affected_schools"].sum()
        print(
            f"\n📊 Expected Number of Schools Affected: {total_expected_affected:.2f}"
        )

        # Show breakdown by grid cells with highest expected impact
        high_impact_cells = merged_gdf[
            merged_gdf["expected_affected_schools"] > 0
        ].sort_values("expected_affected_schools", ascending=False)
        if len(high_impact_cells) > 0:
            print(f"Top 5 grid cells by expected affected schools:")
            for i, (idx, row) in enumerate(high_impact_cells.head(5).iterrows()):
                print(
                    f"  {i+1}. Grid cell: {row['expected_affected_schools']:.2f} schools "
                    f"(prob: {row['hurricane_probability']:.3f}, schools: {row['school_count']})"
                )

        # Calculate risk based on method
        if calculation_method == "multiplication":
            merged_gdf["risk"] = merged_gdf["track_count"] * merged_gdf["school_count"]
        elif calculation_method == "weighted_sum":
            weights = get_config_value(
                config, "impact_analysis.risk_analysis.weights", {}
            )
            hurricane_weight = weights.get("hurricane_probability", 0.7)
            school_weight = weights.get("school_concentration", 0.3)
            merged_gdf["risk"] = (
                hurricane_weight * merged_gdf["track_count"]
                + school_weight * merged_gdf["school_count"]
            )
        else:
            merged_gdf["risk"] = merged_gdf["track_count"] * merged_gdf["school_count"]

        risk_counts = merged_gdf["risk"].tolist()
        non_zero_risk = [r for r in risk_counts if r > 0]
        print(f"Grid cells with nonzero risk: {sum(1 for r in risk_counts if r > 0)}")
        if non_zero_risk:
            print(f"Max risk: {max(non_zero_risk)}")
            print(f"Mean risk (nonzero): {sum(non_zero_risk)/len(non_zero_risk):.1f}")
            print(
                f"Median risk (nonzero): {sorted(non_zero_risk)[len(non_zero_risk)//2]}"
            )
            print(
                f"95th percentile: {sorted(non_zero_risk)[int(len(non_zero_risk)*0.95)]}"
            )

        if use_log_scale:
            print("Using log scale for risk heatmap.")
            merged_gdf["log_risk"] = [np.log10(r + 1) for r in risk_counts]
            color_col = "log_risk"
            legend_label = "Log10(Risk + 1) per Cell"
        else:
            print("Using linear scale for risk heatmap.")
            color_col = "risk"
            legend_label = "Risk per Cell"

        nicaragua_gdf = get_nicaragua_boundary()
        minx, miny, maxx, maxy = nicaragua_gdf.total_bounds
        fig, ax = plt.subplots(figsize=fig_size)
        nicaragua_gdf.plot(
            ax=ax, color="none", edgecolor="black", linewidth=3, alpha=1.0
        )
        merged_gdf.plot(
            ax=ax,
            column=color_col,
            cmap=color_map,
            linewidth=line_width,
            edgecolor=edge_color,
            alpha=alpha,
            legend=True,
            legend_kwds={"label": legend_label},
        )
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_xlabel("Longitude (°E)", fontsize=12)
        ax.set_ylabel("Latitude (°N)", fontsize=12)
        ax.set_title(
            f"Risk Heatmap (Hurricane x Schools)\nGrid Resolution: {grid_res}°\nExpected Affected Schools: {total_expected_affected:.1f}",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if forecast_time is not None:
            date_str = forecast_time.strftime("%Y%m%d_%H%M")
        else:
            date_str = "unknown"
        risk_heatmap_path = os.path.join(
            output_dir,
            f"risk_heatmap_{date_str}.png",
        )
        print(f"[DEBUG] Saving risk heatmap to: {risk_heatmap_path}")
        plt.savefig(risk_heatmap_path, dpi=dpi, bbox_inches="tight")
        print(f"\n✅ Risk heatmap saved:\n   {risk_heatmap_path}")
        plt.close()

        # Return both the merged dataframe and the expected affected schools count
        return merged_gdf, total_expected_affected
    else:
        print("❌ Cannot create risk heatmap: missing hurricane or school data")
        return None, 0
