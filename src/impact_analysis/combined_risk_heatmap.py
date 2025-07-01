"""
Combined Risk Heatmap
This module contains logic to create a risk heatmap by multiplying hurricane probability by school count.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from src.impact_analysis.hurricane_grid_heatmap import get_nicaragua_boundary


def create_risk_heatmap(
    hurricane_grid_gdf, school_grid_gdf, output_dir, grid_res=0.1, forecast_time=None
):
    print(
        "\n🎯 Generating risk heatmap (hurricane probability x schools) over Nicaragua..."
    )
    if hurricane_grid_gdf is not None and school_grid_gdf is not None:
        merged_gdf = hurricane_grid_gdf.copy()
        merged_gdf["school_count"] = school_grid_gdf["school_count"]
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
        use_log = True
        if use_log:
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
        fig, ax = plt.subplots(figsize=(12, 10))
        nicaragua_gdf.plot(
            ax=ax, color="none", edgecolor="black", linewidth=3, alpha=1.0
        )
        merged_gdf.plot(
            ax=ax,
            column=color_col,
            cmap="OrRd",
            linewidth=0.1,
            edgecolor="grey",
            alpha=0.7,
            legend=True,
            legend_kwds={"label": legend_label},
        )
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_xlabel("Longitude (°E)", fontsize=12)
        ax.set_ylabel("Latitude (°N)", fontsize=12)
        ax.set_title(
            f"Risk Heatmap (Hurricane x Schools)\nGrid Resolution: {grid_res}°",
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
        plt.savefig(risk_heatmap_path, dpi=300, bbox_inches="tight")
        print(f"\n✅ Risk heatmap saved:\n   {risk_heatmap_path}")
        plt.close()
        return merged_gdf
    else:
        print("❌ Cannot create risk heatmap: missing hurricane or school data")
        return None
