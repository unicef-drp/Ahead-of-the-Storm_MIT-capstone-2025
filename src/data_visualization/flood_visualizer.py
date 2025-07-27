import os
import matplotlib.pyplot as plt
from src.impact_analysis.layers.flood import FloodExposureLayer
from src.utils.config_utils import load_config


def main():
    config = load_config("config/flood_config.yaml")
    flood_cfg = config["flood"]
    # Use the h25 risk map as default for visualization
    riskmap_dir = flood_cfg["riskmap_dir"]
    riskmap_files = flood_cfg["riskmap_files"]
    flood_raster_path = os.path.join(riskmap_dir, riskmap_files["h25"])
    output_dir = flood_cfg.get("output_directory", "data/results/impact_analysis/")
    layer = FloodExposureLayer(flood_raster_path, config)
    grid_gdf = layer.compute_grid()
    layer.plot(output_dir=output_dir)
    print("Flood exposure visualization complete.")


if __name__ == "__main__":
    main()
