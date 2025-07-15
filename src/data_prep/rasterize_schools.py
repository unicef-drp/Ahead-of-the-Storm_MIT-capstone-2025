"""
Script to rasterize SINAPRED schools point data to a GeoTIFF grid using config paths.
"""
import os
from src.utils.rasterize import rasterize_points_to_tiff
from src.utils.path_utils import get_project_root


def main():
    try:
        # Use config_key to read input/output paths from config
        rasterize_points_to_tiff(config_key='school_rasterization')
    except Exception as e:
        print(f"Error during rasterization: {e}")


if __name__ == "__main__":
    main() 