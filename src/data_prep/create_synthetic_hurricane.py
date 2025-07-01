#!/usr/bin/env python3
"""
Create Synthetic Hurricane Script.

This script takes a hurricane file from weatherlab/processed and applies
latitude and longitude offsets to create a synthetic hurricane track.
The offsets are specified in the config file.
"""

import sys
import pandas as pd
from datetime import datetime

from src.utils.config_utils import load_config, get_config_value
from src.utils.logging_utils import setup_logging, get_logger
from src.utils.path_utils import ensure_directory, get_data_path


def create_synthetic_hurricane(config: dict) -> str:
    """
    Create a synthetic hurricane by applying latitude/longitude offsets.

    Args:
        config: Configuration dictionary

    Returns:
        Path to the created synthetic hurricane file
    """
    logger = get_logger(__name__)

    # Extract configuration parameters
    delta_lat = get_config_value(config, "synthetic_hurricane.delta_latitude")
    delta_lon = get_config_value(config, "synthetic_hurricane.delta_longitude")

    input_dir = get_config_value(config, "synthetic_hurricane.input.data_dir")
    source_file = get_config_value(config, "synthetic_hurricane.input.source_file")
    output_dir = get_config_value(config, "synthetic_hurricane.output.data_dir")
    file_suffix = get_config_value(config, "synthetic_hurricane.output.file_suffix")

    preserve_track_id = get_config_value(
        config, "synthetic_hurricane.processing.preserve_original_track_id"
    )
    new_track_id = get_config_value(
        config, "synthetic_hurricane.processing.new_track_id"
    )

    # Create output directory
    output_path = get_data_path(output_dir)
    ensure_directory(output_path)

    # Load source hurricane data
    source_path = get_data_path(input_dir) / source_file
    logger.info(f"Loading source hurricane data from: {source_path}")

    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    # Read the CSV file
    df = pd.read_csv(source_path)
    logger.info(f"Loaded {len(df):,} records from source file")

    # Display original track statistics
    logger.info("Original track statistics:")
    logger.info(
        f"  Latitude range: {df['latitude'].min():.2f} to {df['latitude'].max():.2f}"
    )
    logger.info(
        f"  Longitude range: {df['longitude'].min():.2f} to {df['longitude'].max():.2f}"
    )
    logger.info(f"  Track ID: {df['track_id'].iloc[0]}")
    logger.info(f"  Ensemble members: {df['ensemble_member'].nunique()}")

    # Apply latitude and longitude offsets
    logger.info(f"Applying offsets: lat +{delta_lat:.2f}°, lon +{delta_lon:.2f}°")

    df_synthetic = df.copy()
    df_synthetic["latitude"] = df_synthetic["latitude"] + delta_lat
    df_synthetic["longitude"] = df_synthetic["longitude"] + delta_lon

    # Update track ID if requested
    if not preserve_track_id:
        df_synthetic["track_id"] = new_track_id
        logger.info(f"Updated track ID to: {new_track_id}")

    # Display new track statistics
    logger.info("New track statistics:")
    logger.info(
        f"  Latitude range: {df_synthetic['latitude'].min():.2f} to "
        f"{df_synthetic['latitude'].max():.2f}"
    )
    logger.info(
        f"  Longitude range: {df_synthetic['longitude'].min():.2f} to "
        f"{df_synthetic['longitude'].max():.2f}"
    )
    logger.info(f"  Track ID: {df_synthetic['track_id'].iloc[0]}")

    # Generate output filename
    source_name = source_file.replace(".csv", "")
    output_filename = f"{source_name}{file_suffix}.csv"
    output_file_path = output_path / output_filename

    # Save synthetic hurricane data
    logger.info(f"Saving synthetic hurricane data to: {output_file_path}")
    df_synthetic.to_csv(output_file_path, index=False)

    logger.info(
        f"Successfully created synthetic hurricane with {len(df_synthetic):,} records"
    )

    return str(output_file_path)


def main():
    """Main function to create synthetic hurricane."""
    # Setup logging
    logger = setup_logging(__name__)

    print("=" * 60)
    print("SYNTHETIC HURRICANE GENERATOR")
    print("=" * 60)

    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = load_config("config/synthetic_hurricane_config.yaml")

        # Create synthetic hurricane
        output_path = create_synthetic_hurricane(config)

        print(f"\n✅ SUCCESS!")
        print(f"📁 Synthetic hurricane saved to: {output_path}")
        print(f"🔧 Applied offsets from config file")
        print(f"📊 Check the log file for detailed statistics")
        print("=" * 60)

    except Exception as e:
        logger.error(f"Error creating synthetic hurricane: {e}")
        print(f"\n❌ ERROR: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
