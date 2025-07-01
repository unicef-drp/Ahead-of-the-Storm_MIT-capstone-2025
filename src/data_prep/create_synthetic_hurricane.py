#!/usr/bin/env python3
"""
Create Synthetic Hurricane Script

This script takes a hurricane file from weatherlab/processed and applies
latitude and longitude offsets to create a synthetic hurricane track.
The offsets are specified in the config file.
"""

import os
import logging
import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add src to path for imports
# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def load_config(config_path: str = "config/synthetic_hurricane_config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("synthetic_hurricane.log"),
        ],
    )


def create_synthetic_hurricane(config: dict) -> str:
    """
    Create a synthetic hurricane by applying latitude/longitude offsets.

    Args:
        config: Configuration dictionary

    Returns:
        Path to the created synthetic hurricane file
    """
    logger = logging.getLogger(__name__)

    # Extract configuration parameters
    delta_lat = config["synthetic_hurricane"]["delta_latitude"]
    delta_lon = config["synthetic_hurricane"]["delta_longitude"]

    input_dir = Path(config["synthetic_hurricane"]["input"]["data_dir"])
    source_file = config["synthetic_hurricane"]["input"]["source_file"]
    output_dir = Path(config["synthetic_hurricane"]["output"]["data_dir"])
    file_suffix = config["synthetic_hurricane"]["output"]["file_suffix"]

    preserve_track_id = config["synthetic_hurricane"]["processing"][
        "preserve_original_track_id"
    ]
    new_track_id = config["synthetic_hurricane"]["processing"]["new_track_id"]

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load source hurricane data
    source_path = input_dir / source_file
    logger.info(f"Loading source hurricane data from: {source_path}")

    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    # Read the CSV file
    df = pd.read_csv(source_path)
    logger.info(f"Loaded {len(df):,} records from source file")

    # Display original track statistics
    logger.info(f"Original track statistics:")
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
    logger.info(f"New track statistics:")
    logger.info(
        f"  Latitude range: {df_synthetic['latitude'].min():.2f} to {df_synthetic['latitude'].max():.2f}"
    )
    logger.info(
        f"  Longitude range: {df_synthetic['longitude'].min():.2f} to {df_synthetic['longitude'].max():.2f}"
    )
    logger.info(f"  Track ID: {df_synthetic['track_id'].iloc[0]}")

    # Generate output filename
    source_name = source_file.replace(".csv", "")
    output_filename = f"{source_name}{file_suffix}.csv"
    output_path = output_dir / output_filename

    # Save synthetic hurricane data
    logger.info(f"Saving synthetic hurricane data to: {output_path}")
    df_synthetic.to_csv(output_path, index=False)

    logger.info(
        f"Successfully created synthetic hurricane with {len(df_synthetic):,} records"
    )

    return str(output_path)


def main():
    """Main function to create synthetic hurricane."""
    setup_logging()
    logger = logging.getLogger(__name__)

    print("=" * 60)
    print("SYNTHETIC HURRICANE GENERATOR")
    print("=" * 60)

    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = load_config()

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
