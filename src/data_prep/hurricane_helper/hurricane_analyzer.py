"""
Hurricane Data Analyzer for FNV3 Ensemble Data

This module provides functionality to analyze and prepare hurricane forecast data
from Google Weather Lab's FNV3 model for visualization and track plotting.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime, timedelta
import yaml


class HurricaneAnalyzer:
    """
    Analyzes hurricane forecast data from FNV3 ensemble model.

    This class provides methods to process, analyze, and prepare hurricane data
    for visualization, including track plotting and ensemble analysis.
    """

    def __init__(self, config_path: str = "config/hurricane_config.yaml"):
        """
        Initialize the hurricane analyzer.

        Args:
            config_path: Path to the configuration file
        """
        # Get project root (3 levels up from this file: hurricane_helper/data_prep/src/)
        current_file = os.path.abspath(__file__)
        self.project_root = Path(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
            )
        )
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._setup_directories()

        # Get processed data directory from config
        processed_dir = self.config.get("processing", {}).get(
            "processed_output_directory", "data/preprocessed/weatherlab/processed"
        )
        self.data_dir = self.project_root / processed_dir

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        config_file = self.project_root / config_path
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        return config

    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = self.project_root / "logs"
        log_dir.mkdir(exist_ok=True)

        log_file = (
            log_dir
            / f"hurricane_analyzer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info("Hurricane Analyzer initialized")

    def _setup_directories(self):
        """Create necessary directories if they don't exist."""
        # Create analysis directory
        analysis_dir = self.project_root / "data" / "weatherlab" / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Analysis directory: {analysis_dir}")

    def load_all_data(self) -> pd.DataFrame:
        """
        Load all processed hurricane data files into a single DataFrame.

        Returns:
            Combined DataFrame with all hurricane data
        """
        csv_files = list(self.data_dir.glob("processed_*.csv"))

        if not csv_files:
            raise FileNotFoundError(f"No processed CSV files found in {self.data_dir}")

        self.logger.info(f"Loading {len(csv_files)} processed data files...")

        dataframes = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                # Add source file information
                df["source_file"] = csv_file.name
                dataframes.append(df)
                self.logger.info(f"Loaded {csv_file.name}: {len(df)} rows")
            except Exception as e:
                self.logger.error(f"Error loading {csv_file}: {e}")

        if not dataframes:
            raise ValueError("No data files could be loaded successfully")

        # Combine all dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)
        self.logger.info(f"Combined data: {len(combined_df)} total rows")

        return combined_df

    def extract_forecast_dates(self, df: pd.DataFrame) -> List[datetime]:
        """
        Extract unique forecast dates from the data.

        Args:
            df: Hurricane data DataFrame

        Returns:
            List of unique forecast dates
        """
        if "forecast_time" in df.columns:
            # Convert forecast_time to datetime if it's not already
            if df["forecast_time"].dtype == "object":
                df["forecast_time"] = pd.to_datetime(
                    df["forecast_time"], format="mixed"
                )

            forecast_dates = df["forecast_time"].dt.date.unique()
            forecast_dates = [
                datetime.combine(date, datetime.min.time()) for date in forecast_dates
            ]
            forecast_dates.sort()

            self.logger.info(f"Found {len(forecast_dates)} unique forecast dates")
            return forecast_dates
        else:
            self.logger.warning("No 'forecast_time' column found in data")
            return []

    def get_ensemble_members(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of unique ensemble members.

        Args:
            df: Hurricane data DataFrame

        Returns:
            List of ensemble member identifiers
        """
        if "ensemble_member" in df.columns:
            members = df["ensemble_member"].unique().tolist()
            members.sort()
            self.logger.info(f"Found {len(members)} ensemble members")
            return members
        else:
            self.logger.warning("No 'ensemble_member' column found in data")
            return []

    def get_lead_times(self, df: pd.DataFrame) -> List[int]:
        """
        Get list of unique lead times in hours.

        Args:
            df: Hurricane data DataFrame

        Returns:
            List of lead times in hours
        """
        if "lead_time" in df.columns:
            lead_times = sorted(df["lead_time"].unique().tolist())
            self.logger.info(f"Found {len(lead_times)} unique lead times: {lead_times}")
            return lead_times
        else:
            self.logger.warning("No 'lead_time' column found in data")
            return []

    def filter_by_forecast_date(
        self, df: pd.DataFrame, forecast_date: datetime
    ) -> pd.DataFrame:
        """
        Filter data by specific forecast date.

        Args:
            df: Hurricane data DataFrame
            forecast_date: Forecast date to filter by

        Returns:
            Filtered DataFrame
        """
        if "forecast_time" in df.columns:
            if df["forecast_time"].dtype == "object":
                df["forecast_time"] = pd.to_datetime(
                    df["forecast_time"], format="mixed"
                )

            filtered_df = df[df["forecast_time"].dt.date == forecast_date.date()]
            self.logger.info(
                f"Filtered to forecast date {forecast_date.date()}: {len(filtered_df)} rows"
            )
            return filtered_df
        else:
            self.logger.warning("No 'forecast_time' column found in data")
            return df

    def filter_by_ensemble_member(self, df: pd.DataFrame, member: str) -> pd.DataFrame:
        """
        Filter data by specific ensemble member.

        Args:
            df: Hurricane data DataFrame
            member: Ensemble member identifier

        Returns:
            Filtered DataFrame
        """
        if "ensemble_member" in df.columns:
            filtered_df = df[df["ensemble_member"] == member]
            self.logger.info(
                f"Filtered to ensemble member {member}: {len(filtered_df)} rows"
            )
            return filtered_df
        else:
            self.logger.warning("No 'ensemble_member' column found in data")
            return df

    def filter_by_lead_time(self, df: pd.DataFrame, lead_time: int) -> pd.DataFrame:
        """
        Filter data by specific lead time.

        Args:
            df: Hurricane data DataFrame
            lead_time: Lead time in hours

        Returns:
            Filtered DataFrame
        """
        if "lead_time" in df.columns:
            filtered_df = df[df["lead_time"] == lead_time]
            self.logger.info(
                f"Filtered to lead time {lead_time}h: {len(filtered_df)} rows"
            )
            return filtered_df
        else:
            self.logger.warning("No 'lead_time' column found in data")
            return df

    def calculate_track_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate statistics for hurricane tracks.

        Args:
            df: Hurricane data DataFrame

        Returns:
            Dictionary with track statistics
        """
        stats = {}

        if "latitude" in df.columns and "longitude" in df.columns:
            stats["latitude"] = {
                "mean": df["latitude"].mean(),
                "std": df["latitude"].std(),
                "min": df["latitude"].min(),
                "max": df["latitude"].max(),
            }
            stats["longitude"] = {
                "mean": df["longitude"].mean(),
                "std": df["longitude"].std(),
                "min": df["longitude"].min(),
                "max": df["longitude"].max(),
            }

        if "wind_speed" in df.columns:
            stats["wind_speed"] = {
                "mean": df["wind_speed"].mean(),
                "std": df["wind_speed"].std(),
                "min": df["wind_speed"].min(),
                "max": df["wind_speed"].max(),
            }

        if "pressure" in df.columns:
            stats["pressure"] = {
                "mean": df["pressure"].mean(),
                "std": df["pressure"].std(),
                "min": df["pressure"].min(),
                "max": df["pressure"].max(),
            }

        return stats

    def prepare_track_data(
        self, df: pd.DataFrame, forecast_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Prepare data specifically for track plotting.

        Args:
            df: Hurricane data DataFrame
            forecast_date: Optional forecast date to filter by

        Returns:
            DataFrame prepared for track plotting
        """
        # Filter by forecast date if provided
        if forecast_date:
            df = self.filter_by_forecast_date(df, forecast_date)

        # Ensure we have the required columns
        required_columns = ["latitude", "longitude", "ensemble_member", "lead_time"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(
                f"Missing required columns for track plotting: {missing_columns}"
            )

        # Sort by ensemble member and lead time for proper track ordering
        track_df = df.sort_values(["ensemble_member", "lead_time"])

        # Add track identifier
        track_df["track_id"] = track_df["ensemble_member"].astype(str)

        # Convert lead time to hours if it's not already
        if track_df["lead_time"].dtype == "object":
            track_df["lead_time"] = pd.to_numeric(
                track_df["lead_time"], errors="coerce"
            )

        self.logger.info(
            f"Prepared track data: {len(track_df)} points, {track_df['track_id'].nunique()} tracks"
        )
        return track_df

    def prepare_intensity_data(
        self, df: pd.DataFrame, forecast_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Prepare data specifically for intensity analysis (wind speed, pressure).

        Args:
            df: Hurricane data DataFrame
            forecast_date: Optional forecast date to filter by

        Returns:
            DataFrame prepared for intensity analysis
        """
        # Filter by forecast date if provided
        if forecast_date:
            df = self.filter_by_forecast_date(df, forecast_date)

        # Ensure we have the required columns
        required_columns = ["wind_speed", "pressure", "ensemble_member", "lead_time"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(
                f"Missing required columns for intensity analysis: {missing_columns}"
            )

        # Sort by ensemble member and lead time
        intensity_df = df.sort_values(["ensemble_member", "lead_time"])

        # Convert numeric columns
        intensity_df["wind_speed"] = pd.to_numeric(
            intensity_df["wind_speed"], errors="coerce"
        )
        intensity_df["pressure"] = pd.to_numeric(
            intensity_df["pressure"], errors="coerce"
        )
        intensity_df["lead_time"] = pd.to_numeric(
            intensity_df["lead_time"], errors="coerce"
        )

        # Remove invalid values
        intensity_df = intensity_df.dropna(
            subset=["wind_speed", "pressure", "lead_time"]
        )

        self.logger.info(f"Prepared intensity data: {len(intensity_df)} points")
        return intensity_df

    def calculate_ensemble_spread(
        self, df: pd.DataFrame, forecast_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Calculate ensemble spread statistics across all members.

        Args:
            df: Hurricane data DataFrame
            forecast_date: Optional forecast date to filter by

        Returns:
            DataFrame with ensemble spread statistics
        """
        # Filter by forecast date if provided
        if forecast_date:
            df = self.filter_by_forecast_date(df, forecast_date)

        # Group by lead time and calculate statistics
        group_columns = ["lead_time"]
        if "forecast_time" in df.columns:
            group_columns.append("forecast_time")

        spread_stats = (
            df.groupby(group_columns)
            .agg(
                {
                    "latitude": ["mean", "std", "min", "max"],
                    "longitude": ["mean", "std", "min", "max"],
                    "wind_speed": ["mean", "std", "min", "max"],
                    "pressure": ["mean", "std", "min", "max"],
                }
            )
            .reset_index()
        )

        # Flatten column names
        spread_stats.columns = [
            "_".join(col).strip("_") for col in spread_stats.columns.values
        ]

        self.logger.info(
            f"Calculated ensemble spread for {len(spread_stats)} lead times"
        )
        return spread_stats

    def save_analysis_results(
        self, df: pd.DataFrame, output_path: str, analysis_type: str = "general"
    ):
        """
        Save analysis results to file.

        Args:
            df: DataFrame to save
            output_path: Output file name (will be saved in analysis directory)
            analysis_type: Type of analysis performed

        Returns:
            Full path to the saved file
        """
        # Use the analysis directory that was set up in __init__
        analysis_dir = self.project_root / "data" / "weatherlab" / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        # Create full path
        full_path = analysis_dir / output_path

        # Save the file
        df.to_csv(full_path, index=False)
        self.logger.info(f"Saved {analysis_type} analysis results to: {full_path}")

        return str(full_path)


def main():
    """Main function to demonstrate hurricane data analysis."""
    try:
        # Initialize analyzer
        analyzer = HurricaneAnalyzer()

        # Load all data
        print("Loading hurricane data...")
        df = analyzer.load_all_data()

        # Get basic information
        forecast_dates = analyzer.extract_forecast_dates(df)
        ensemble_members = analyzer.get_ensemble_members(df)
        lead_times = analyzer.get_lead_times(df)

        print(f"\nData Summary:")
        print(f"  - Total data points: {len(df)}")
        print(f"  - Forecast dates: {len(forecast_dates)}")
        print(f"  - Ensemble members: {len(ensemble_members)}")
        print(f"  - Lead times: {len(lead_times)}")

        # Calculate track statistics
        stats = analyzer.calculate_track_statistics(df)
        print(f"\nTrack Statistics:")
        for variable, values in stats.items():
            print(f"  {variable}: mean={values['mean']:.2f}, std={values['std']:.2f}")

        # Prepare data for visualization
        print(f"\nPreparing data for visualization...")
        track_data = analyzer.prepare_track_data(df)
        intensity_data = analyzer.prepare_intensity_data(df)
        ensemble_spread = analyzer.calculate_ensemble_spread(df)

        # Save analysis results
        output_dir = "../data/results/weatherlab/analysis"
        analyzer.save_analysis_results(
            track_data, f"{output_dir}/track_data.csv", "track"
        )
        analyzer.save_analysis_results(
            intensity_data, f"{output_dir}/intensity_data.csv", "intensity"
        )
        analyzer.save_analysis_results(
            ensemble_spread, f"{output_dir}/ensemble_spread.csv", "ensemble_spread"
        )

        print(f"\n✅ Analysis completed successfully!")
        print(f"Results saved to: {output_dir}")

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        raise


if __name__ == "__main__":
    main()
