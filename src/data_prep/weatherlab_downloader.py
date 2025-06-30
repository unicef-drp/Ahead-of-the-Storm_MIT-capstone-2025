#!/usr/bin/env python3
"""
Google DeepMind WeatherLab ensemble data downloader.
Downloads hurricane ensemble forecasts and supports track modification.
"""

import requests
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
from io import StringIO
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeatherLabDownloader:
    """Downloader for Google DeepMind WeatherLab ensemble hurricane data."""
    
    def __init__(self, output_dir="data/weatherlab", track_type="ensemble"):
        """
        Initialize WeatherLab downloader.
        
        Args:
            output_dir (str): Directory to save downloaded data
            track_type (str): Track type to download (ensemble, ensemble_mean, best_track)
        """
        self.base_url = "https://deepmind.google.com/science/weatherlab/download"
        self.output_dir = output_dir
        self.track_type = track_type
        self.ensure_output_dir()
        
        # Hurricane Rafael 2024 dates (extended timeline)
        self.rafael_dates = [
            "2024-11-01", "2024-11-02", "2024-11-03", "2024-11-04",
            "2024-11-05", "2024-11-06", "2024-11-07", "2024-11-08", 
            "2024-11-09", "2024-11-10", "2024-11-11", "2024-11-12",
            "2024-11-13", "2024-11-14", "2024-11-15"
        ]
        
        # Forecast times
        self.forecast_times = ["00:00", "06:00", "12:00", "18:00"]
        
        # Nicaragua coordinates for track modification
        self.nicaragua_coords = {
            'lat_min': 10.7, 'lat_max': 15.0,
            'lon_min': -87.7, 'lon_max': -82.7
        }
    
    def ensure_output_dir(self):
        """Create output directory if it doesn't exist."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/raw_{self.track_type}", exist_ok=True)
        os.makedirs(f"{self.output_dir}/modified_{self.track_type}", exist_ok=True)
        os.makedirs(f"{self.output_dir}/metadata", exist_ok=True)
    
    def generate_url(self, date, time, model="FNV3", track_type=None, 
                    paired=True, format_type="csv"):
        """
        Generate WeatherLab download URL.
        
        Args:
            date (str): Date in YYYY-MM-DD format
            time (str): Time in HH:MM format
            model (str): Model version (default: FNV3)
            track_type (str): Track type (default: self.track_type)
            paired (bool): Whether to use paired tracks
            format_type (str): Output format (default: csv)
        
        Returns:
            str: Complete download URL
        """
        # Use instance track_type if not provided
        if track_type is None:
            track_type = self.track_type
        # Parse date and time
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        time_obj = datetime.strptime(time, "%H:%M")
        
        # Generate filename
        filename = f"{model}_{date_obj.strftime('%Y_%m_%d')}T{time_obj.strftime('%H_%M')}_paired.csv"
        
        # Build URL
        url = f"{self.base_url}/cyclones/{model}/{track_type}/paired/{format_type}/{filename}"
        
        return url
    
    def download_data(self, url, filename=None):
        """
        Download data from WeatherLab URL.
        
        Args:
            url (str): Download URL
            filename (str): Optional filename to save as
        
        Returns:
            pd.DataFrame: Downloaded data or None if failed
        """
        try:
            logger.info(f"Downloading: {url}")
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                logger.info(f"✓ Downloaded {len(response.content)} bytes")
                
                # Parse CSV data
                # Skip header comments (lines starting with #)
                lines = response.text.split('\n')
                data_lines = [line for line in lines if not line.startswith('#') and line.strip()]
                data_text = '\n'.join(data_lines)
                
                df = pd.read_csv(StringIO(data_text))
                logger.info(f"✓ Parsed CSV: {df.shape}")
                
                # Save raw data
                if filename:
                    filepath = f"{self.output_dir}/raw_{self.track_type}/{filename}"
                    df.to_csv(filepath, index=False)
                    logger.info(f"✓ Saved to: {filepath}")
                
                return df
                
            else:
                logger.error(f"✗ HTTP error {response.status_code}: {response.text[:200]}")
                return None
                
        except Exception as e:
            logger.error(f"✗ Download error: {e}")
            return None
    
    def test_data_availability(self, max_test_dates=5):
        """
        Test which dates have available data.
        
        Args:
            max_test_dates (int): Maximum number of dates to test
        
        Returns:
            list: List of dates with available data
        """
        logger.info("=" * 60)
        logger.info("TESTING DATA AVAILABILITY")
        logger.info("=" * 60)
        
        available_dates = []
        
        for i, date in enumerate(self.rafael_dates[:max_test_dates]):
            logger.info(f"Testing date {i+1}/{min(max_test_dates, len(self.rafael_dates))}: {date}")
            
            # Test one time slot per date
            test_url = self.generate_url(date, "12:00")
            
            try:
                response = requests.get(test_url, timeout=30)
                
                if response.status_code == 200:
                    logger.info(f"✓ Data available for {date}")
                    available_dates.append(date)
                else:
                    logger.info(f"✗ No data for {date} (HTTP {response.status_code})")
                    
            except Exception as e:
                logger.info(f"✗ Error testing {date}: {e}")
        
        logger.info(f"Found {len(available_dates)} dates with available data")
        return available_dates
    
    def download_rafael_data(self, max_files=None, test_availability=True):
        """
        Download Hurricane Rafael ensemble data.
        
        Args:
            max_files (int): Maximum number of files to download (None for all)
            test_availability (bool): Whether to test data availability first
        
        Returns:
            list: List of downloaded DataFrames
        """
        logger.info("=" * 60)
        logger.info(f"DOWNLOADING HURRICANE RAFAEL WEATHERLAB DATA ({self.track_type})")
        logger.info("=" * 60)
        
        # Test data availability if requested
        if test_availability and max_files is None:
            available_dates = self.test_data_availability()
            if available_dates:
                logger.info(f"Using available dates: {available_dates}")
                dates_to_download = available_dates
            else:
                logger.warning("No data available, using original date range")
                dates_to_download = self.rafael_dates
        else:
            dates_to_download = self.rafael_dates
        
        downloaded_data = []
        url_count = 0
        
        for date in dates_to_download:
            for time in self.forecast_times:
                if max_files and url_count >= max_files:
                    break
                
                url = self.generate_url(date, time)
                filename = f"rafael_{date}_{time.replace(':', '_')}.csv"
                
                df = self.download_data(url, filename)
                if df is not None:
                    downloaded_data.append({
                        'date': date,
                        'time': time,
                        'data': df,
                        'filename': filename
                    })
                
                url_count += 1
        
        logger.info(f"✓ Downloaded {len(downloaded_data)} files")
        return downloaded_data
    
    def analyze_data_structure(self, df):
        """
        Analyze the structure of downloaded data.
        
        Args:
            df (pd.DataFrame): Downloaded data
        
        Returns:
            dict: Analysis results
        """
        analysis = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'missing_values': df.isnull().sum().to_dict(),
            'unique_track_ids': df['track_id'].nunique() if 'track_id' in df.columns else 0,
            'unique_samples': df['sample'].nunique() if 'sample' in df.columns else 0,
            'date_range': {
                'min': str(df['valid_time'].min()) if 'valid_time' in df.columns else None,
                'max': str(df['valid_time'].max()) if 'valid_time' in df.columns else None
            }
        }
        
        return analysis
    
    def modify_track_for_nicaragua(self, df, modification_factor=0.3):
        """
        Modify hurricane track to simulate Nicaragua landfall.
        
        Args:
            df (pd.DataFrame): Original track data
            modification_factor (float): How much to modify coordinates (0-1)
        
        Returns:
            pd.DataFrame: Modified track data
        """
        logger.info("Modifying track for Nicaragua landfall simulation...")
        
        # Create copy to avoid modifying original
        modified_df = df.copy()
        
        # Calculate Nicaragua center
        nicaragua_center_lat = (self.nicaragua_coords['lat_min'] + self.nicaragua_coords['lat_max']) / 2
        nicaragua_center_lon = (self.nicaragua_coords['lon_min'] + self.nicaragua_coords['lon_max']) / 2
        
        # Shift ALL tracks by the same amount to simulate Nicaragua landfall
        if 'lon' in modified_df.columns and 'lat' in modified_df.columns:
            # Calculate the shift needed to move tracks toward Nicaragua
            # Shift longitude westward by 5 degrees
            lon_shift = -5.0 * modification_factor
            
            # Shift latitude toward Nicaragua center
            # Calculate current average position
            current_avg_lat = modified_df['lat'].mean()
            current_avg_lon = modified_df['lon'].mean()
            
            # Calculate shift toward Nicaragua
            lat_shift = (nicaragua_center_lat - current_avg_lat) * modification_factor
            
            # Apply shifts to ALL tracks
            modified_df['lon'] = modified_df['lon'] + lon_shift
            modified_df['lat'] = modified_df['lat'] + lat_shift
            
            logger.info(f"Modified ALL {len(modified_df)} track points")
            logger.info(f"Applied shifts: lon={lon_shift:.2f}°, lat={lat_shift:.2f}°")
        
        # Adjust intensity for land interaction
        if 'minimum_sea_level_pressure_hpa' in modified_df.columns:
            # Increase pressure (weaken storm) when over land
            in_nicaragua = (
                (modified_df['lat'] >= self.nicaragua_coords['lat_min']) &
                (modified_df['lat'] <= self.nicaragua_coords['lat_max']) &
                (modified_df['lon'] >= self.nicaragua_coords['lon_min']) &
                (modified_df['lon'] <= self.nicaragua_coords['lon_max'])
            )
            
            if in_nicaragua.any():
                # Increase pressure by 10-20 hPa over land
                pressure_increase = np.random.uniform(10, 20, in_nicaragua.sum())
                modified_df.loc[in_nicaragua, 'minimum_sea_level_pressure_hpa'] += pressure_increase
                logger.info(f"Adjusted pressure for {in_nicaragua.sum()} land points")
        
        return modified_df
    
    def save_modified_tracks(self, original_data, modified_data, filename_prefix="rafael_nicaragua"):
        """
        Save modified tracks to files.
        
        Args:
            original_data (list): List of original data dictionaries
            modified_data (list): List of modified data dictionaries
            filename_prefix (str): Prefix for output files
        """
        logger.info("Saving modified tracks...")
        
        for i, (orig, mod) in enumerate(zip(original_data, modified_data)):
            # Save modified data
            mod_filename = f"{filename_prefix}_{orig['date']}_{orig['time'].replace(':', '_')}.csv"
            mod_filepath = f"{self.output_dir}/modified_{self.track_type}/{mod_filename}"
            mod['data'].to_csv(mod_filepath, index=False)
            
            # Save comparison metadata
            comparison = {
                'original_file': orig['filename'],
                'modified_file': mod_filename,
                'original_shape': orig['data'].shape,
                'modified_shape': mod['data'].shape,
                'modification_date': datetime.now().isoformat(),
                'nicaragua_coords': self.nicaragua_coords
            }
            
            meta_filename = f"comparison_{orig['date']}_{orig['time'].replace(':', '_')}.json"
            meta_filepath = f"{self.output_dir}/metadata/{meta_filename}"
            
            with open(meta_filepath, 'w') as f:
                json.dump(comparison, f, indent=2)
            
            logger.info(f"✓ Saved: {mod_filename}")
    
    def create_summary_report(self, original_data, modified_data):
        """
        Create a summary report of the download and modification process.
        
        Args:
            original_data (list): List of original data dictionaries
            modified_data (list): List of modified data dictionaries
        """
        logger.info("Creating summary report...")
        
        report = {
            'download_summary': {
                'total_files': len(original_data),
                'download_date': datetime.now().isoformat(),
                'hurricane': 'Rafael (2024)',
                'model': 'WeatherLab FNV3',
                'data_type': 'Ensemble forecasts'
            },
            'data_analysis': [],
            'modification_summary': {
                'target_region': 'Nicaragua',
                'modification_type': 'Track shift and intensity adjustment',
                'nicaragua_coordinates': self.nicaragua_coords
            }
        }
        
        # Analyze each file
        for orig, mod in zip(original_data, modified_data):
            orig_analysis = self.analyze_data_structure(orig['data'])
            mod_analysis = self.analyze_data_structure(mod['data'])
            
            file_summary = {
                'date': orig['date'],
                'time': orig['time'],
                'original': orig_analysis,
                'modified': mod_analysis
            }
            
            report['data_analysis'].append(file_summary)
        
        # Save report
        report_filepath = f"{self.output_dir}/metadata/download_summary.json"
        with open(report_filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"✓ Summary report saved: {report_filepath}")
        return report

def main():
    """Main function to download and modify Hurricane Rafael data."""
    
    # Initialize downloader
    downloader = WeatherLabDownloader()
    
    # Download Rafael data (full timeline)
    logger.info("Starting WeatherLab data download...")
    original_data = downloader.download_rafael_data(max_files=None, test_availability=True)
    
    if not original_data:
        logger.error("No data downloaded. Exiting.")
        return
    
    # Modify tracks for Nicaragua landfall
    modified_data = []
    for data_dict in original_data:
        modified_df = downloader.modify_track_for_nicaragua(data_dict['data'])
        modified_data.append({
            'date': data_dict['date'],
            'time': data_dict['time'],
            'data': modified_df,
            'filename': data_dict['filename']
        })
    
    # Save modified tracks
    downloader.save_modified_tracks(original_data, modified_data)
    
    # Create summary report
    downloader.create_summary_report(original_data, modified_data)
    
    logger.info("=" * 60)
    logger.info("WEATHERLAB DOWNLOAD COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Downloaded: {len(original_data)} files")
    logger.info(f"Modified: {len(modified_data)} files")
    logger.info(f"Output directory: {downloader.output_dir}")

if __name__ == "__main__":
    main() 