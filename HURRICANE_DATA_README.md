# Hurricane Data Download System

## Overview

This system downloads and analyzes hurricane forecast data from Google Weather Lab's experimental FNV3 model. It's specifically configured for Hurricane Rafael (2024) and provides comprehensive ensemble forecast data including all ensemble members, lead times, and relevant variables (wind speed, pressure, etc.) for plotting forecast tracks.

## Features

- **Automated Data Download**: Downloads FNV3 ensemble data from Google Weather Lab
- **Comprehensive Configuration**: All parameters configurable via YAML file
- **Data Processing**: Automatic cleaning and validation of downloaded data
- **Analysis Tools**: Statistical analysis and data preparation for visualization
- **Visualization**: Create ensemble tracks, intensity curves, and spread analysis plots
- **Error Handling**: Robust error handling with retry logic and detailed logging

## System Architecture

```
├── config/
│   └── hurricane_config.yaml          # Configuration file
├── src/
│   ├── data_prep/
│   │   └── hurricane_helper/
│   │       ├── __init__.py            # Package initialization
│   │       ├── hurricane_downloader.py # Data download module
│   │       ├── hurricane_analyzer.py   # Data analysis module
│   │       └── hurricane_visualizer.py # Visualization module
│   └── download_hurricane_data.py     # Main download script
├── tests/
│   └── test_hurricane_system.py       # System test script
├── data/weatherlab/
│   ├── raw_ensemble/                  # Raw downloaded data
│   ├── processed/                     # Cleaned and processed data
│   ├── analysis/                      # Analysis results
│   └── plots/                         # Generated visualizations
└── logs/                              # Log files
```

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Configuration**:
   The system uses `config/hurricane_config.yaml` for all parameters. Review and modify as needed.

## Configuration

The `config/hurricane_config.yaml` file contains all configurable parameters:

### Hurricane Settings
- **Name**: Hurricane Rafael (2024)
- **Date Range**: October 1-31, 2024 (covers formation to dissipation)
- **Forecast Time**: 12:00 UTC daily

### Model Settings
- **Model**: FNV3 (Google Weather Lab's experimental model)
- **Type**: Ensemble forecasts
- **Track Type**: Paired (includes both track and intensity data)
- **Format**: CSV

### Download Settings
- **Base URL**: Google Weather Lab download endpoint
- **Output Directory**: `data/weatherlab/raw_ensemble`
- **Overwrite Existing**: False (skip existing files)

### Ensemble Settings
- **Download All Members**: True (downloads all available ensemble members)
- **Lead Times**: 0-240 hours (10 days) at 6-hour intervals

### Variables
The system downloads and processes:
- Latitude and Longitude (track position)
- Wind Speed (intensity)
- Pressure (intensity)
- Ensemble Member ID
- Forecast Time
- Lead Time

## Usage

### 1. Download Hurricane Data

Run the main download script:

```bash
python src/download_hurricane_data.py
```

This will:
- Download FNV3 ensemble data for Hurricane Rafael (2024)
- Process and clean the data
- Generate download statistics
- Save data to `data/weatherlab/` directories

### 2. Analyze Downloaded Data

```python
from src.data_prep.hurricane_helper.hurricane_analyzer import HurricaneAnalyzer

# Initialize analyzer
analyzer = HurricaneAnalyzer()

# Load all data
df = analyzer.load_all_data()

# Get basic information
forecast_dates = analyzer.extract_forecast_dates(df)
ensemble_members = analyzer.get_ensemble_members(df)
lead_times = analyzer.get_lead_times(df)

# Prepare data for visualization
track_data = analyzer.prepare_track_data(df)
intensity_data = analyzer.prepare_intensity_data(df)
ensemble_spread = analyzer.calculate_ensemble_spread(df)
```

### 3. Create Visualizations

```python
from src.data_prep.hurricane_helper.hurricane_visualizer import HurricaneVisualizer

# Initialize visualizer
visualizer = HurricaneVisualizer()

# Create different types of plots
visualizer.plot_ensemble_tracks(df, forecast_date)
visualizer.plot_intensity_curves(df, forecast_date)
visualizer.plot_ensemble_spread(df, forecast_date)
visualizer.create_summary_dashboard(df, forecast_date)
```

## Data Structure

### Raw Data Format
The downloaded CSV files contain:
- **ensemble_member**: Unique identifier for each ensemble member
- **forecast_time**: Date and time of the forecast
- **lead_time**: Forecast lead time in hours
- **latitude**: Hurricane center latitude
- **longitude**: Hurricane center longitude
- **wind_speed**: Maximum sustained wind speed (knots)
- **pressure**: Minimum central pressure (millibars)

### Processed Data
After processing, the data includes:
- Data validation and cleaning
- Proper data types
- Sorted by ensemble member and lead time
- Track identifiers for visualization

## Output Files

### Data Files
- `data/weatherlab/raw_ensemble/`: Raw downloaded CSV files
- `data/weatherlab/processed/`: Cleaned and processed data files
- `data/weatherlab/analysis/`: Analysis results and statistics

### Visualization Files
- `data/weatherlab/plots/`: Generated plots including:
  - Ensemble tracks
  - Intensity curves (wind speed and pressure)
  - Ensemble spread analysis
  - Summary dashboards

### Log Files
- `logs/hurricane_download.log`: Download and processing logs
- `logs/failed_downloads.txt`: List of failed downloads

## Example Plots

### Ensemble Tracks
Shows all ensemble member tracks for a specific forecast date, with:
- Different colors for each ensemble member
- Start points (circles) and end points (squares)
- Grid overlay for geographic reference

### Intensity Curves
Displays wind speed and pressure forecasts over time:
- Individual ensemble member curves
- Ensemble mean and standard deviation
- Hurricane category threshold lines

### Ensemble Spread
Statistical analysis showing:
- Latitude and longitude spread over time
- Wind speed and pressure uncertainty
- Min/max ranges and standard deviations

## Error Handling

The system includes robust error handling:

- **Retry Logic**: Failed downloads are retried up to 3 times
- **Continue on Error**: System continues processing even if some downloads fail
- **Detailed Logging**: All operations are logged with timestamps
- **Failed Download Tracking**: Failed downloads are saved to a separate file

## Customization

### Modifying Configuration
Edit `config/hurricane_config.yaml` to:
- Change date ranges
- Modify download parameters
- Adjust processing options
- Configure logging levels

### Adding New Variables
To download additional variables:
1. Add variable names to the `variables` list in the config
2. Update the data processing functions in `src/data_prep/hurricane_helper/hurricane_downloader.py`
3. Modify visualization functions in `src/data_prep/hurricane_helper/hurricane_visualizer.py`

### Supporting Different Hurricanes
To download data for other hurricanes:
1. Update the hurricane name and date range in the config
2. Verify the URL structure for the specific hurricane
3. Run the download script

## Troubleshooting

### Common Issues

1. **No Data Downloaded**
   - Check internet connection
   - Verify URL structure in configuration
   - Check if data is available for the specified date range

2. **Processing Errors**
   - Verify CSV file format
   - Check for missing required columns
   - Review log files for specific error messages

3. **Visualization Issues**
   - Ensure matplotlib and seaborn are installed
   - Check data format and column names
   - Verify output directory permissions

### Log Analysis
Check the log files for detailed information:
- `logs/hurricane_download.log`: Main operation log
- `logs/failed_downloads.txt`: Failed download details

## License and Attribution

This system downloads data from Google Weather Lab's experimental FNV3 model. Please refer to Google's license terms:

- **Historic Data** (>48 hours old): Creative Commons Attribution 4.0 (CC BY 4.0)
- **Real-Time Data** (≤48 hours old): GDM Real-Time Weather Forecasting Experimental Data Terms of Use

## Support

For issues or questions:
1. Check the log files for error details
2. Review the configuration file
3. Verify data availability on Google Weather Lab
4. Check the troubleshooting section above

## Future Enhancements

Potential improvements:
- Support for additional hurricane models
- Real-time data updates
- Advanced statistical analysis
- Interactive web-based visualizations
- Integration with other weather data sources 