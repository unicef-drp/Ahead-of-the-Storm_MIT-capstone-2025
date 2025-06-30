# Ahead-of-the-Storm MIT Capstone 2025

A comprehensive hurricane data analysis and visualization system using Google Weather Lab's FNV3 ensemble model.

## Project Overview

This project provides tools to download, process, analyze, and visualize hurricane forecast data from Google Weather Lab's FNV3 ensemble model. The system is specifically configured for Hurricane Rafael (2024) and generates comprehensive visualizations including ensemble tracks, intensity curves, and forecast dashboards.

## Project Structure

```
Ahead-of-the-Storm_MIT-capstone-2025/
│   README.md
│   requirements.txt
│   Makefile
│   .gitignore
│   activate
│   HURRICANE_DATA_README.md
│
├── src/
│   └── data_prep/
│       ├── hurricane_helper/
│       │   ├── hurricane_downloader.py
│       │   ├── hurricane_analyzer.py
│       │   └── hurricane_visualizer.py
│       ├── download_hurricane_data.py
│       ├── analyze_hurricane_data.py
│       ├── plot_forecast_dashboards.py
│       └── plot_single_trajectory.py
│
├── config/
│   └── hurricane_config.yaml
│
├── data/
│   └── weatherlab/
│       ├── raw_ensemble/
│       ├── processed/
│       ├── analysis/
│       └── plots/
│
└── tests/
    └── test_hurricane_system.py
```

## Environment Setup

### Prerequisites
- Python 3.10 or higher
- Conda package manager

### Installation

1. **Create the conda environment:**
   ```bash
   make create-env
   ```

2. **Activate the environment and install dependencies:**
   ```bash
   . ./activate
   ```

This will:
- Create a conda environment named `aots_env`
- Install all required Python dependencies
- Set the `PYTHONPATH` to the project root for module imports

### Manual Installation (Alternative)
```bash
conda create -n aots_env python=3.10
conda activate aots_env
pip install -r requirements.txt
```

## Usage

### 1. Download Hurricane Data
```bash
cd src/data_prep
python download_hurricane_data.py
```
Downloads ensemble forecast data for Hurricane Rafael (2024) from Google Weather Lab.

### 2. Analyze and Visualize Data
```bash
cd src/data_prep
python analyze_hurricane_data.py
```
Generates comprehensive analysis and visualizations including:
- Ensemble tracks
- Intensity curves (wind speed and pressure)
- Ensemble spread analysis
- Summary dashboard

### 3. Generate Forecast-Specific Dashboards
```bash
cd src/data_prep
python plot_forecast_dashboards.py
```
Creates separate dashboard visualizations for all 6-hour forecast intervals from November 4-10, 2024.

### 4. Plot Single Trajectory
```bash
cd src/data_prep
python plot_single_trajectory.py
```
Generates a single trajectory plot for a specific ensemble member and forecast time.

## Configuration

The system uses `config/hurricane_config.yaml` for configuration settings including:
- Hurricane ID (AL182024 for Rafael)
- Date ranges
- Model parameters
- Output directories

## Output

All generated files are saved in the `data/weatherlab/` directory:
- **Raw data**: `data/weatherlab/raw_ensemble/`
- **Processed data**: `data/weatherlab/processed/`
- **Analysis results**: `data/weatherlab/analysis/`
- **Visualizations**: `data/weatherlab/plots/`

## Testing

Run the test suite to verify the system is working correctly:
```bash
python tests/test_hurricane_system.py
```

## Cleanup

To remove the conda environment:
```bash
make clean
```

## Dependencies

- **numpy**: Numerical computing
- **pandas**: Data manipulation and analysis
- **matplotlib**: Plotting and visualization
- **seaborn**: Statistical data visualization
- **requests**: HTTP library for data download
- **pyyaml**: YAML configuration file parsing

## License

MIT Capstone Project 2025 