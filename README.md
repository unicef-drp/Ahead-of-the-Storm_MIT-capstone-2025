# Ahead-of-the-Storm MIT Capstone 2025

## Project Structure

```
Ahead-of-the-Storm_MIT-capstone-2025/
│   README.md
│   requirements.txt
│   Makefile
│   .gitignore
│   activate
│
├── src/
│   ├── data_prep/
│   │   ├── __init__.py
│   │   ├── osm_helper/
│   │   │   ├── __init__.py
│   │   │   ├── overpass_client.py
│   │   │   └── osm_downloader.py
│   │   ├── census/
│   │   │   ├── __init__.py
│   │   │   └── population_downloader.py
│   │   ├── download_nicaragua_osm_data.py
│   │   └── download_nicaragua_population_data.py
│   ├── impact_analysis/
│   ├── optimization/
│
├── config/
│   ├── data_config.yaml
│   └── census_config.yaml
│
├── data/
│   ├── raw/
│   │   ├── osm/
│   │   └── census/
│   └── processed/
│       └── census/
│
├── notebooks/
```

- `src/`: Source code for the project
  - `data_prep/`: Data preparation scripts
    - `osm_helper/`: OpenStreetMap data processing package
      - `overpass_client.py`: Client for Overpass API
      - `osm_downloader.py`: OpenStreetMap data downloader class
    - `census/`: Census data processing package
      - `population_downloader.py`: Script to download population data
    - `download_nicaragua_osm_data.py`: Script to download Nicaragua OpenStreetMap data
    - `download_nicaragua_population_data.py`: Script to download population data
  - `impact_analysis/`: Impact analysis modules
  - `optimization/`: Optimization algorithms
- `config/`: Configuration files and parameters
  - `data_config.yaml`: Configuration for OpenStreetMap data download
    - Timeout settings
    - Data categories to download
    - Output formats
    - Logging settings
  - `census_config.yaml`: Configuration for population data download
    - Data sources (WorldPop, UN WPP)
    - Age group definitions
    - Gender categories
    - Administrative unit levels
    - Processing settings
- `data/`: Data storage (not tracked by git)
  - `raw/`: Raw downloaded data
  - `osm/`: OpenStreetMap data
  - `census/`: Population data
  - `processed/`: Processed data
    - `census/`: Census data
- `notebooks/`: Jupyter notebooks for ad-hoc analysis

## Usage

### Environment Setup

1. Create the conda environment:
   ```bash
   make env
   ```

2. Install dependencies:
   ```bash
   make install-deps
   ```

3. Activate the environment:
   ```bash
   conda activate aots_env
   ```

### Data Download

#### OpenStreetMap Data
Download Nicaragua OpenStreetMap data (roads, buildings, etc.):
```bash
make download-data
```

#### Population Data
Download Nicaragua population data with age and gender breakdowns:
```bash
make download-population
```

This will download:
- UN World Population Prospects data with age and gender breakdowns
- WorldPop spatial population data (if available)
- Processed CSV files with population counts by:
  - Age groups (0-4, 5-9, 10-14, ..., 80+)
  - Gender (Male/Female)
  - Combined age and gender breakdowns

### Manual Execution

If you prefer to run scripts directly without `make`:

1. Activate the conda environment:
   ```bash
   conda activate aots_env
   ```

2. Set the Python path:
   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

3. Run the scripts:
   ```bash
   # Download OpenStreetMap data
   python src/data_prep/download_nicaragua_osm_data.py
   
   # Download population data
   python src/data_prep/download_nicaragua_population_data.py
   ```

## Configuration

The project uses YAML configuration files to manage parameters:

- `config/data_config.yaml`: Configuration for OpenStreetMap data download
  - Timeout settings
  - Data categories to download
  - Output formats
  - Logging settings

- `config/census_config.yaml`: Configuration for population data download
  - Data sources (WorldPop, UN WPP)
  - Age group definitions
  - Gender categories
  - Administrative unit levels
  - Processing settings

## Environment Cleanup

To remove the conda environment:
```bash
make clean
``` 