# Ahead-of-the-Storm MIT Capstone 2025

A comprehensive **multi-hazard impact analysis system** for disaster risk assessment and vulnerability mapping in Nicaragua. This project integrates hurricane, flood, landslide, and storm surge modeling with demographic and infrastructure vulnerability analysis.

## 🎯 Project Overview

This system provides end-to-end capabilities for:
- **Multi-hazard data collection** (hurricane, flood, landslide, storm surge)
- **Vulnerability assessment** (population, infrastructure, socioeconomic factors)
- **Impact analysis** across different hazard types and vulnerability layers
- **Risk visualization** and mapping for decision support
- **Ensemble forecasting** integration for uncertainty quantification

## 🏗️ Project Structure

```
Ahead-of-the-Storm_MIT-capstone-2025/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── Makefile                    # Build and workflow commands
├── .gitignore                  # Git ignore patterns
├── activate                    # Environment activation script
│
├── config/                     # Configuration files
│   ├── census_config.yaml      # Census data configuration
│   ├── flood_config.yaml       # Flood modeling configuration
│   ├── hurricane_config.yaml   # Hurricane data configuration
│   ├── landslide_config.yaml   # Landslide data configuration
│   ├── surge_config.yaml       # Storm surge configuration
│   ├── nightlights_config.yaml # Nightlights data configuration
│   ├── impact_analysis_config.yaml # Impact analysis configuration
│   └── data_config.yaml        # General data configuration
│
├── src/                        # Source code
│   ├── __init__.py
│   ├── data_prep/             # Data preparation modules
│   │   ├── census_helper/      # Population and census data
│   │   ├── hurricane_helper/   # Hurricane data processing
│   │   ├── landslide_helper/   # Landslide data processing
│   │   ├── flood_exposure/     # Flood modeling and exposure
│   │   ├── nightlights_helper/ # Nightlights data processing
│   │   ├── osm_helper/         # OpenStreetMap data
│   │   └── rainfall_helper/    # Rainfall data processing
│   │
│   ├── impact_analysis/        # Impact analysis modules
│   │   ├── analysis/           # Hazard-specific analysis
│   │   ├── layers/             # Vulnerability and exposure layers
│   │   ├── helper/             # Analysis utilities
│   │   └── utils/              # Analysis utilities
│   │
│   ├── data_visualization/     # Visualization modules
│   │   ├── hurricane_helper/   # Hurricane visualization
│   │   └── ...                 # Other visualization modules
│   │
│   └── utils/                  # Shared utilities
│       ├── config_utils.py     # Configuration management
│       ├── path_utils.py       # Path management
│       ├── logging_utils.py    # Logging utilities
│       └── hurricane_geom.py   # Hurricane geometry utilities
│
├── data/                       # Data storage
│   ├── raw/                    # Raw downloaded data
│   │   ├── census/             # Population data
│   │   ├── bathymetry/         # Bathymetry data
│   │   ├── gadm/               # Administrative boundaries
│   │   ├── sinapred/           # Infrastructure data
│   │   └── flood_riskmap/      # Flood risk maps
│   │
│   ├── preprocessed/           # Processed data
│   │   ├── flood/              # Flood exposure data
│   │   ├── landslide/          # Landslide forecasts
│   │   ├── surge/              # Storm surge data
│   │   ├── weatherlab/         # Hurricane data
│   │   └── nightlights/        # Nightlights data
│   │
│   └── results/                # Analysis results
│       └── impact_analysis/    # Impact analysis outputs
│
├── logs/                       # Log files
├── notebooks/                  # Jupyter notebooks
└── tests/                      # Test scripts
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd Ahead-of-the-Storm_MIT-capstone-2025

# Create and activate conda environment
make setup-env
source ./activate

# Or manually:
conda activate aots_env
export PYTHONPATH=$(pwd)
```

### 2. Install Dependencies

```bash
make install-deps
```

### 3. Download Data

```bash
# Download all data types
make download-data-all

# Or download specific data types:
make download-hurricane-data
make download-population-data
make download-osm-data
make download-landslide-data
make download-nightlights-data
```

### 4. Run Analysis

```bash
# Run complete impact analysis
make run-impact-analysis

# Or run specific hazard analysis:
make run-hurricane-analysis
make run-flood-analysis
make run-landslide-analysis
make run-surge-analysis
```

### 5. Generate Visualizations

```bash
# Generate all visualizations
make run-viz-pipeline

# Or specific visualizations:
make plot-hurricane-dashboards
make plot-flood-maps
make plot-landslide-heatmaps
```

## 📊 Available Commands

Run `make help` to see all available commands:

### Environment Management
- `make setup-env` - Create conda environment
- `make activate-env` - Activate environment and install dependencies
- `make install-deps` - Install Python dependencies
- `make clean` - Remove conda environment

### Data Download
- `make download-osm-data` - Download OpenStreetMap data
- `make download-population-data` - Download population data
- `make download-hurricane-data` - Download hurricane data
- `make download-landslide-data` - Download landslide data
- `make download-nightlights-data` - Download nightlights data
- `make download-discharge-data` - Download discharge data
- `make download-data-all` - Download all data types

### Data Processing
- `make create-synthetic-hurricane` - Create synthetic hurricane tracks
- `make create-flood-exposure` - Create flood exposure data
- `make process-nightlights` - Process nightlights data
- `make create-nightlights-vulnerability` - Create vulnerability grid

### Impact Analysis
- `make run-impact-analysis` - Run all impact analysis
- `make run-hurricane-analysis` - Run hurricane impact analysis
- `make run-flood-analysis` - Run flood impact analysis
- `make run-landslide-analysis` - Run landslide impact analysis
- `make run-surge-analysis` - Run storm surge impact analysis

### Visualization
- `make plot-hurricane-dashboards` - Plot hurricane forecast dashboards
- `make plot-flood-maps` - Plot flood maps
- `make plot-landslide-heatmaps` - Plot landslide heatmaps
- `make plot-census-data` - Plot census data
- `make plot-poverty-maps` - Plot poverty maps
- `make plot-nightlights` - Plot nightlights data

### Full Pipelines
- `make run-data-prep-pipeline` - Run complete data preparation
- `make run-impact-pipeline` - Run complete impact analysis
- `make run-viz-pipeline` - Run complete visualization
- `make run-full-pipeline` - Run complete end-to-end pipeline

## 🔧 Configuration

The system uses YAML configuration files for all parameters:

### Key Configuration Files
- **`config/impact_analysis_config.yaml`** - Main impact analysis parameters
- **`config/hurricane_config.yaml`** - Hurricane data and modeling parameters
- **`config/flood_config.yaml`** - Flood modeling parameters
- **`config/landslide_config.yaml`** - Landslide analysis parameters
- **`config/surge_config.yaml`** - Storm surge modeling parameters
- **`config/census_config.yaml`** - Population data parameters

### Configuration Structure
```yaml
impact_analysis:
  cache:
    use_cache: true
  grid:
    resolution_degrees: 0.1
  output:
    base_directory: "data/results/impact_analysis"
  runs:
    surge:
      - schools
      - population
      - health_facilities
```

## 📁 Data Organization

### Input Data
- **Population**: WorldPop high-resolution population data
- **Infrastructure**: OpenStreetMap data (schools, hospitals, shelters)
- **Hazards**: Hurricane forecasts, flood maps, landslide predictions, storm surge
- **Vulnerability**: Poverty data, vaccination rates, nightlights

### Output Data
- **Exposure layers**: Hazard-specific exposure maps
- **Vulnerability layers**: Demographic and infrastructure vulnerability
- **Impact maps**: Combined hazard-vulnerability impact assessments
- **Risk visualizations**: Maps, charts, and dashboards

## 🎯 Use Cases

### 1. Emergency Response Planning
- Identify vulnerable populations and critical infrastructure
- Assess exposure to multiple hazards
- Prioritize evacuation and response efforts

### 2. Risk Assessment
- Quantify risk across different hazard types
- Identify high-risk areas for targeted interventions
- Support long-term planning and policy decisions

### 3. Infrastructure Planning
- Site critical facilities in low-risk areas
- Design resilient infrastructure systems
- Optimize resource allocation

### 4. Community Engagement
- Communicate risk to stakeholders
- Support community-based disaster preparedness
- Facilitate participatory planning processes

## 🔬 Technical Details

### Hazard Models
- **Hurricane**: Google Weather Lab FNV3 ensemble model
- **Flood**: Hydrodynamic modeling with terrain analysis
- **Landslide**: NASA LHASA-F landslide hazard assessment
- **Storm Surge**: Ensemble storm surge modeling

### Vulnerability Assessment
- **Demographic**: Age, gender, poverty, vaccination status
- **Infrastructure**: Schools, hospitals, shelters, roads
- **Socioeconomic**: Poverty levels, access to services
- **Physical**: Nightlights as proxy for development

### Analysis Methods
- **Spatial overlay**: Hazard-vulnerability intersection analysis
- **Ensemble methods**: Uncertainty quantification
- **Multi-criteria**: Weighted vulnerability assessment
- **Impact-prone analysis**: High-risk area identification

## 🚨 Troubleshooting

### Common Issues
1. **Environment activation**: Ensure you're in the conda environment
2. **Path issues**: Check that `PYTHONPATH` is set correctly
3. **Data missing**: Run data download commands first
4. **Configuration errors**: Verify YAML syntax and file paths

### Getting Help
- Run `make help` for command overview
- Check logs in the `logs/` directory
- Review configuration files for parameter settings
- Ensure all dependencies are installed

## 🤝 Contributing

### Development Workflow
1. **Create feature branch** from main
2. **Follow coding standards**: PEP8, proper imports, configuration-driven
3. **Test thoroughly** with separate test scripts
4. **Update documentation** and configuration files
5. **Submit pull request** with clear description

### Code Standards
- **Import patterns**: `from src.module.submodule import Class`
- **Configuration**: Use YAML config files for parameters
- **Error handling**: Proper logging and exception handling
- **Documentation**: Docstrings and inline comments

## 📚 Dependencies

### Core Dependencies
- **numpy**: Numerical computing
- **pandas**: Data manipulation and analysis
- **geopandas**: Geospatial data handling
- **rasterio**: Raster data processing
- **matplotlib**: Plotting and visualization
- **scikit-learn**: Machine learning utilities

### Geospatial Dependencies
- **fiona**: Vector data I/O
- **shapely**: Geometric operations
- **pyproj**: Coordinate transformations
- **rasterstats**: Raster statistics

### Configuration & Utilities
- **pyyaml**: YAML configuration parsing
- **requests**: HTTP data download
- **tqdm**: Progress bars
- **logging**: Structured logging

## 📄 License

MIT Capstone Project 2025 - Massachusetts Institute of Technology

## 🙏 Acknowledgments

- **Google Weather Lab** for hurricane forecast data
- **NASA** for landslide hazard assessment data
- **WorldPop** for population data
- **OpenStreetMap** contributors for infrastructure data
- **MIT faculty and staff** for project guidance and support

---

**For questions or support, please refer to the project documentation or contact the development team.** 