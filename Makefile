.PHONY: setup-env activate-env install-deps clean download-data-all run-impact-analysis run-full-pipeline

ENV_NAME=aots_env
PYTHON_VERSION=3.10

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

# Create the conda environment
setup-env:
	conda create -y -n $(ENV_NAME) python=$(PYTHON_VERSION)

# Activate environment and install dependencies
activate-env:
	conda activate $(ENV_NAME)
	make install-deps
	export PYTHONPATH=$(pwd)

# Install dependencies into the environment
install-deps:
	conda run -n $(ENV_NAME) pip install -r requirements.txt

# Remove the environment (for cleanup)
clean:
	conda env remove -n $(ENV_NAME) -y

# =============================================================================
# DATA DOWNLOAD COMMANDS
# =============================================================================

# Download Nicaragua OpenStreetMap data
download-osm-data:
	conda run -n $(ENV_NAME) python -m src.data_prep.download_nicaragua_osm_data

# Download Nicaragua population data
download-population-data:
	conda run -n $(ENV_NAME) python -m src.data_prep.download_nicaragua_population_data

# Download hurricane data
download-hurricane-data:
	conda run -n $(ENV_NAME) python -m src.data_prep.download_hurricane_data

# Download landslide data
download-landslide-data:
	conda run -n $(ENV_NAME) python -m src.data_prep.download_landslide_data

# Download nightlights data
download-nightlights-data:
	conda run -n $(ENV_NAME) python -m src.data_prep.download_nightlights_data

# Download discharge data
download-discharge-data:
	conda run -n $(ENV_NAME) python -m src.data_prep.download_nicaragua_discharge

# Download all data types
download-data-all: download-osm-data download-population-data download-hurricane-data download-landslide-data download-nightlights-data download-discharge-data

# =============================================================================
# DATA PROCESSING COMMANDS
# =============================================================================

# Create synthetic hurricane by applying lat/lon offsets
create-synthetic-hurricane:
	conda run -n $(ENV_NAME) python -m src.data_prep.create_synthetic_hurricane

# Create flood exposure data
create-flood-exposure:
	conda run -n $(ENV_NAME) python -m src.data_prep.create_flood_exposure

# Process nightlights data
process-nightlights:
	conda run -n $(ENV_NAME) python -m src.data_prep.process_nightlights_raw

# Create nightlights vulnerability grid
create-nightlights-vulnerability:
	conda run -n $(ENV_NAME) python -m src.data_prep.create_nightlights_vulnerability_grid

# Convert departments to GeoJSON
convert-departments:
	conda run -n $(ENV_NAME) python -m src.data_prep.convert_departments_to_geojson

# =============================================================================
# IMPACT ANALYSIS COMMANDS
# =============================================================================

# Run main impact analysis (all hazard types)
run-impact-analysis:
	conda run -n $(ENV_NAME) python -m src.impact_analysis.main_impact_analysis

# Run specific hazard analysis
run-hurricane-analysis:
	conda run -n $(ENV_NAME) python -c "from src.impact_analysis.analysis.hurricane_impact import run_hurricane_impact_analysis; run_hurricane_impact_analysis()"

run-flood-analysis:
	conda run -n $(ENV_NAME) python -c "from src.impact_analysis.analysis.flood_impact import run_flood_impact_analysis; run_flood_impact_analysis()"

run-landslide-analysis:
	conda run -n $(ENV_NAME) python -c "from src.impact_analysis.analysis.landslide_impact import run_landslide_impact_analysis; run_landslide_impact_analysis()"

run-surge-analysis:
	conda run -n $(ENV_NAME) python -c "from src.impact_analysis.analysis.surge_impact import run_surge_impact_analysis; run_surge_impact_analysis('data/preprocessed/weatherlab/synthetic/processed_FNV3_2024_11_04_00_00_ensemble_data_synthetic.csv')"

# =============================================================================
# VISUALIZATION COMMANDS
# =============================================================================

# Hurricane visualization
plot-hurricane-dashboards:
	conda run -n $(ENV_NAME) python -m src.data_visualization.plot_forecast_dashboards

plot-hurricane-tracks:
	conda run -n $(ENV_NAME) python -m src.data_visualization.analyze_hurricane_data

plot-hurricane-wind:
	conda run -n $(ENV_NAME) python -m src.data_visualization.hurricane_wind_viz

# Flood visualization
plot-flood-maps:
	conda run -n $(ENV_NAME) python -m src.data_visualization.flood_visualizer

plot-discharge-flood:
	conda run -n $(ENV_NAME) python -m src.data_visualization.plot_discharge_flood_boundary

# Landslide visualization
plot-landslide-heatmaps:
	conda run -n $(ENV_NAME) python -m src.data_visualization.landslide_heatmap

# Census and demographic visualization
plot-census-data:
	conda run -n $(ENV_NAME) python -m src.data_visualization.census_viz

plot-poverty-maps:
	conda run -n $(ENV_NAME) python -m src.data_visualization.plot_poverty_regions

plot-unvaccinated-regions:
	conda run -n $(ENV_NAME) python -m src.data_visualization.plot_unvaccinated_regions

# Nightlights visualization
plot-nightlights:
	conda run -n $(ENV_NAME) python -m src.data_visualization.plot_nightlights

# OSM visualization
plot-osm-data:
	conda run -n $(ENV_NAME) python -m src.data_visualization.osm_viz

# =============================================================================
# UTILITY COMMANDS
# =============================================================================

# Clean data and cache
clean-data:
	rm -rf data/results/impact_analysis/cache/*
	rm -rf logs/*.log

clean-cache:
	rm -rf data/results/impact_analysis/cache/*

# Plot single trajectory from synthetic hurricane data
plot-synthetic-trajectory:
	conda run -n $(ENV_NAME) python src/data_prep/plot_single_trajectory.py

# =============================================================================
# FULL PIPELINE COMMANDS
# =============================================================================

# Run complete data preparation pipeline
run-data-prep-pipeline: download-data-all create-synthetic-hurricane create-flood-exposure process-nightlights create-nightlights-vulnerability

# Run complete impact analysis pipeline
run-impact-pipeline: run-impact-analysis

# Run complete visualization pipeline
run-viz-pipeline: plot-hurricane-dashboards plot-flood-maps plot-landslide-heatmaps plot-census-data plot-poverty-maps plot-nightlights

# Run full end-to-end pipeline
run-full-pipeline: run-data-prep-pipeline run-impact-pipeline run-viz-pipeline

# =============================================================================
# HELP AND INFORMATION
# =============================================================================

# Show available commands
help:
	@echo "Available commands:"
	@echo ""
	@echo "Environment Setup:"
	@echo "  setup-env              - Create conda environment"
	@echo "  activate-env           - Activate environment and install deps"
	@echo "  install-deps           - Install Python dependencies"
	@echo "  clean                  - Remove conda environment"
	@echo ""
	@echo "Data Download:"
	@echo "  download-osm-data      - Download OpenStreetMap data"
	@echo "  download-population-data - Download population data"
	@echo "  download-hurricane-data - Download hurricane data"
	@echo "  download-landslide-data - Download landslide data"
	@echo "  download-nightlights-data - Download nightlights data"
	@echo "  download-discharge-data - Download discharge data"
	@echo "  download-data-all      - Download all data types"
	@echo ""
	@echo "Data Processing:"
	@echo "  create-synthetic-hurricane - Create synthetic hurricane tracks"
	@echo "  create-flood-exposure   - Create flood exposure data"
	@echo "  process-nightlights     - Process nightlights data"
	@echo "  create-nightlights-vulnerability - Create vulnerability grid"
	@echo ""
	@echo "Impact Analysis:"
	@echo "  run-impact-analysis    - Run all impact analysis"
	@echo "  run-hurricane-analysis - Run hurricane impact analysis"
	@echo "  run-flood-analysis     - Run flood impact analysis"
	@echo "  run-landslide-analysis - Run landslide impact analysis"
	@echo "  run-surge-analysis     - Run storm surge impact analysis"
	@echo ""
	@echo "Visualization:"
	@echo "  plot-hurricane-dashboards - Plot hurricane forecast dashboards"
	@echo "  plot-flood-maps        - Plot flood maps"
	@echo "  plot-landslide-heatmaps - Plot landslide heatmaps"
	@echo "  plot-census-data       - Plot census data"
	@echo "  plot-poverty-maps      - Plot poverty maps"
	@echo "  plot-nightlights       - Plot nightlights data"
	@echo ""
	@echo "Utility:"
	@echo "  clean-data             - Clean data and cache"
	@echo "  clean-cache            - Clean cache only"
	@echo "  help                   - Show this help message"
	@echo ""
	@echo "Full Pipelines:"
	@echo "  run-data-prep-pipeline - Run complete data preparation"
	@echo "  run-impact-pipeline    - Run complete impact analysis"
	@echo "  run-viz-pipeline       - Run complete visualization"
	@echo "  run-full-pipeline      - Run complete end-to-end pipeline" 