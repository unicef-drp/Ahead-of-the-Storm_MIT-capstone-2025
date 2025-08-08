.PHONY: create-env install-deps clean download-data download-population download-all create-synthetic-hurricane 

ENV_NAME=aots_env
PYTHON_VERSION=3.10

# Create the conda environment
create-env:
	conda create -y -n $(ENV_NAME) python=$(PYTHON_VERSION)

# Install dependencies into the environment
install-deps:
	conda run -n $(ENV_NAME) pip install -r requirements.txt

# Download Nicaragua OpenStreetMap data
download-data-osm:
	conda run -n $(ENV_NAME) python -m src.data_prep.download_nicaragua_osm_data

# Download Nicaragua population data
download-data-population:
	conda run -n $(ENV_NAME) python -m src.data_prep.download_nicaragua_population_data

# Download hurricane data
download-data-hurricane:
	conda run -n $(ENV_NAME) python -m src.data_prep.download_hurricane_data

# Download both OSM and population data
download-data-all: download-data-osm download-data-population

# Download all data (OSM, population, and hurricane)
download-data-all-complete: download-data-osm download-data-population download-data-hurricane

# Create synthetic hurricane by applying lat/lon offsets
create-synthetic-hurricane:
	conda run -n $(ENV_NAME) python src/data_prep/create_synthetic_hurricane.py

# Plot single trajectory from synthetic hurricane data
plot-synthetic-trajectory:
	conda run -n $(ENV_NAME) python src/data_prep/plot_single_trajectory.py

# Test bathymetry processing
test-bathymetry:
	conda run -n $(ENV_NAME) python test_bathymetry_processing.py

# Test surge layer
test-surge-layer:
	conda run -n $(ENV_NAME) python test_surge_layer.py

# Test surge impact analysis
test-surge-impact:
	conda run -n $(ENV_NAME) python test_surge_impact_analysis.py

# Run surge impact analysis
run-surge-analysis:
	conda run -n $(ENV_NAME) python -c "from src.impact_analysis.analysis.surge_impact import run_surge_impact_analysis; run_surge_impact_analysis('data/preprocessed/weatherlab/synthetic/processed_FNV3_2024_11_04_00_00_ensemble_data_synthetic.csv')"


# Remove the environment (for cleanup)
clean:
	conda env remove -n $(ENV_NAME) -y 