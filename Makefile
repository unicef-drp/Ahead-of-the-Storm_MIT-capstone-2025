.PHONY: create-env install-deps clean download-data download-population download-all

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

# Download both OSM and population data
download-data-all: download-data-osm download-data-population

# Remove the environment (for cleanup)
clean:
	conda env remove -n $(ENV_NAME) -y 