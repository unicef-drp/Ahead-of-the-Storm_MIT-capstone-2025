# Makefile for Ahead of the Storm MIT Capstone 2025

.PHONY: help create-env clean install-deps download-ecmwf process-hurricanes run-notebook

help:
	@echo "Available commands:"
	@echo "  create-env        - Create conda environment"
	@echo "  clean            - Remove conda environment"
	@echo "  install-deps     - Install Python dependencies"
	@echo "  download-ecmwf   - Download ECMWF hurricane data"
	@echo "  process-hurricanes - Process downloaded hurricane data"
	@echo "  run-notebook     - Start Jupyter notebook server"

create-env:
	conda create -n aots_env python=3.9 -y

clean:
	conda env remove -n aots_env -y

install-deps:
	pip install -r requirements.txt

download-ecmwf:
	@echo "Downloading ECMWF hurricane data..."
	python download_hurricane_data.py --process

process-hurricanes:
	@echo "Processing hurricane data..."
	python -c "from src.data_prep.hurricane_data_processor import HurricaneDataProcessor; processor = HurricaneDataProcessor(); processor.export_processed_data()"

run-notebook:
	jupyter notebook notebooks/hurricane_data_analysis.ipynb

setup: create-env install-deps
	@echo "Environment setup complete. Activate with: conda activate aots_env" 