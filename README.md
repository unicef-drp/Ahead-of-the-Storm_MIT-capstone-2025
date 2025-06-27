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
│   │   ├── overpass_client.py
│   │   ├── data_downloader.py
│   │   └── download_nicaragua_data.py
│   ├── impact_analysis/
│   ├── optimization/
│
├── config/
│   └── data_config.yaml
│
├── data/
│   └── raw/
│
├── notebooks/
```

- `src/`: Source code for the project
  - `data_prep/`: Data preparation scripts
    - `overpass_client.py`: Client for Overpass API
    - `data_downloader.py`: Main data downloader class
    - `download_nicaragua_data.py`: Script to download Nicaragua data
  - `impact_analysis/`: Impact analysis modules
  - `optimization/`: Optimization algorithms
- `config/`: Configuration files and parameters
  - `data_config.yaml`: Configuration for data download
- `data/`: Data storage (not tracked by git)
  - `raw/`: Raw downloaded data
- `notebooks/`: Jupyter notebooks for ad-hoc analysis

## Environment Setup

To set up your Python environment for this project:

1. **Create the conda environment:**
   ```bash
   make env
   ```

2. **Activate the environment:**
   ```bash
   . ./activate
   ```

This will:
- Create a new conda environment named `aots_env` with Python 3.10
- Install all required dependencies from `requirements.txt`
- Set the `PYTHONPATH` to include the project root

## Data Download

To download Nicaragua data from OpenStreetMap:

```bash
make download-data
```

This will download the following data categories:
- **Roads**: Highway networks (excluding footways, paths, cycleways)
- **Schools**: Educational institutions (schools, universities, colleges)
- **Hospitals**: Healthcare facilities (hospitals, clinics)
- **Population**: Population centers (cities, towns, villages, hamlets)
- **Vaccination Centers**: Pharmacies and vaccination centers
- **Childcare**: Kindergartens, childcare facilities, daycares
- **Baby Goods**: Baby goods stores, convenience stores, supermarkets

All data is saved in GeoJSON format in the `data/raw/` directory.

## Configuration

The data download behavior can be customized by editing `config/data_config.yaml`. You can:
- Enable/disable specific data categories
- Modify search tags for each category
- Change output format and coordinate reference system
- Adjust API timeout settings

## Environment Cleanup

To remove the conda environment:
```bash
make clean
``` 