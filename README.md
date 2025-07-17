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
│
└── tests/
    └── test_hurricane_system.py
```


1. Create the conda environment:
   ```bash
   make env
   ```


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

## Landslide Pipeline File Naming Convention

All files in the landslide data pipeline now follow a clear, descriptive, and reproducible naming convention:

- **Raw Data:**
  - `landslide_forecast_48h_YYYYMMDDTHHMM.tif`
  - Example: `landslide_forecast_48h_20250716T0600.tif`
  - Contains global forecast data; no region in the name.

- **Processed Data (Nicaragua):**
  - `landslide_forecast_48h_YYYYMMDDTHHMM_nicaragua.tif`
  - Example: `landslide_forecast_48h_20250716T0600_nicaragua.tif`
  - Indicates the data is clipped to Nicaragua.

- **Results/Plots:**
  - `landslide_heatmap_48h_YYYYMMDDTHHMM_nicaragua.png`
  - Example: `landslide_heatmap_48h_20250716T0600_nicaragua.png`
  - Visualization of the processed data for Nicaragua.

**Rationale:**
- No script-run timestamps; only the forecast initialization time is used.
- "nicaragua" is only present for region-specific outputs.
- This makes the pipeline outputs easy to interpret, sort, and compare.

**Cleanup Process:**
- All debug/test scripts and artifacts have been removed.
- Only files matching the above conventions are kept in the repo.
- `.gitkeep` files are used to ensure empty folders are tracked by git. 