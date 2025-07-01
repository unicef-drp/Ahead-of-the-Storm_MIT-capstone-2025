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