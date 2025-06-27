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
│   ├── impact_analysis/
│   ├── optimization/
│
├── config/
│
├── data/
│
├── notebooks/
```

- `src/`: Source code for the project
  - `data_prep/`: Data preparation scripts
  - `impact_analysis/`: Impact analysis modules
  - `optimization/`: Optimization algorithms
- `config/`: Configuration files and parameters
- `data/`: Data storage (not tracked by git)
- `notebooks/`: Jupyter notebooks for ad-hoc analysis

## Environment Setup

To set up your Python environment for this project:

1. **Create the conda environment:**
   ```bash
   make env
   ```
2. **Activate the environment and install dependencies:**
   ```bash
   . ./activate
   ```

This will:
- Create a conda environment named `aots_env` (if it doesn't exist)
- Install all required Python dependencies
- Set the `PYTHONPATH` to the project root for module imports

If you need to remove the environment, run:
```bash
make clean
``` 