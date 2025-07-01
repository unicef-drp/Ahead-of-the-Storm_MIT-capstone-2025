"""
Census and population data package.

This package contains modules for downloading and processing
census data including population counts by age and gender.
"""

from .population_downloader import PopulationDownloader

__all__ = ["PopulationDownloader"]
