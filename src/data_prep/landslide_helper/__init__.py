"""
Landslide hazard data package.

This package contains modules for downloading and processing
NASA LHASA-F landslide hazard prediction data.
"""

from .landslide_downloader import LandslideDownloader

__all__ = ["LandslideDownloader"] 