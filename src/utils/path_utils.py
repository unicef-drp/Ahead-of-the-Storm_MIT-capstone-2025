"""
Path utilities for the Ahead of the Storm project.

This module provides shared functions for managing file paths
and directory creation consistently across the project.
"""

from pathlib import Path
from typing import Union, Optional
from .config_utils import get_project_root


def ensure_directory(path: Union[str, Path], create_parents: bool = True) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path
        create_parents: Whether to create parent directories

    Returns:
        Path object for the directory
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=create_parents, exist_ok=True)
    return path_obj


def get_data_path(relative_path: str, create: bool = True) -> Path:
    """
    Get a path relative to the project's data directory.

    Args:
        relative_path: Path relative to data directory (can include 'data/' prefix)
        create: Whether to create the directory if it doesn't exist

    Returns:
        Full path to the data location
    """
    project_root = get_project_root()

    # Handle case where relative_path already includes 'data/' prefix
    if relative_path.startswith("data/"):
        data_path = project_root / relative_path
    else:
        data_path = project_root / "data" / relative_path

    if create and data_path.suffix == "":  # Directory path
        ensure_directory(data_path)

    return data_path


def get_results_path(relative_path: str, create: bool = True) -> Path:
    """
    Get a path relative to the project's results directory.

    Args:
        relative_path: Path relative to results directory
        create: Whether to create the directory if it doesn't exist

    Returns:
        Full path to the results location
    """
    project_root = get_project_root()
    results_path = project_root / "data" / "results" / relative_path

    if create and results_path.suffix == "":  # Directory path
        ensure_directory(results_path)

    return results_path


def get_config_path(config_file: str) -> Path:
    """
    Get the full path to a configuration file.

    Args:
        config_file: Configuration filename

    Returns:
        Full path to the config file
    """
    project_root = get_project_root()
    return project_root / "config" / config_file


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing/replacing invalid characters.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    # Replace invalid characters with underscores
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, "_")

    # Remove leading/trailing spaces and dots
    filename = filename.strip(" .")

    return filename


def create_output_filename(
    base_name: str, suffix: str = "", extension: str = ".csv", timestamp: bool = False
) -> str:
    """
    Create a standardized output filename.

    Args:
        base_name: Base name for the file
        suffix: Optional suffix to add
        extension: File extension (with or without dot)
        timestamp: Whether to add timestamp

    Returns:
        Formatted filename
    """
    from datetime import datetime

    # Ensure extension starts with dot
    if not extension.startswith("."):
        extension = f".{extension}"

    # Build filename parts
    parts = [base_name]

    if suffix:
        parts.append(suffix)

    if timestamp:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        parts.append(timestamp_str)

    # Join parts and add extension
    filename = "_".join(parts) + extension

    return sanitize_filename(filename)
