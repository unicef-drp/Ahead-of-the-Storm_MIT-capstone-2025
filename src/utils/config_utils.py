"""
Configuration utilities for the Ahead of the Storm project.

This module provides shared functions for loading configuration files
and managing project paths.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def get_project_root() -> Path:
    """
    Get the project root directory.

    Returns:
        Path to the project root directory
    """
    # Go up from src/utils to project root
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    return project_root


def load_config(
    config_path: str, relative_to_project_root: bool = True
) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the configuration file
        relative_to_project_root: If True, config_path is relative to project root

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    if relative_to_project_root:
        project_root = get_project_root()
        config_file = project_root / config_path
    else:
        config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in config file {config_file}: {e}")


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get a nested configuration value using dot notation.

    Args:
        config: Configuration dictionary
        key_path: Dot-separated path to the key (e.g., 'data.input.source_file')
        default: Default value if key doesn't exist

    Returns:
        Configuration value or default
    """
    keys = key_path.split(".")
    value = config

    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default
