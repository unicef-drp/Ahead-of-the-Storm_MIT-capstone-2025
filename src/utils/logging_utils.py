"""
Logging utilities for the Ahead of the Storm project.

This module provides shared functions for setting up logging
consistently across the project.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logging(
    name: str,
    log_dir: str = "logs",
    level: int = logging.INFO,
    include_console: bool = True,
) -> logging.Logger:
    """
    Setup logging configuration for a module.

    Args:
        name: Logger name (usually __name__)
        log_dir: Directory to store log files
        level: Logging level
        include_console: Whether to include console handler

    Returns:
        Configured logger instance
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{name.replace('.', '_')}_{timestamp}.log"
    log_file = log_path / log_filename

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Add file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Add console handler if requested
    if include_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_function_call(logger: logging.Logger, func_name: str, **kwargs):
    """
    Log function call with parameters.

    Args:
        logger: Logger instance
        func_name: Name of the function being called
        **kwargs: Function parameters to log
    """
    params_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
    logger.debug(f"Calling {func_name}({params_str})")


def log_function_result(logger: logging.Logger, func_name: str, result: any):
    """
    Log function result.

    Args:
        logger: Logger instance
        func_name: Name of the function
        result: Function result to log
    """
    logger.debug(f"{func_name} returned: {result}")
