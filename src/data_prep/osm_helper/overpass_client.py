"""
Overpass API client for downloading OpenStreetMap data.

This module provides a client for querying the Overpass API to download
various types of geographic data from OpenStreetMap.
"""

import time
import requests
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class OverpassClient:
    """Client for interacting with the Overpass API."""

    def __init__(
        self,
        base_url: str = "https://overpass-api.de/api/interpreter",
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize the Overpass client.

        Args:
            base_url: The Overpass API endpoint URL
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session = requests.Session()

    def query(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Execute a query against the Overpass API.

        Args:
            query: The Overpass QL query string

        Returns:
            JSON response from the API or None if failed
        """
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Executing Overpass query (attempt {attempt + 1})")
                response = self.session.post(
                    self.base_url, data={"data": query}, timeout=self.timeout
                )
                response.raise_for_status()

                result = response.json()
                logger.info(
                    f"Query successful, returned {len(result.get('elements', []))} elements"
                )
                return result

            except requests.exceptions.RequestException as e:
                logger.warning(f"Query attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2**attempt))  # Exponential backoff
                else:
                    logger.error(f"All {self.max_retries} query attempts failed")
                    return None
