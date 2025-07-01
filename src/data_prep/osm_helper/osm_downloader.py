"""
Data downloader for OpenStreetMap data using Overpass API.

This module provides functionality to download various types of geographic
data from OpenStreetMap for specific regions.
"""

import yaml
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Point, LineString

from .overpass_client import OverpassClient

logger = logging.getLogger(__name__)


class OSMDataDownloader:
    """Downloader for OpenStreetMap data using Overpass API."""

    def __init__(self, config_path: str = "config/data_config.yaml"):
        """
        Initialize the data downloader.

        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.client = OverpassClient(
            base_url=self.config["overpass"]["base_url"],
            timeout=self.config["overpass"]["timeout"],
            max_retries=self.config["overpass"].get("max_retries", 3),
            retry_delay=self.config["overpass"].get("retry_delay", 1.0),
        )
        self.output_dir = Path(self.config["output"]["raw_data_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _build_query(
        self, category: str, tags: List[str], exclude_tags: Optional[List[str]] = None
    ) -> str:
        """
        Build an Overpass QL query for a specific category.

        Args:
            category: Name of the data category
            tags: List of tags to search for
            exclude_tags: List of tags to exclude

        Returns:
            Overpass QL query string
        """
        country_iso = self.config["country"]["iso_code"]
        admin_level = self.config["country"]["admin_level"]

        # Get category-specific timeout
        category_config = self.config["data_categories"][category]
        timeout = category_config.get("timeout", self.config["overpass"]["timeout"])
        query_type = category_config.get("query_type", "standard")
        output_format = category_config.get("output_format", "center")

        # Simple ways query (for roads)
        if query_type == "simple_ways":
            query = f"""
            [out:json][timeout:{timeout}];
            area["ISO3166-1"="{country_iso}"][admin_level={admin_level}];
            way(area)["highway"];
            out {output_format};
            """
            return query

        # Standard query for other categories
        # Build tag filters
        tag_filters = []
        for tag in tags:
            if "=" in tag:
                key, value = tag.split("=", 1)
                tag_filters.append(f'["{key}"="{value}"]')
            else:
                tag_filters.append(f'["{tag}"]')

        # Build exclude filters
        exclude_filters = []
        if exclude_tags:
            for tag in exclude_tags:
                if "=" in tag:
                    key, value = tag.split("=", 1)
                    exclude_filters.append(f'["{key}"!="{value}"]')

        # Combine all filters
        all_filters = tag_filters + exclude_filters

        query = f"""
        [out:json][timeout:{timeout}];
        area["ISO3166-1"="{country_iso}"][admin_level={admin_level}]->.searchArea;
        (
        """

        # Add queries for nodes, ways, and relations
        for filter_str in all_filters:
            query += f"  node(area.searchArea){filter_str};\n"
            query += f"  way(area.searchArea){filter_str};\n"
            query += f"  relation(area.searchArea){filter_str};\n"

        query += f"""
        );
        out {output_format};
        """

        return query

    def _extract_geometries(
        self, response: Dict[str, Any], category: Optional[str] = None
    ) -> List[Any]:
        """
        Extract geometries from Overpass API response.

        Args:
            response: JSON response from Overpass API
            category: Category name for special handling

        Returns:
            List of Shapely geometry objects
        """
        geometries = []
        elements = response.get("elements", [])

        if category is None:
            return geometries

        category_config = self.config["data_categories"][category]
        query_type = category_config.get("query_type", "standard")
        processing_config = self.config.get("processing", {})
        min_coordinates = processing_config.get("min_coordinates", 2)

        # Simple ways processing (for roads)
        if query_type == "simple_ways":
            exclude_highways = category_config.get("exclude_highways", [])

            for element in elements:
                if element["type"] != "way":
                    continue

                tags = element.get("tags", {})
                hwy = tags.get("highway")

                # Skip if it's in the exclude list or missing
                if hwy is None or hwy in exclude_highways:
                    continue

                # Extract coordinates from geometry
                if "geometry" in element:
                    coords = [(pt["lon"], pt["lat"]) for pt in element["geometry"]]
                    if len(coords) >= min_coordinates:
                        from shapely.geometry import LineString

                        geom = LineString(coords)
                        geometries.append(geom)

            return geometries

        # Standard processing for other categories
        for element in elements:
            geom = None

            if element["type"] == "node":
                # Point geometry
                lon = element.get("lon")
                lat = element.get("lat")
                if lon is not None and lat is not None:
                    geom = Point(lon, lat)

            elif element["type"] == "way":
                # LineString or Polygon geometry
                if "center" in element:
                    # Use center point for ways
                    lon = element["center"]["lon"]
                    lat = element["center"]["lat"]
                    geom = Point(lon, lat)

            elif element["type"] == "relation":
                # Use center point for relations
                if "center" in element:
                    lon = element["center"]["lon"]
                    lat = element["center"]["lat"]
                    geom = Point(lon, lat)

            if geom is not None:
                geometries.append(geom)

        return geometries

    def download_category(self, category: str) -> Optional[gpd.GeoDataFrame]:
        """
        Download data for a specific category.

        Args:
            category: Name of the category to download

        Returns:
            GeoDataFrame with the downloaded data or None if failed
        """
        if not self.config["data_categories"][category]["enabled"]:
            logger.info(f"Category {category} is disabled, skipping")
            return None

        logger.info(f"Downloading data for category: {category}")

        category_config = self.config["data_categories"][category]
        tags = category_config["tags"]
        exclude_tags = category_config.get("exclude_tags", [])

        # Use category-specific timeout if available
        timeout = category_config.get("timeout", self.config["overpass"]["timeout"])
        self.client.timeout = timeout

        query = self._build_query(category, tags, exclude_tags)
        response = self.client.query(query)

        if response is None:
            logger.error(f"Failed to download data for category: {category}")
            return None

        geometries = self._extract_geometries(response, category)

        if not geometries:
            logger.warning(f"No geometries found for category: {category}")
            return None

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(geometry=geometries, crs=self.config["output"]["crs"])

        # Add configurable columns
        processing_config = self.config.get("processing", {})

        if processing_config.get("add_category_column", True):
            gdf["category"] = category

        if processing_config.get("add_id_column", True):
            gdf["id"] = range(len(gdf))

        logger.info(f"Downloaded {len(gdf)} features for category: {category}")
        return gdf

    def save_data(self, gdf: gpd.GeoDataFrame, category: str) -> str:
        """
        Save GeoDataFrame to file.

        Args:
            gdf: GeoDataFrame to save
            category: Category name for filename

        Returns:
            Path to the saved file
        """
        file_format = self.config["output"]["file_format"]
        filename = f"{category}.{file_format}"
        filepath = self.output_dir / filename

        if file_format == "geojson":
            gdf.to_file(filepath, driver="GeoJSON")
        elif file_format == "shp":
            gdf.to_file(filepath, driver="ESRI Shapefile")
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        logger.info(f"Saved {category} data to: {filepath}")
        return str(filepath)

    def download_all(self) -> Dict[str, str]:
        """
        Download all enabled categories.

        Returns:
            Dictionary mapping category names to file paths
        """
        results = {}

        for category in self.config["data_categories"]:
            try:
                gdf = self.download_category(category)
                if gdf is not None:
                    filepath = self.save_data(gdf, category)
                    results[category] = filepath
                else:
                    logger.warning(f"Failed to download category: {category}")
            except Exception as e:
                logger.error(f"Error downloading category {category}: {e}")

        return results
