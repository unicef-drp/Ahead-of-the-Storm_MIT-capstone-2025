"""
Universal plotting utilities for impact analysis layers.

This module provides functions to plot exposure, vulnerability, and impact layers
with both linear and log scales, maintaining consistent styling and metadata handling.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

from src.utils.config_utils import load_config, get_config_value
from src.utils.hurricane_geom import get_nicaragua_boundary


def plot_layer_with_scales(
    layer, 
    output_dir: str = "data/results/impact_analysis/",
    scales: List[str] = None,
    config_path: str = "config/plotting_config.yaml"
) -> List[str]:
    """
    Universal plotting function that handles all layer types and scales.
    
    Args:
        layer: Any exposure, vulnerability, or impact layer
        output_dir: Output directory for plots
        scales: List of scales to generate ("linear", "log", or both)
        config_path: Path to plotting configuration file
    
    Returns:
        List of generated plot file paths
    """
    # Load plotting configuration
    config = load_config(config_path)
    plotting_config = config.get("plotting", {})
    
    # Default to both scales if not specified
    if scales is None:
        scales = plotting_config.get("default_scales", ["linear", "log"])
    
    # Get layer metadata
    metadata = get_layer_metadata(layer, plotting_config)
    
    # Get plot data
    data_column, data_values = layer.get_plot_data()
    
    # Generate plots for each scale
    generated_files = []
    
    for scale in scales:
        if scale not in ["linear", "log"]:
            continue
            
        # Apply scale transformation
        plot_values, scale_metadata = apply_scale_transformation(
            data_values, scale, metadata, plotting_config
        )
        
        # Generate filename
        filename = generate_filename(layer, metadata, scale, plotting_config)
        
        # Create plot
        filepath = create_plot(
            layer, 
            plot_values, 
            scale_metadata, 
            output_dir, 
            filename, 
            plotting_config
        )
        
        if filepath:
            generated_files.append(filepath)
    
    return generated_files


def get_layer_metadata(layer, plotting_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract metadata from any layer for plotting.
    
    Args:
        layer: Any exposure, vulnerability, or impact layer
        plotting_config: Plotting configuration dictionary
    
    Returns:
        Dictionary containing layer metadata
    """
    # Get base metadata from layer
    if hasattr(layer, 'get_plot_metadata'):
        metadata = layer.get_plot_metadata()
    else:
        # Fallback metadata extraction
        metadata = extract_fallback_metadata(layer, plotting_config)
    
    return metadata


def extract_fallback_metadata(layer, plotting_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract metadata when layer doesn't implement get_plot_metadata().
    
    Args:
        layer: Any layer object
        plotting_config: Plotting configuration dictionary
    
    Returns:
        Dictionary containing fallback metadata
    """
    layer_class = layer.__class__.__name__
    
    # Determine layer type
    if "Exposure" in layer_class:
        layer_type = "exposure"
        hazard_types = plotting_config.get("hazard_types", {})
        hazard_type = hazard_types.get(layer_class, "Unknown")
    elif "Vulnerability" in layer_class:
        layer_type = "vulnerability"
        vulnerability_types = plotting_config.get("vulnerability_types", {})
        vulnerability_type = vulnerability_types.get(layer_class, "Unknown")
    elif "Impact" in layer_class:
        layer_type = "impact"
        hazard_type = "Unknown"
        vulnerability_type = "Unknown"
    else:
        layer_type = "unknown"
        hazard_type = "Unknown"
        vulnerability_type = "Unknown"
    
    # Get colormap
    colormaps = plotting_config.get("colormaps", {})
    layer_colormaps = colormaps.get(layer_type, {})
    
    if layer_type == "exposure":
        colormap = layer_colormaps.get(hazard_type.lower(), "viridis")
    elif layer_type == "vulnerability":
        colormap = layer_colormaps.get(vulnerability_type.lower().replace(" ", "_"), "viridis")
    else:
        colormap = layer_colormaps.get("default", "viridis")
    
    return {
        "layer_type": layer_type,
        "hazard_type": hazard_type,
        "vulnerability_type": vulnerability_type,
        "colormap": colormap,
        "title_template": plotting_config.get("titles", {}).get(layer_type, "Unknown Layer"),
        "legend_template": plotting_config.get("legend_labels", {}).get("linear", {}).get(layer_type, "Unknown"),
        "filename_template": plotting_config.get("filenames", {}).get(layer_type, "unknown_layer"),
        "special_features": []
    }


def apply_scale_transformation(
    data_values: np.ndarray, 
    scale: str, 
    metadata: Dict[str, Any],
    plotting_config: Dict[str, Any]
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply scale transformation to data values.
    
    Args:
        data_values: Raw data values
        scale: Scale type ("linear" or "log")
        metadata: Layer metadata
        plotting_config: Plotting configuration
    
    Returns:
        Tuple of (transformed_values, updated_metadata)
    """
    if scale == "linear":
        transformed_values = data_values
        legend_label = plotting_config.get("legend_labels", {}).get("linear", {}).get(
            metadata["layer_type"], "Value per Cell"
        )
    elif scale == "log":
        # Apply log10 transformation with +1 offset to handle zeros
        transformed_values = np.log10(data_values + 1)
        legend_label = plotting_config.get("legend_labels", {}).get("log", {}).get(
            metadata["layer_type"], "Log10(Value + 1) per Cell"
        )
    else:
        transformed_values = data_values
        legend_label = "Value per Cell"
    
    # Update metadata with scale-specific information
    scale_metadata = metadata.copy()
    scale_metadata["scale"] = scale
    scale_metadata["legend_label"] = legend_label.format(
        hazard_type=metadata.get("hazard_type", ""),
        vulnerability_type=metadata.get("vulnerability_type", "")
    )
    
    return transformed_values, scale_metadata


def generate_filename(
    layer, 
    metadata: Dict[str, Any], 
    scale: str, 
    plotting_config: Dict[str, Any]
) -> str:
    """
    Generate appropriate filename for the plot.
    
    Args:
        layer: Layer object
        metadata: Layer metadata
        scale: Scale type
        plotting_config: Plotting configuration
    
    Returns:
        Generated filename
    """
    # Get base filename template
    filename_template = metadata.get("filename_template", "unknown_layer")
    
    # Get parameters for filename
    parameters = get_filename_parameters(layer, metadata, plotting_config)
    
    # Apply template
    base_filename = filename_template.format(
        hazard_type=metadata.get("hazard_type", "").lower(),
        vulnerability_type=metadata.get("vulnerability_type", "").lower().replace(" ", "_"),
        parameters=parameters
    )
    
    # Add scale suffix
    scale_suffixes = plotting_config.get("scale_suffixes", {})
    scale_suffix = scale_suffixes.get(scale, f"_{scale}")
    
    return f"{base_filename}{scale_suffix}.png"


def get_filename_parameters(layer, metadata: Dict[str, Any], plotting_config: Dict[str, Any]) -> str:
    """
    Get parameters for filename generation.
    
    Args:
        layer: Layer object
        metadata: Layer metadata
        plotting_config: Plotting configuration
    
    Returns:
        Parameter string for filename
    """
    parameters = []
    
    # Add forecast time for hurricane layers
    if hasattr(layer, 'chosen_forecast') and layer.chosen_forecast:
        date_str = str(layer.chosen_forecast).replace(":", "-").replace(" ", "_")
        parameters.append(date_str)
    
    # Add vulnerability parameters
    if hasattr(layer, 'vulnerability_layer'):
        vuln_layer = layer.vulnerability_layer
        if hasattr(vuln_layer, 'age_groups') and hasattr(vuln_layer, 'gender'):
            age_str = "_".join(map(str, vuln_layer.age_groups))
            parameters.append(f"{vuln_layer.gender}_ages_{age_str}")
    
    # Add resampling method for landslide
    if hasattr(layer, 'resampling_method'):
        parameters.append(layer.resampling_method)
    
    # Add ensemble suffix for landslide
    if metadata.get("hazard_type") == "Landslide":
        parameters.append("ensemble")
    
    return "_".join(parameters) if parameters else ""


def create_plot(
    layer,
    plot_values: np.ndarray,
    metadata: Dict[str, Any],
    output_dir: str,
    filename: str,
    plotting_config: Dict[str, Any]
) -> Optional[str]:
    """
    Create the actual plot.
    
    Args:
        layer: Layer object
        plot_values: Values to plot
        metadata: Layer metadata
        output_dir: Output directory
        filename: Output filename
        plotting_config: Plotting configuration
    
    Returns:
        Path to generated plot file, or None if failed
    """
    try:
        # Get grid data - try different methods
        grid_gdf = None
        if hasattr(layer, 'get_visualization_grid'):
            grid_gdf = layer.get_visualization_grid()
        elif hasattr(layer, 'compute_grid'):
            grid_gdf = layer.compute_grid()
        elif hasattr(layer, 'compute_impact'):
            grid_gdf = layer.compute_impact()
        else:
            # Create a simple grid for testing
            import geopandas as gpd
            from shapely.geometry import box
            
            # Create a simple grid
            grid_cells = [box(0, 0, 1, 1), box(1, 0, 2, 1), box(0, 1, 1, 2), box(1, 1, 2, 2)]
            grid_gdf = gpd.GeoDataFrame(grid_cells, columns=["geometry"], crs="EPSG:4326")
        
        if grid_gdf is None:
            return None
        
        # Add plot values to grid
        data_column = metadata.get("data_column", "value")
        grid_gdf[data_column] = plot_values
        
        # Create figure
        fig, ax = plt.subplots(figsize=plotting_config.get("figure_size", [12, 10]))
        
        # Plot the data
        grid_gdf.plot(
            ax=ax,
            column=data_column,
            cmap=metadata["colormap"],
            linewidth=plotting_config.get("linewidth", 0.1),
            edgecolor=plotting_config.get("edgecolor", "grey"),
            alpha=plotting_config.get("alpha", 0.7),
            legend=True,
            legend_kwds={"label": metadata["legend_label"]},
        )
        
        # Add Nicaragua boundary
        nicaragua_gdf = get_nicaragua_boundary()
        if nicaragua_gdf is not None:
            nicaragua_gdf.plot(
                ax=ax,
                color="none",
                edgecolor=plotting_config.get("boundary_color", "black"),
                linewidth=plotting_config.get("boundary_linewidth", 3),
                alpha=1.0
            )
        
        # Apply special features
        apply_special_features(ax, layer, metadata, plotting_config)
        
        # Set title
        title = generate_title(layer, metadata, plotting_config)
        ax.set_title(title)
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(output_dir, filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=plotting_config.get("dpi", 300), bbox_inches="tight")
        plt.close(fig)
        
        return output_path
        
    except Exception as e:
        print(f"Error creating plot: {e}")
        return None


def apply_special_features(ax, layer, metadata: Dict[str, Any], plotting_config: Dict[str, Any]):
    """
    Apply special features based on layer type.
    
    Args:
        ax: Matplotlib axes
        layer: Layer object
        metadata: Layer metadata
        plotting_config: Plotting configuration
    """
    special_features = plotting_config.get("special_features", {})
    
    # Set axis limits for flood layers
    if metadata.get("hazard_type") == "Flood" and special_features.get("flood", {}).get("set_axis_limits", False):
        nicaragua_gdf = get_nicaragua_boundary()
        if nicaragua_gdf is not None:
            minx, miny, maxx, maxy = nicaragua_gdf.total_bounds
            ax.set_xlim(minx, maxx)
            ax.set_ylim(miny, maxy)


def generate_title(layer, metadata: Dict[str, Any], plotting_config: Dict[str, Any]) -> str:
    """
    Generate plot title.
    
    Args:
        layer: Layer object
        metadata: Layer metadata
        plotting_config: Plotting configuration
    
    Returns:
        Generated title
    """
    title_template = metadata.get("title_template", "Unknown Layer")
    
    # Get vulnerability type for impact layers
    vulnerability_type = metadata.get("vulnerability_type", "")
    if metadata.get("layer_type") == "impact" and hasattr(layer, 'vulnerability_layer'):
        vuln_layer = layer.vulnerability_layer
        
        # Handle children case
        if (hasattr(vuln_layer, 'age_groups') and 
            vuln_layer.age_groups == [0, 5, 10, 15]):
            vulnerability_type = plotting_config.get("children_vulnerability_type", "Children")
        
        # Handle population-weighted cases
        if hasattr(vuln_layer, 'weighted_by_population') and vuln_layer.weighted_by_population:
            population_weighted_types = plotting_config.get("population_weighted_vulnerability_types", {})
            if "health_facilities_population" in population_weighted_types:
                vulnerability_type = population_weighted_types["health_facilities_population"]
        
        if hasattr(vuln_layer, 'weighted_by_capacity') and vuln_layer.weighted_by_capacity:
            population_weighted_types = plotting_config.get("population_weighted_vulnerability_types", {})
            if "shelters_population" in population_weighted_types:
                vulnerability_type = population_weighted_types["shelters_population"]
    
    return title_template.format(
        hazard_type=metadata.get("hazard_type", ""),
        vulnerability_type=vulnerability_type
    ) 