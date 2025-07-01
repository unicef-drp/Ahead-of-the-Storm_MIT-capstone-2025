import rasterio
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import requests
from shapely.geometry import shape
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from rasterio.plot import show
from rasterio.warp import transform_geom


def get_nicaragua_polygon():
    """
    Get Nicaragua polygon using the same method as the hurricane code.
    Returns:
        shapely.geometry.Polygon: Nicaragua boundary polygon
    """
    countries_url = "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"
    countries = requests.get(countries_url).json()

    nic_poly = None
    for feat in countries["features"]:
        props = feat.get("properties", {})
        if any(
            isinstance(v, str) and v.strip().lower() == "nicaragua"
            for v in props.values()
        ):
            nic_poly = shape(feat["geometry"])
            break

    if nic_poly is None:
        raise RuntimeError("Could not find Nicaragua in GeoJSON.")

    return nic_poly


def plot_child_population_heatmap(
    age_groups=[0, 1, 5],
    gender="both",
    figsize=(12, 8),
    log_scale=True,
    cmap="viridis",
    show_boundary=True,
):
    """
    Plot child population heatmap with proper Nicaragua boundary.

    Args:
        age_groups: List of age groups to plot (e.g., [0, 1, 5, 10, 15])
        gender: 'female', 'male', or 'both'
        figsize: Figure size
        log_scale: Whether to use log scale for better visualization
        cmap: Colormap for the heatmap
        show_boundary: Whether to show Nicaragua boundary

    Returns:
        fig, ax: The matplotlib figure and axis
    """
    base_path = Path("./data/raw/census")

    # Get Nicaragua boundary
    if show_boundary:
        nic_poly = get_nicaragua_polygon()

    # Determine which files to load based on gender selection
    files_to_load = []

    for age in age_groups:
        if gender == "female" or gender == "both":
            female_file = base_path / f"nic_f_{age}_2020_constrained_UNadj.tif"
            if female_file.exists():
                files_to_load.append(str(female_file))

        if gender == "male" or gender == "both":
            male_file = base_path / f"nic_m_{age}_2020_constrained_UNadj.tif"
            if male_file.exists():
                files_to_load.append(str(male_file))

    if not files_to_load:
        print("No population files found for the specified age groups and gender!")
        return None, None

    # Create single plot
    fig, ax = plt.subplots(figsize=figsize)

    # Load and combine all data
    combined_data = None
    first_src = None

    for filepath in files_to_load:
        with rasterio.open(filepath) as src:
            data = src.read(1)

            if first_src is None:
                first_src = src

            if combined_data is None:
                combined_data = data.copy()
            else:
                combined_data += data

    if combined_data is None:
        print("No valid data found!")
        return None, None

    # Apply log scale if requested
    if log_scale:
        combined_data = np.log10(combined_data + 1)

        # Use rasterio's show function which handles CRS properly
    show(combined_data, ax=ax, transform=first_src.transform, cmap=cmap)  # type: ignore

    # Get the mappable from the axes
    im = ax.get_images()[0] if ax.get_images() else None  # type: ignore

    # Add colorbar
    if im is not None:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        cbar = plt.colorbar(ax=ax, fraction=0.046, pad=0.04)
    if log_scale:
        cbar.set_label("Log10(Population + 1)", fontsize=12)
    else:
        cbar.set_label("Population", fontsize=12)

    # Add Nicaragua boundary if requested
    if show_boundary and nic_poly:
        # Convert Nicaragua polygon to the same CRS as the raster
        nic_poly_transformed = transform_geom(
            "EPSG:4326",  # Source CRS (geographic)
            first_src.crs,  # type: ignore # Target CRS (raster CRS)
            nic_poly.__geo_interface__,
        )

        # Convert back to shapely geometry
        nic_poly_raster_crs = shape(nic_poly_transformed)

        # Plot the transformed boundary
        if hasattr(nic_poly_raster_crs, "exterior"):
            # Single polygon
            coords = list(nic_poly_raster_crs.exterior.coords)  # type: ignore
            x_coords, y_coords = zip(*coords)
            ax.plot(  # type: ignore
                x_coords,
                y_coords,
                color="black",
                linewidth=2,
                label="Nicaragua Boundary",
            )
        else:
            # MultiPolygon - plot each polygon
            for poly in nic_poly_raster_crs.geoms:  # type: ignore
                coords = list(poly.exterior.coords)  # type: ignore
                x_coords, y_coords = zip(*coords)
                ax.plot(x_coords, y_coords, color="black", linewidth=2)  # type: ignore

    # Add title
    gender_text = {"female": "Female", "male": "Male", "both": "Female & Male"}[gender]
    age_text = ", ".join(map(str, age_groups))
    ax.set_title(  # type: ignore
        f"Child Population Heatmap: {gender_text} Ages {age_text}",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()
    return fig, ax
