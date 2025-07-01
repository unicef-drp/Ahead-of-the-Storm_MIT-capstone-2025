import geopandas as gpd
import matplotlib.pyplot as plt
import requests
from shapely.geometry import shape


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


def get_nicaragua_boundary():
    """
    Get Nicaragua administrative boundary using the actual polygon.
    Returns:
        GeoDataFrame with Nicaragua boundary
    """
    nic_poly = get_nicaragua_polygon()
    nicaragua_gdf = gpd.GeoDataFrame(geometry=[nic_poly], crs="EPSG:4326")
    return nicaragua_gdf


def plot_osm_geodata(
    gdf,
    title=None,
    ax=None,
    color="blue",
    marker="o",
    figsize=(10, 10),
    show_boundary=True,
    boundary_color="black",
    boundary_linewidth=1,
):
    """
    Plot OSM GeoDataFrame (points, lines, polygons) on a map.
    Args:
        gdf: GeoDataFrame to plot
        title: Optional plot title
        ax: Optional matplotlib axis
        color: Color for geometries
        marker: Marker for points
        figsize: Figure size
        show_boundary: Whether to show Nicaragua boundary
        boundary_color: Color for boundary
        boundary_linewidth: Line width for boundary
    Returns:
        ax: The matplotlib axis with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    # Ensure ax is a single axis, not an ndarray
    if not isinstance(ax, plt.Axes):
        ax = ax.flat[0]

    # Plot Nicaragua boundary first if requested
    if show_boundary:
        boundary = get_nicaragua_boundary()
        if boundary is not None:
            boundary.plot(
                ax=ax,
                color="none",
                edgecolor=boundary_color,
                linewidth=boundary_linewidth,
                alpha=0.8,
            )

    # Plot the main data
    gdf.plot(ax=ax, color=color, marker=marker, edgecolor="k", linewidth=0.7, alpha=0.7)
    if title:
        ax.set_title(title)
    ax.set_axis_off()
    return ax


def plot_multiple_osm_layers(
    layers,
    titles=None,
    colors=None,
    figsize=(15, 10),
    show_boundary=True,
    boundary_color="black",
    boundary_linewidth=1,
):
    """
    Plot multiple OSM GeoDataFrames as subplots.
    Args:
        layers: List of GeoDataFrames
        titles: List of titles for each subplot
        colors: List of colors for each layer
        figsize: Figure size
        show_boundary: Whether to show Nicaragua boundary
        boundary_color: Color for boundary
        boundary_linewidth: Line width for boundary
    Returns:
        fig, axes: The matplotlib figure and axes
    """
    n = len(layers)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    # Ensure axes is always a list of axes
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flat if hasattr(axes, "flat") else axes
    for i, gdf in enumerate(layers):
        color = colors[i] if colors else "blue"
        title = titles[i] if titles else None
        plot_osm_geodata(
            gdf,
            title=title,
            ax=axes[i],
            color=color,
            show_boundary=show_boundary,
            boundary_color=boundary_color,
            boundary_linewidth=boundary_linewidth,
        )
    plt.tight_layout()
    return fig, axes
