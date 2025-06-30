import rasterio
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from .osm_viz import get_nicaragua_boundary


def plot_census_raster(
    filepath,
    title=None,
    ax=None,
    cmap="viridis",
    figsize=(10, 8),
    vmin=None,
    vmax=None,
    show_boundary=True,
    boundary_color="red",
    boundary_linewidth=2,
):
    """
    Plot a single census raster (GeoTIFF) file.
    Args:
        filepath: Path to the GeoTIFF file
        title: Optional plot title
        ax: Optional matplotlib axis
        cmap: Colormap
        figsize: Figure size
        vmin, vmax: Value range for display
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

    with rasterio.open(filepath) as src:
        data = src.read(1)
        # Mask zeros for better visualization
        masked = np.ma.masked_where(data == 0, data)
        im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label="Population")

        # Plot Nicaragua boundary if requested
        if show_boundary:
            boundary = get_nicaragua_boundary()
            if boundary is not None:
                # Transform boundary to raster CRS if needed
                if boundary.crs != src.crs:
                    boundary = boundary.to_crs(src.crs)
                boundary.plot(
                    ax=ax,
                    color="none",
                    edgecolor=boundary_color,
                    linewidth=boundary_linewidth,
                    alpha=0.9,
                )

        if title:
            ax.set_title(title)
        ax.set_axis_off()
    return ax


def plot_multiple_census_rasters(
    filepaths,
    titles=None,
    cmap="viridis",
    figsize=(15, 10),
    vmin=None,
    vmax=None,
    show_boundary=True,
    boundary_color="red",
    boundary_linewidth=2,
):
    """
    Plot multiple census rasters as subplots.
    Args:
        filepaths: List of GeoTIFF file paths
        titles: List of titles for each subplot
        cmap: Colormap
        figsize: Figure size
        vmin, vmax: Value range for display
        show_boundary: Whether to show Nicaragua boundary
        boundary_color: Color for boundary
        boundary_linewidth: Line width for boundary
    Returns:
        fig, axes: The matplotlib figure and axes
    """
    n = len(filepaths)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flat if hasattr(axes, "flat") else axes
    for i, filepath in enumerate(filepaths):
        title = titles[i] if titles else None
        plot_census_raster(
            filepath,
            title=title,
            ax=axes[i],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            show_boundary=show_boundary,
            boundary_color=boundary_color,
            boundary_linewidth=boundary_linewidth,
        )
    plt.tight_layout()
    return fig, axes
