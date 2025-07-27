import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show as rioshow
from rasterio.mask import mask as rio_mask
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import os
from shapely.geometry import mapping
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Paths
DISCHARGE_PATH = "data/preprocessed/flood/nicaragua_discharge_20201117.geojson"
FLOOD_EXTENT_PATH = "data/preprocessed/flood/nicaragua_flood_extent_20201117.tif"
BOUNDARY_PATH = "data/raw/gadm41_NIC_shp/gadm41_NIC_0.shp"
TRUE_FLOOD_PATH = "data/raw/true_flood_iota/TC20201116NIC_SHP/VIIRS_20201113_20201117_FloodExtent_Nicaragua.shp"
CLOUD_OBSTRUCTION_PATH = "data/raw/true_flood_iota/TC20201116NIC_SHP/VIIRS_20201113_20201117_CloudObstruction_Nicaragua.shp"

# Load boundary
boundary = gpd.read_file(BOUNDARY_PATH)

# Load discharge rivers
rivers = gpd.read_file(DISCHARGE_PATH)

# Load flood extent raster
flood_src = rasterio.open(FLOOD_EXTENT_PATH)
flood_data = flood_src.read(1)
flood_data = np.ma.masked_where(flood_data == flood_src.nodata, flood_data)

# Load true flood shapefile
true_flood = gpd.read_file(TRUE_FLOOD_PATH)

# Load cloud obstruction shapefile
cloud_obstruction = gpd.read_file(CLOUD_OBSTRUCTION_PATH)

# Reproject all to raster CRS
raster_crs = flood_src.crs
if boundary.crs != raster_crs:
    boundary = boundary.to_crs(raster_crs)
if rivers.crs != raster_crs:
    rivers = rivers.to_crs(raster_crs)
if true_flood.crs != raster_crs:
    true_flood = true_flood.to_crs(raster_crs)
if cloud_obstruction.crs != raster_crs:
    cloud_obstruction = cloud_obstruction.to_crs(raster_crs)

# Clip rivers to Nicaragua boundary
rivers = gpd.clip(rivers, boundary)

# Clip flood raster to Nicaragua boundary
shapes = [mapping(geom) for geom in boundary.geometry]
flood_clipped, flood_transform = rio_mask(flood_src, shapes, crop=True, filled=True)
flood_clipped = flood_clipped[0]
flood_clipped = np.ma.masked_where(flood_clipped == flood_src.nodata, flood_clipped)

# Enhance flood visibility: mask very low values
flood_visible = np.ma.masked_less(flood_clipped, 0.1)

# Clip true flood to Nicaragua boundary
true_flood = gpd.clip(true_flood, boundary)
cloud_obstruction = gpd.clip(cloud_obstruction, boundary)

# Discrete color mapping for exceeds_rp
rp_values = [0, 2, 5, 10, 25]
colors = ["gray", "green", "yellow", "orange", "red"]
cmap = ListedColormap(colors)
boundaries = [0, 1, 3.5, 7.5, 17.5, 30]  # bins: 0, 2, 5, 10, 25
norm = BoundaryNorm(boundaries, ncolors=len(colors))

# Plot side by side
fig, axes = plt.subplots(1, 2, figsize=(18, 9))

# Left: Predicted flood
ax = axes[0]
rioshow(flood_visible, transform=flood_transform, ax=ax, cmap="Blues_r", alpha=0.9)
rivers.plot(
    ax=ax,
    column="exceeds_rp",
    cmap=cmap,
    linewidth=0.7,
    legend=False,
    norm=norm,
)
boundary.boundary.plot(ax=ax, color="black", linewidth=1.5)
legend_elements = [
    Line2D([0], [0], color=colors[i], lw=2, label=f"RP {rp_values[i]}")
    for i in range(len(rp_values))
]
ax.legend(handles=legend_elements, title="Return Period Exceeded")
ax.set_title("Predicted Flood Extent (Model)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

# Right: Actual flood
ax2 = axes[1]
true_flood.plot(ax=ax2, color="blue", alpha=0.7, edgecolor="k", linewidth=0.5)
cloud_obstruction.plot(ax=ax2, color="orange", alpha=0.5, edgecolor="k", linewidth=0.5)
boundary.boundary.plot(ax=ax2, color="black", linewidth=1.5)
legend_elements2 = [
    Patch(facecolor="blue", edgecolor="k", label="Observed Flood"),
    Patch(facecolor="orange", edgecolor="k", label="Cloud Obstruction"),
]
ax2.legend(handles=legend_elements2, title="Observed Layers")
ax2.set_title("Observed Flood Extent (VIIRS)")
ax2.set_xlabel("Longitude")
ax2.set_ylabel("Latitude")

# Save the figure instead of showing it
output_dir = "data/results/flood"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "predicted_vs_true_flood.png")
plt.tight_layout()
plt.savefig(output_path, dpi=300)
print(f"Saved flood comparison map to {output_path}")
