import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import os
import unicodedata

# Path to Nicaragua regions GeoJSON (update if needed)
REGIONS_GEOJSON = "data/raw/gadm/nicaragua_departments.geojson"
COUNTRY_SHP = "gadm41_NIC_shp/gadm41_NIC_0.shp"
OUTPUT_DIR = "data/results/impact_analysis/poverty_maps"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hardcoded poverty data from the table (region names must match the GeoJSON)
poverty_data = pd.DataFrame(
    {
        "Region": [
            "Boaco",
            "Carazo",
            "Chinandega",
            "Chontales",
            "Esteli",
            "Granada",
            "Jinotega",
            "Leon",
            "Madriz",
            "Managua",
            "Masaya",
            "Matagalpa",
            "Nueva Segovia",
            "Raan",
            "Raas",
            "Rio San Juan",
            "Rivas",
        ],
        "H": [
            28.1,
            5.5,
            9.7,
            18.5,
            10.3,
            8.2,
            43.8,
            9.0,
            26.0,
            3.8,
            4.8,
            22.9,
            26.7,
            35.5,
            32.1,
            28.9,
            7.8,
        ],
        "Severe_Poverty": [
            10.3,
            0.1,
            1.4,
            4.9,
            1.2,
            2.2,
            19.5,
            2.1,
            6.7,
            1.3,
            0.9,
            7.4,
            9.6,
            15.2,
            12.1,
            9.7,
            1.0,
        ],
    }
)

# Load regions GeoJSON
regions_gdf = gpd.read_file(REGIONS_GEOJSON)
# Load country boundary
country_gdf = gpd.read_file(COUNTRY_SHP)


def normalize_name(name):
    # Lowercase, remove accents, strip spaces
    if not isinstance(name, str):
        return name
    name = name.lower().strip()
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    # Special mappings
    mapping = {
        "atlantico norte": "raan",
        "atlantico sur": "raas",
        "esteli": "esteli",
        "leon": "leon",
        "rio san juan": "rio san juan",
    }
    # Map special cases
    if name in mapping:
        return mapping[name]
    return name


# Normalize region names in both GeoDataFrame and DataFrame
regions_gdf["region_norm"] = regions_gdf["NAME_1"].apply(normalize_name)
poverty_data["region_norm"] = poverty_data["Region"].apply(normalize_name)

# Exclude 'lago nicaragua' from regions_gdf
regions_gdf = regions_gdf[regions_gdf["region_norm"] != "lago nicaragua"]

# Print normalized region names for debugging
print("Normalized GeoJSON region names:", sorted(regions_gdf["region_norm"].unique()))
print(
    "Normalized Poverty DataFrame region names:",
    sorted(poverty_data["region_norm"].unique()),
)

# Find missing matches after normalization
geojson_names = set(regions_gdf["region_norm"].unique())
df_names = set(poverty_data["region_norm"].unique())
missing_in_df = geojson_names - df_names
missing_in_geojson = df_names - geojson_names
print("Regions in GeoJSON but not in DataFrame (normalized):", missing_in_df)
print("Regions in DataFrame but not in GeoJSON (normalized):", missing_in_geojson)

# Merge on normalized names
regions_gdf = regions_gdf.merge(
    poverty_data, left_on="region_norm", right_on="region_norm", how="left"
)

# Plot H (Headcount Ratio)
fig, ax = plt.subplots(figsize=(12, 10))
regions_gdf.plot(
    ax=ax,
    column="H",
    cmap="OrRd",
    linewidth=0.8,
    edgecolor="black",
    legend=True,
    legend_kwds={"label": "Headcount Ratio (H) %"},
)
country_gdf.boundary.plot(ax=ax, color="black", linewidth=2.5)
ax.set_title("Nicaragua - Headcount Ratio (H) by Region")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "nicaragua_headcount_ratio_H.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

# Plot Severe Poverty
fig, ax = plt.subplots(figsize=(12, 10))
regions_gdf.plot(
    ax=ax,
    column="Severe_Poverty",
    cmap="Blues",
    linewidth=0.8,
    edgecolor="black",
    legend=True,
    legend_kwds={"label": "Severe Poverty %"},
)
country_gdf.boundary.plot(ax=ax, color="black", linewidth=2.5)
ax.set_title("Nicaragua - Severe Poverty by Region")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "nicaragua_severe_poverty.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

print(f"Saved poverty maps to {OUTPUT_DIR}")
