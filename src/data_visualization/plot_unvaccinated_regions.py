import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import os
import unicodedata

# Path to Nicaragua regions GeoJSON (update if needed)
REGIONS_GEOJSON = "data/raw/gadm/nicaragua_departments.geojson"
OUTPUT_DIR = "data/results/impact_analysis/vaccination_maps"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hardcoded vaccination data from the table (region names must match the GeoJSON)
vaccination_data = pd.DataFrame(
    {
        "Region": [
            "Nueva Segovia",
            "Jinotega", 
            "Madriz",
            "Estelí",
            "Chinandega",
            "León",
            "Matagalpa",
            "Boaco",
            "Managua",
            "Masaya",
            "Chontales",
            "Granada",
            "Carazo",
            "Rivas",
            "Río San Juan",
            "RACCN",
            "RACCS",
        ],
        "Vaccination_Rate": [
            95.5,
            89.6,
            97.7,
            96.9,
            82.2,
            87.7,
            89.0,
            91.9,
            74.2,
            81.3,
            98.0,
            87.2,
            93.8,
            96.7,
            89.3,
            68.5,
            82.6,
        ],
    }
)

# Calculate unvaccinated rates
vaccination_data["Unvaccinated_Rate"] = 100.0 - vaccination_data["Vaccination_Rate"]

# Load regions GeoJSON
regions_gdf = gpd.read_file(REGIONS_GEOJSON)


def normalize_vaccination_name(name):
    # Lowercase, remove accents, strip spaces
    if not isinstance(name, str):
        return name
    name = name.lower().strip()
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    # Special mappings for vaccination data
    mapping = {
        "raccn": "atlantico norte",
        "raccs": "atlantico sur",
        "estelí": "esteli",
        "león": "leon",
        "río san juan": "rio san juan",
    }
    # Map special cases
    if name in mapping:
        return mapping[name]
    return name


# Normalize region names in both GeoDataFrame and DataFrame
regions_gdf["region_norm"] = regions_gdf["NAME_1"].apply(normalize_vaccination_name)
vaccination_data["region_norm"] = vaccination_data["Region"].apply(normalize_vaccination_name)

# Exclude 'lago nicaragua' from regions_gdf
regions_gdf = regions_gdf[regions_gdf["region_norm"] != "lago nicaragua"]

# Print normalized region names for debugging
print("Normalized GeoJSON region names:", sorted(regions_gdf["region_norm"].unique()))
print(
    "Normalized Vaccination DataFrame region names:",
    sorted(vaccination_data["region_norm"].unique()),
)

# Find missing matches after normalization
geojson_names = set(regions_gdf["region_norm"].unique())
df_names = set(vaccination_data["region_norm"].unique())
missing_in_df = geojson_names - df_names
missing_in_geojson = df_names - geojson_names
print("Regions in GeoJSON but not in DataFrame (normalized):", missing_in_df)
print("Regions in DataFrame but not in GeoJSON (normalized):", missing_in_geojson)

# Merge on normalized names
regions_gdf = regions_gdf.merge(
    vaccination_data, left_on="region_norm", right_on="region_norm", how="left"
)

# Print summary statistics
print(f"\nVaccination Rate Summary:")
print(f"Mean vaccination rate: {regions_gdf['Vaccination_Rate'].mean():.1f}%")
print(f"Min vaccination rate: {regions_gdf['Vaccination_Rate'].min():.1f}%")
print(f"Max vaccination rate: {regions_gdf['Vaccination_Rate'].max():.1f}%")

print(f"\nUnvaccinated Rate Summary:")
print(f"Mean unvaccinated rate: {regions_gdf['Unvaccinated_Rate'].mean():.1f}%")
print(f"Min unvaccinated rate: {regions_gdf['Unvaccinated_Rate'].min():.1f}%")
print(f"Max unvaccinated rate: {regions_gdf['Unvaccinated_Rate'].max():.1f}%")

# Plot Vaccination Rate
fig, ax = plt.subplots(figsize=(12, 10))
regions_gdf.plot(
    ax=ax,
    column="Vaccination_Rate",
    cmap="Greens",
    linewidth=0.8,
    edgecolor="black",
    legend=True,
    legend_kwds={"label": "Vaccination Rate (%)"},
)
ax.set_title("Nicaragua - Vaccination Rate by Region (18-29 months)")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "nicaragua_vaccination_rate.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

# Plot Unvaccinated Rate
fig, ax = plt.subplots(figsize=(12, 10))
regions_gdf.plot(
    ax=ax,
    column="Unvaccinated_Rate",
    cmap="Reds",
    linewidth=0.8,
    edgecolor="black",
    legend=True,
    legend_kwds={"label": "Unvaccinated Rate (%)"},
)
ax.set_title("Nicaragua - Unvaccinated Rate by Region (18-29 months)")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "nicaragua_unvaccinated_rate.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

# Create a detailed table with region statistics
print(f"\nDetailed Region Statistics:")
print("-" * 80)
print(f"{'Region':<20} {'Vaccination %':<15} {'Unvaccinated %':<15}")
print("-" * 80)
for _, row in regions_gdf.sort_values('Unvaccinated_Rate', ascending=False).iterrows():
    print(f"{row['NAME_1']:<20} {row['Vaccination_Rate']:<15.1f} {row['Unvaccinated_Rate']:<15.1f}")

print(f"\nSaved vaccination maps to {OUTPUT_DIR}") 