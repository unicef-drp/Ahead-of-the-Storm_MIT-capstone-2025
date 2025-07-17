import geopandas as gpd
import os

# Input shapefile (update path if needed)
INPUT_SHP = "gadm41_NIC_shp/gadm41_NIC_1.shp"
# Output GeoJSON
OUTPUT_GEOJSON = "data/raw/gadm/nicaragua_departments.geojson"
os.makedirs(os.path.dirname(OUTPUT_GEOJSON), exist_ok=True)


def main():
    print(f"Loading shapefile: {INPUT_SHP}")
    gdf = gpd.read_file(INPUT_SHP)
    print(f"Loaded {len(gdf)} regions.")
    print(f"Saving to GeoJSON: {OUTPUT_GEOJSON}")
    gdf.to_file(OUTPUT_GEOJSON, driver="GeoJSON")
    print("Done.")


if __name__ == "__main__":
    main()
