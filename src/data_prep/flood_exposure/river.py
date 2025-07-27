import numpy as np
import pandas as pd
from rasterio.features import rasterize
from scipy.ndimage import binary_dilation

# --- RIVER NETWORK CREATION FOR FLOOD MODELING ---


def create_river_network_raster(segments, shape, transform, crs, river_width_m=None):
    print("[Flood] Creating river network raster...")
    try:
        segments_reproj = segments.to_crs(crs)

        def stream_order_to_width(order):
            # Map stream order to river width in meters
            if order <= 1:
                return 5
            elif order == 2:
                return 10
            elif order == 3:
                return 20
            elif order == 4:
                return 40
            elif order == 5:
                return 60
            elif order == 6:
                return 100
            elif order == 7:
                return 150
            elif order == 8:
                return 250
            else:
                return 60

        buffered_segments = []
        for idx, row in segments_reproj.iterrows():
            try:
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue
                # Buffer by stream order if available, else use fixed width
                if "streamorder" in row and not pd.isna(row["streamorder"]):
                    width = stream_order_to_width(int(row["streamorder"]))
                elif river_width_m is not None:
                    width = river_width_m
                else:
                    width = 60
                radius = width / 2.0
                buffered_geom = geom.buffer(radius)
                if buffered_geom.is_empty:
                    buffered_geom = geom.centroid.buffer(radius)
                buffered_segments.append(buffered_geom)
            except Exception:
                continue
        if not buffered_segments:
            print("[Flood] No valid river segments found. Returning empty raster.")
            return np.zeros(shape, dtype=np.uint8)
        # Rasterize buffered river polygons
        river_raster = rasterize(
            buffered_segments,
            out_shape=shape,
            transform=transform,
            fill=0,
            default_value=1,
            dtype=np.uint8,
        )
        # Enhance connectivity with binary dilation
        river_raster = binary_dilation(river_raster, iterations=1)
        print(
            f"[Flood] River network raster created. River pixels: {np.sum(river_raster)}"
        )
        return river_raster
    except Exception:
        print(
            "[Flood][ERROR] Failed to create river network raster. Returning empty raster."
        )
        return np.zeros(shape, dtype=np.uint8)
