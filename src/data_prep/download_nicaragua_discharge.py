#!/usr/bin/env python3
"""
Download Nicaragua Discharge Data Script

This script downloads river network (COMID) geometries, return periods, and daily discharge data for Nicaragua
from the GEOGloWS API, and saves the result as a GeoJSON file for use in flood modeling.

Usage:
    python download_nicaragua_discharge.py [config_path]
    # Example: python download_nicaragua_discharge.py config/flood_config.yaml

Output:
    nicaragua_discharge_<target_date>.geojson
"""
import requests
import pandas as pd
import math
from datetime import date
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pyproj import Transformer
import geopandas as gpd
from shapely.geometry import LineString
import os
import json
import time
import random
from requests.exceptions import RequestException
import sys
from src.utils.config_utils import load_config

# Defaults in case config is missing
DEFAULT_CONFIG_PATH = "config/flood_config.yaml"


def main(config_path=DEFAULT_CONFIG_PATH):
    config = load_config(config_path)
    geoglows_cfg = config["geoglows"]
    flood_cfg = config["flood"]
    api_root = geoglows_cfg["api_root"]
    layer_url = geoglows_cfg["layer_url"]
    where_clause = geoglows_cfg["country_where_clause"]
    target_date = geoglows_cfg["target_date"]
    cache_dir = geoglows_cfg["cache_dir"]
    max_workers = geoglows_cfg["max_workers"]
    min_workers = geoglows_cfg["min_workers"]
    chunk_size = geoglows_cfg["chunk_size"]
    os.makedirs(cache_dir, exist_ok=True)
    # Use cache files in cache_dir
    geometry_cache = os.path.join(cache_dir, "comid_geometry_cache.json")
    rp_cache = os.path.join(cache_dir, "return_periods_cache.json")

    def create_session():
        session = requests.Session()
        session.headers.update({"User-Agent": "geoglows-comid-fetcher/1.0"})
        return session

    import threading

    _thread_local = threading.local()

    def get_session():
        if not hasattr(_thread_local, "session"):
            _thread_local.session = create_session()
        return _thread_local.session

    def safe_get_discharge_json(url, retries=5, base_delay=0.1, max_delay=5):
        session = get_session()
        for attempt in range(retries):
            try:
                if attempt > 0:
                    jitter = random.uniform(0, base_delay)
                    time.sleep(jitter)
                r = session.get(url, timeout=30)
                if r.status_code == 200:
                    return r.json()
                elif r.status_code == 500:
                    if attempt < retries - 1:
                        wait_time = min(base_delay * (2**attempt), max_delay)
                        time.sleep(wait_time)
                        continue
                    else:
                        return None
                else:
                    r.raise_for_status()
            except RequestException as e:
                if attempt < retries - 1:
                    wait_time = min(base_delay * (2**attempt), max_delay)
                    time.sleep(wait_time)
                else:
                    return None
        return None

    def fetch_comids_with_geometry(
        base_url,
        where_clause="1=1",
        batch_size=2000,
        cache_file=geometry_cache,
    ):
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                print(f"🔄 Loading COMID geometries from cache: {cache_file}")
                return json.load(f)
        transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        all_comids_with_geometry = {}
        offset = 0
        while True:
            params = {
                "where": where_clause,
                "outFields": "comid,streamorder",
                "returnGeometry": "true",
                "f": "json",
                "resultOffset": offset,
                "resultRecordCount": batch_size,
            }
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            features = data.get("features", [])
            if not features:
                break
            for feature in features:
                attrs = feature.get("attributes", {})
                geometry = feature.get("geometry", {})
                comid = attrs.get("comid")
                streamorder = attrs.get("streamorder")
                if comid is not None and "paths" in geometry:
                    raw_path = geometry["paths"][0]
                    latlon_path = [transformer.transform(x, y) for x, y in raw_path]
                    all_comids_with_geometry[comid] = {
                        "polyline": latlon_path,
                        "streamorder": streamorder,
                    }
            offset += batch_size
        with open(cache_file, "w") as f:
            json.dump(all_comids_with_geometry, f)
            print(f"✅ Saved COMID geometries to cache: {cache_file}")
        if all_comids_with_geometry:
            sample_comid = next(iter(all_comids_with_geometry))
            sample_coords = all_comids_with_geometry[sample_comid]["polyline"]
            lons = [coord[0] for coord in sample_coords]
            lats = [coord[1] for coord in sample_coords]
            print(
                f"🔍 Coordinate check - Lon range: {min(lons):.2f} to {max(lons):.2f}, Lat range: {min(lats):.2f} to {max(lats):.2f}"
            )
            print(f"   Nicaragua should be roughly: Lon -87 to -83, Lat 10 to 15")
        return all_comids_with_geometry

    def fetch_return_periods(comid_list, cache_file=rp_cache, max_workers=max_workers):
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                print(f"🔄 Loading return periods from cache: {cache_file}")
                return json.load(f)

        def fetch_single_rp(comid):
            session = get_session()
            url = f"{api_root}/returnperiods/{comid}?format=json&bias_corrected=true"
            try:
                r = session.get(url, timeout=30)
                r.raise_for_status()
                js = r.json()
                if "return_periods" in js:
                    return comid, js["return_periods"]
            except:
                pass
            return comid, None

        rp_dict = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(fetch_single_rp, comid): comid for comid in comid_list
            }
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Fetching return periods",
            ):
                comid, data = fut.result()
                if data:
                    rp_dict[comid] = data
        with open(cache_file, "w") as f:
            json.dump(rp_dict, f)
            print(f"✅ Saved return periods to cache: {cache_file}")
        return rp_dict

    def get_retrospective_day_robust(river_id: int, day: str, rp_data: dict):
        try:
            retro_url = f"{api_root}/retrospectivedaily/{river_id}?format=json&start_date={day}&end_date={day}&bias_corrected=true"
            js = safe_get_discharge_json(retro_url)
            if not js:
                return None
            date_str = js["datetime"][0]
            discharge = js[str(river_id)][0]
            unit = js["metadata"]["units"]["long"]
            return_periods = rp_data.get(str(river_id)) or rp_data.get(int(river_id))
            return_period = 0
            if return_periods:
                for rp, threshold in sorted(return_periods.items(), key=lambda x: x[1]):
                    if discharge >= threshold:
                        return_period = int(rp)
            return {
                "comid": river_id,
                "date": date_str,
                "discharge_m3s": discharge,
                "unit": unit,
                "exceeds_rp": return_period,
            }
        except Exception as e:
            return None

    def fetch_all_discharge_adaptive(
        comid_list,
        target_date,
        rp_data,
        initial_workers=max_workers,
        min_workers=min_workers,
    ):
        failed_comids = []
        successful_results = []
        current_workers = initial_workers

        def process_batch(comids, workers):
            results = []
            failures = []
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        get_retrospective_day_robust, comid, target_date, rp_data
                    ): comid
                    for comid in comids
                }
                for fut in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Discharge ({workers} workers)",
                ):
                    result = fut.result()
                    original_comid = futures[fut]
                    if result:
                        results.append(result)
                    else:
                        failures.append(original_comid)
            return results, failures

        print(
            f"📈 First pass: trying {len(comid_list)} COMIDs with {current_workers} workers..."
        )
        results, failures = process_batch(comid_list, current_workers)
        successful_results.extend(results)
        retry_count = 1
        max_retries = 5
        while failures and retry_count <= max_retries:
            current_workers = max(min_workers, current_workers // 2)
            print(
                f"🔄 Retry {retry_count}: {len(failures)} failed COMIDs with {current_workers} workers..."
            )
            time.sleep(2)
            retry_results, new_failures = process_batch(failures, current_workers)
            successful_results.extend(retry_results)
            failures = new_failures
            retry_count += 1
        if failures:
            print(
                f"⚠️ Final status: {len(failures)} COMIDs still failed after {max_retries} retries"
            )
            print(
                f"Failed COMIDs: {failures[:10]}{'...' if len(failures) > 10 else ''}"
            )
        success_rate = len(successful_results) / len(comid_list) * 100
        print(
            f"✅ Overall success rate: {success_rate:.1f}% ({len(successful_results)}/{len(comid_list)})"
        )
        return successful_results

    def fetch_all_discharge_chunked(
        comid_list, target_date, rp_data, chunk_size=chunk_size, max_workers=max_workers
    ):
        all_results = []
        chunks = [
            comid_list[i : i + chunk_size]
            for i in range(0, len(comid_list), chunk_size)
        ]
        print(
            f"📊 Processing {len(comid_list)} COMIDs in {len(chunks)} chunks of {chunk_size}"
        )
        for chunk_num, chunk in enumerate(chunks, 1):
            print(f"\n🔄 Chunk {chunk_num}/{len(chunks)} ({len(chunk)} COMIDs)...")
            chunk_results = fetch_all_discharge_adaptive(
                chunk,
                target_date,
                rp_data,
                initial_workers=max_workers,
                min_workers=min_workers,
            )
            all_results.extend(chunk_results)
            if chunk_num < len(chunks):
                time.sleep(1)
        return all_results

    print("🔍 Fetching geometries...")
    geometry_dict = fetch_comids_with_geometry(layer_url, where_clause=where_clause)
    comid_list = list(geometry_dict.keys())
    print(f"Found {len(comid_list)} COMIDs")
    print(f"📊 Fetching return periods for {len(comid_list)} COMIDs...")
    rp_data = fetch_return_periods(comid_list, max_workers=max_workers)
    if len(comid_list) <= 1000:
        print(f"📈 Using adaptive approach for {len(comid_list)} COMIDs...")
        discharge_data = fetch_all_discharge_adaptive(comid_list, target_date, rp_data)
    else:
        print(f"📈 Using chunked approach for {len(comid_list)} COMIDs...")
        discharge_data = fetch_all_discharge_chunked(
            comid_list,
            target_date,
            rp_data,
            chunk_size=chunk_size,
            max_workers=max_workers,
        )
    combined_records = []
    for item in discharge_data:
        comid = item["comid"]
        if comid in geometry_dict:
            line = geometry_dict[comid]["polyline"]
            streamorder = geometry_dict[comid]["streamorder"]
            geom = LineString(line)
            record = {**item, "geometry": geom, "streamorder": streamorder}
            combined_records.append(record)
    if combined_records:
        gdf = gpd.GeoDataFrame(combined_records, geometry="geometry", crs="EPSG:4326")
        out_file = flood_cfg["discharge_file"]
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        gdf.to_file(out_file, driver="GeoJSON")
        final_success_rate = len(combined_records) / len(comid_list) * 100
        print(f"\n🎉 FINAL RESULTS:")
        print(f"✅ Saved {len(gdf)} features to {out_file}")
        print(
            f"📊 Final success rate: {final_success_rate:.1f}% ({len(combined_records)}/{len(comid_list)})"
        )
    else:
        print("❌ No data retrieved successfully")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH
    main(config_path)
