#!/usr/bin/env python3
"""
Research script for Google DeepMind's WeatherLab ensemble model.
Focusing on hurricane forecasting capabilities and data access.
"""

import requests
import json
import os
from datetime import datetime, timedelta
import pandas as pd

def research_weatherlab():
    """Research Google DeepMind's WeatherLab ensemble model."""
    
    print("=" * 80)
    print("GOOGLE DEEPMIND WEATHERLAB RESEARCH")
    print("=" * 80)
    
    print("\n1. WEATHERLAB OVERVIEW:")
    print("   ")
    print("   A. Model Information:")
    print("   - Google DeepMind's experimental weather forecasting")
    print("   - Ensemble forecasting capabilities")
    print("   - Available from 2023 to present")
    print("   - Multiple ensemble members")
    print("   ")
    print("   B. Hurricane Forecasting:")
    print("   - Cyclone track predictions")
    print("   - Ensemble mean forecasts")
    print("   - Paired track analysis")
    print("   - Multiple lead times")
    print("   ")
    print("   C. Data Access:")
    print("   - Scriptable URLs available")
    print("   - CSV format downloads")
    print("   - Real-time and historic data")
    print("   - Different license terms based on age")
    
    print("\n2. URL STRUCTURE ANALYSIS:")
    print("   ")
    print("   Base URL: https://deepmind.google.com/science/weatherlab/download/")
    print("   ")
    print("   Components:")
    print("   - cyclones/: Hurricane/cyclone data")
    print("   - FNV3/: Model version")
    print("   - ensemble_mean/: Ensemble mean forecast")
    print("   - paired/: Paired track analysis")
    print("   - csv/: CSV format")
    print("   - FNV3_YYYY_MM_DDTHH_MM_paired.csv: Filename")
    
    print("\n3. HURRICANE RAFAEL (2024) INFORMATION:")
    print("   ")
    print("   Hurricane Rafael Details:")
    print("   - Formed: November 2024")
    print("   - Category: Tropical Storm/Hurricane")
    print("   - Location: Atlantic Ocean")
    print("   - Did not make landfall in Nicaragua")
    print("   - Good candidate for track modification")
    print("   ")
    print("   Available Data:")
    print("   - Ensemble forecasts from formation")
    print("   - Multiple forecast dates")
    print("   - Track evolution over time")
    print("   - Intensity predictions")
    
    print("\n4. DATA FORMAT EXPECTATIONS:")
    print("   ")
    print("   CSV Structure:")
    print("   - Date/Time columns")
    print("   - Latitude/Longitude coordinates")
    print("   - Intensity metrics (pressure, wind speed)")
    print("   - Ensemble member information")
    print("   - Forecast lead times")
    print("   - Track confidence intervals")
    
    print("\n5. TRACK MODIFICATION STRATEGY:")
    print("   ")
    print("   For Nicaragua Landfall Simulation:")
    print("   1. Download original Rafael tracks")
    print("   2. Identify track segments near Central America")
    print("   3. Modify longitude/latitude to hit Nicaragua")
    print("   4. Adjust intensity based on land interaction")
    print("   5. Validate modified tracks")
    
    print("\n6. LICENSE CONSIDERATIONS:")
    print("   ")
    print("   Real-Time Data (< 48 hours):")
    print("   - GDM Real-Time Weather Forecasting Terms")
    print("   - More restrictive usage")
    print("   ")
    print("   Historic Data (> 48 hours):")
    print("   - Creative Commons Attribution 4.0")
    print("   - More permissive for research")
    print("   - Rafael 2024 data should be historic")

def test_weatherlab_url():
    """Test WeatherLab URL structure and data access."""
    
    print("\n" + "=" * 50)
    print("TESTING WEATHERLAB URL ACCESS")
    print("=" * 50)
    
    # Test URL from the provided example
    test_url = "https://deepmind.google.com/science/weatherlab/download/cyclones/FNV3/ensemble_mean/paired/csv/FNV3_2024_11_05T12_00_paired.csv"
    
    print(f"Testing URL: {test_url}")
    
    try:
        response = requests.get(test_url, timeout=30)
        
        if response.status_code == 200:
            print("✓ URL accessible")
            print(f"Content length: {len(response.content)} bytes")
            print(f"Content type: {response.headers.get('content-type', 'unknown')}")
            
            # Try to read as CSV
            try:
                df = pd.read_csv(pd.StringIO(response.text))
                print(f"✓ CSV readable, shape: {df.shape}")
                print(f"Columns: {list(df.columns)}")
                print("\nFirst few rows:")
                print(df.head())
                
                return df
                
            except Exception as e:
                print(f"✗ CSV parsing error: {e}")
                print("Raw content preview:")
                print(response.text[:500])
                
        else:
            print(f"✗ HTTP error: {response.status_code}")
            print(f"Response: {response.text[:200]}")
            
    except Exception as e:
        print(f"✗ Request error: {e}")
    
    return None

def generate_rafael_urls():
    """Generate URLs for Hurricane Rafael data."""
    
    print("\n" + "=" * 50)
    print("GENERATING RAFAEL DATA URLS")
    print("=" * 50)
    
    # Hurricane Rafael likely dates (November 2024)
    rafael_dates = [
        "2024-11-05",  # Example from provided URL
        "2024-11-06",
        "2024-11-07", 
        "2024-11-08",
        "2024-11-09",
        "2024-11-10"
    ]
    
    times = ["00:00", "06:00", "12:00", "18:00"]
    
    urls = []
    
    for date in rafael_dates:
        for time in times:
            # Format: FNV3_YYYY_MM_DDTHH_MM_paired.csv
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            time_obj = datetime.strptime(time, "%H:%M")
            
            filename = f"FNV3_{date_obj.strftime('%Y_%m_%d')}T{time_obj.strftime('%H_%M')}_paired.csv"
            url = f"https://deepmind.google.com/science/weatherlab/download/cyclones/FNV3/ensemble_mean/paired/csv/{filename}"
            
            urls.append({
                'date': date,
                'time': time,
                'url': url,
                'filename': filename
            })
    
    print(f"Generated {len(urls)} URLs for Rafael data:")
    for i, url_info in enumerate(urls[:5]):  # Show first 5
        print(f"  {i+1}. {url_info['date']} {url_info['time']}: {url_info['filename']}")
    
    if len(urls) > 5:
        print(f"  ... and {len(urls) - 5} more")
    
    return urls

def test_multiple_urls(urls, max_tests=3):
    """Test multiple URLs to find accessible data."""
    
    print("\n" + "=" * 50)
    print("TESTING MULTIPLE URLS")
    print("=" * 50)
    
    accessible_urls = []
    
    for i, url_info in enumerate(urls[:max_tests]):
        print(f"\nTesting {i+1}/{min(max_tests, len(urls))}: {url_info['date']} {url_info['time']}")
        
        try:
            response = requests.get(url_info['url'], timeout=30)
            
            if response.status_code == 200:
                print(f"✓ Accessible: {len(response.content)} bytes")
                accessible_urls.append(url_info)
                
                # Quick CSV check
                try:
                    df = pd.read_csv(pd.StringIO(response.text))
                    print(f"  CSV shape: {df.shape}, Columns: {len(df.columns)}")
                except:
                    print("  CSV parsing failed")
            else:
                print(f"✗ HTTP {response.status_code}")
                
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print(f"\nFound {len(accessible_urls)} accessible URLs")
    return accessible_urls

if __name__ == "__main__":
    research_weatherlab()
    test_weatherlab_url()
    urls = generate_rafael_urls()
    accessible_urls = test_multiple_urls(urls) 