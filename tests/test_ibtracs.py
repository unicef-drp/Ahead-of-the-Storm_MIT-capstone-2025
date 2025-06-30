#!/usr/bin/env python3
"""
Test script to access IBTrACS hurricane track data.
IBTrACS contains official hurricane track positions from National Hurricane Center.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import requests
import io

def test_ibtracs_access():
    """Test accessing IBTrACS data for Eta and Iota hurricanes."""
    
    print("Testing IBTrACS hurricane track data access...")
    
    # IBTrACS data URL (North Atlantic basin)
    ibtracs_url = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv/ibtracs.NA.list.v04r00.csv"
    
    try:
        print(f"Downloading IBTrACS data from: {ibtracs_url}")
        
        # Download the data
        response = requests.get(ibtracs_url)
        response.raise_for_status()
        
        # Read the CSV data, skipping the first row which contains headers
        df = pd.read_csv(io.StringIO(response.text), skiprows=1)
        
        print(f"✓ Successfully downloaded IBTrACS data")
        print(f"  Total records: {len(df)}")
        print(f"  Available columns: {list(df.columns)}")
        
        # Check the first few rows to understand the data structure
        print(f"\nFirst few rows:")
        print(df.head())
        
        # Convert Year to numeric, handling any non-numeric values
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        
        # Filter for Eta and Iota (2020 season)
        # Based on the data structure, we need to look for the hurricane names in the data
        eta_iota = df[(df['Year'] == 2020) & 
                     (df[' .4'].isin(['ETA', 'IOTA']))].copy()
        
        print(f"\nFound {len(eta_iota)} records for Eta and Iota:")
        if len(eta_iota) > 0:
            # Use working pressure columns (mb.1 and mb.2 have actual data)
            print(eta_iota[[' .4', 'degrees_north', 'degrees_east', 'kts', 'mb.1', 'mb.2']].head(10))
            
            # Show track points for each hurricane
            for hurricane in ['ETA', 'IOTA']:
                hurricane_data = eta_iota[eta_iota[' .4'] == hurricane]
                if len(hurricane_data) > 0:
                    print(f"\n{hurricane} Track Summary:")
                    print(f"  Track points: {len(hurricane_data)}")
                    print(f"  Lat range: {hurricane_data['degrees_north'].min():.1f}°N to {hurricane_data['degrees_north'].max():.1f}°N")
                    print(f"  Lon range: {hurricane_data['degrees_east'].min():.1f}°E to {hurricane_data['degrees_east'].max():.1f}°E")
                    print(f"  Max wind speed: {hurricane_data['kts'].max()} knots")
                    
                    # Check pressure data from working columns
                    mb1_data = pd.to_numeric(hurricane_data['mb.1'], errors='coerce').dropna()
                    mb2_data = pd.to_numeric(hurricane_data['mb.2'], errors='coerce').dropna()
                    
                    if len(mb1_data) > 0:
                        print(f"  Pressure (mb.1): {mb1_data.min():.0f} - {mb1_data.max():.0f} hPa")
                    if len(mb2_data) > 0:
                        print(f"  Pressure (mb.2): {mb2_data.min():.0f} - {mb2_data.max():.0f} hPa")
                    
                    # Show sample track points with pressure data
                    print(f"  Sample track points with pressure data:")
                    sample_data = hurricane_data[['degrees_north', 'degrees_east', 'kts', 'mb.1', 'mb.2']].head(5)
                    print(sample_data.to_string(index=False))
        else:
            print("No records found for Eta and Iota in 2020")
            # Let's check what hurricanes are available in 2020
            hurricanes_2020 = df[df['Year'] == 2020][' .4'].unique()
            print(f"Hurricanes in 2020: {hurricanes_2020}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error accessing IBTrACS data: {e}")
        return False

if __name__ == "__main__":
    test_ibtracs_access() 