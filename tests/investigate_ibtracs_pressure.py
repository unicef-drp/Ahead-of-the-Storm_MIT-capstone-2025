#!/usr/bin/env python3
"""
Investigate IBTrACS pressure data issue and find alternative sources.
"""

import pandas as pd
import numpy as np
import requests
import io

def investigate_pressure_data():
    """Investigate why pressure data is missing and find alternatives."""
    
    print("Investigating IBTrACS pressure data issue...")
    
    # IBTrACS data URL
    ibtracs_url = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv/ibtracs.NA.list.v04r00.csv"
    
    try:
        # Download and read data
        response = requests.get(ibtracs_url)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text), skiprows=1)
        
        # Convert Year to numeric
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        
        # Filter for Eta and Iota
        eta_iota = df[(df['Year'] == 2020) & 
                     (df[' .4'].isin(['ETA', 'IOTA']))].copy()
        
        print(f"\n=== IBTrACS Pressure Data Analysis ===")
        print(f"Total records for Eta and Iota: {len(eta_iota)}")
        
        # Check pressure data availability
        pressure_cols = [col for col in df.columns if 'mb' in col.lower()]
        print(f"\nPressure-related columns found: {pressure_cols}")
        
        # Check each pressure column for Eta and Iota
        for col in pressure_cols:
            non_null = eta_iota[col].notna().sum()
            total = len(eta_iota)
            print(f"  {col}: {non_null}/{total} records ({non_null/total*100:.1f}%)")
        
        # Check if there are other agencies with pressure data
        print(f"\n=== Checking Other Agencies ===")
        
        # Look for other pressure columns from different agencies
        for hurricane in ['ETA', 'IOTA']:
            hurricane_data = eta_iota[eta_iota[' .4'] == hurricane]
            print(f"\n{hurricane} - Checking all pressure columns:")
            
            for col in pressure_cols:
                non_null = hurricane_data[col].notna().sum()
                if non_null > 0:
                    print(f"  {col}: {non_null} records with data")
                    print(f"    Range: {hurricane_data[col].min()} - {hurricane_data[col].max()}")
        
        # Check if we can get pressure from ERA5 at IBTrACS track points
        print(f"\n=== Alternative: ERA5 Pressure at Track Points ===")
        print("We can use ERA5 reanalysis to get pressure data at the exact")
        print("IBTrACS track positions and times.")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def check_alternative_sources():
    """Check alternative sources for hurricane pressure data."""
    
    print(f"\n=== Alternative Data Sources ===")
    
    alternatives = [
        {
            "name": "ERA5 Reanalysis",
            "description": "Get pressure at IBTrACS track points",
            "pros": "Complete pressure fields, high resolution",
            "cons": "Model-based, not direct observations"
        },
        {
            "name": "HURDAT2 (NOAA)",
            "description": "NOAA's hurricane database",
            "pros": "Official NOAA data, includes pressure",
            "cons": "Same as IBTrACS (they share data)"
        },
        {
            "name": "NHC Best Track",
            "description": "National Hurricane Center official tracks",
            "pros": "Most authoritative source",
            "cons": "Same data as IBTrACS"
        }
    ]
    
    for alt in alternatives:
        print(f"\n{alt['name']}:")
        print(f"  {alt['description']}")
        print(f"  Pros: {alt['pros']}")
        print(f"  Cons: {alt['cons']}")

if __name__ == "__main__":
    investigate_pressure_data()
    check_alternative_sources() 