#!/usr/bin/env python3
"""
Test script to access ECMWF operational ensemble forecast data.
This will test access to ensemble forecast tracks for hurricane predictions.
"""

import cdsapi
import os

def test_ecmwf_ensemble():
    """Test accessing ECMWF ensemble forecast data for hurricane predictions."""
    
    print("Testing ECMWF operational ensemble forecast access...")
    
    # Set up the .cdsapirc file with the working API token
    cdsapirc_content = """url: https://cds.climate.copernicus.eu/api
key: 702eb373-f029-4c31-acd5-b2d268f43cb4"""
    
    with open(os.path.expanduser('~/.cdsapirc'), 'w') as f:
        f.write(cdsapirc_content)
    
    try:
        # Initialize CDS API client
        print("Initializing CDS API client...")
        c = cdsapi.Client()
        print("✓ CDS API client initialized successfully")
        
        # Test with a minimal ensemble forecast request
        print("\nTesting TIGGE ensemble forecast data access...")
        
        # Try to access TIGGE ensemble forecast data for a small region and time period
        c.retrieve(
            'tigge',
            {
                'dataset': 'tigge',
                'step_type': 'instant',
                'originating_centre': 'ecmf',
                'system': '1',
                'variable': 'msl',
                'date': '2020-11-01',
                'time': '00:00:00',
                'step': '0',
                'area': [20.0, -85.0, 10.0, -75.0],  # Smaller region around Nicaragua
                'format': 'netcdf',
            },
            'test_tigge_ensemble.nc'
        )
        
        print("✓ Successfully accessed TIGGE ensemble forecast data")
        return True
        
    except Exception as e:
        print(f"✗ Error accessing ECMWF ensemble forecast data: {e}")
        return False

if __name__ == "__main__":
    test_ecmwf_ensemble() 