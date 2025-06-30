#!/usr/bin/env python3
"""
Explore available ECMWF datasets to find ensemble forecast data.
"""

import cdsapi
import os

def explore_ecmwf_datasets():
    """Explore available ECMWF datasets."""
    
    print("Exploring available ECMWF datasets...")
    
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
        
        # Try different dataset names that might contain ensemble forecasts
        datasets_to_try = [
            'reanalysis-era5-single-levels',
            'seasonal-monthly-single-levels',
            'seasonal-monthly-pressure-levels',
            'reanalysis-era5-pressure-levels',
            'c3s-seasonal-original-single-levels',
            'c3s-seasonal-original-pressure-levels'
        ]
        
        print(f"\nTesting different dataset names:")
        
        for dataset in datasets_to_try:
            print(f"\nTrying dataset: {dataset}")
            try:
                # Try a minimal request to see if the dataset exists
                c.retrieve(
                    dataset,
                    {
                        'variable': 'mean_sea_level_pressure',
                        'year': '2020',
                        'month': '11',
                        'day': '01',
                        'time': '00:00',
                        'area': [20.0, -85.0, 10.0, -75.0],
                        'format': 'netcdf',
                    },
                    f'test_{dataset.replace("-", "_")}.nc'
                )
                print(f"  ✓ Dataset {dataset} is accessible")
                
                # Try to get dataset info
                print(f"  Checking dataset parameters...")
                
            except Exception as e:
                print(f"  ✗ Dataset {dataset} failed: {str(e)[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_seasonal_forecasts():
    """Test seasonal forecasts which might have ensemble data."""
    
    print(f"\n=== Testing Seasonal Forecasts ===")
    
    try:
        c = cdsapi.Client()
        
        print("Testing seasonal forecast dataset...")
        c.retrieve(
            'seasonal-monthly-single-levels',
            {
                'originating_centre': 'ecmwf',
                'system': '5',
                'variable': 'mean_sea_level_pressure',
                'year': '2020',
                'month': '11',
                'day': '01',
                'leadtime_month': '1',
                'area': [20.0, -85.0, 10.0, -75.0],
                'format': 'netcdf',
            },
            'test_seasonal_forecast.nc'
        )
        
        print("✓ Seasonal forecast dataset is accessible")
        return True
        
    except Exception as e:
        print(f"✗ Seasonal forecast failed: {e}")
        return False

if __name__ == "__main__":
    explore_ecmwf_datasets()
    test_seasonal_forecasts() 