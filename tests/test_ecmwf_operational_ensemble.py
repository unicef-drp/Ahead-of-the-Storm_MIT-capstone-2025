#!/usr/bin/env python3
"""
Test script to access ECMWF operational ensemble forecasts using Web API.
This will test access to ensemble forecast data for hurricane predictions.
"""

import ecmwfapi
import os
from datetime import datetime, timedelta

def setup_ecmwf_api():
    """Set up ECMWF Web API configuration."""
    
    print("Setting up ECMWF Web API configuration...")
    
    # Create .ecmwfapirc file with the provided credentials
    ecmwfapirc_content = """{
    "url"   : "https://api.ecmwf.int/v1",
    "key"   : "b33ee70d55beeb0e00d45ddf7a1a010d",
    "email" : "sidvijay@mit.edu"
}"""
    
    with open(os.path.expanduser('~/.ecmwfapirc'), 'w') as f:
        f.write(ecmwfapirc_content)
    
    print("✓ ECMWF API configuration file created")

def test_operational_ensemble_access():
    """Test accessing ECMWF operational ensemble forecast data."""
    
    print("\nTesting ECMWF operational ensemble forecast access...")
    
    try:
        # Initialize ECMWF API client
        print("Initializing ECMWF API client...")
        client = ecmwfapi.ECMWFDataServer()
        print("✓ ECMWF API client initialized successfully")
        
        # Test with a minimal ensemble forecast request
        print("\nTesting operational ensemble forecast data access...")
        
        # Define parameters for ensemble forecast request
        # Focus on Hurricane Eta period (November 2020)
        request_params = {
            'class': 'od',  # Operational data
            'dataset': 'enfo',  # Ensemble forecast
            'expver': '1',  # Experiment version
            'stream': 'enfo',  # Ensemble forecast stream
            'type': 'pf',  # Perturbed forecast (ensemble members)
            'levtype': 'sfc',  # Surface level
            'param': '151.128',  # Mean sea level pressure
            'date': '2020-11-01/to/2020-11-02',  # Date range
            'time': '00:00:00/12:00:00',  # Analysis times
            'step': '0/6/12/18/24',  # Forecast lead times
            'area': [20.0, -85.0, 10.0, -75.0],  # Region around Nicaragua
            'grid': '0.5/0.5',  # Grid resolution
            'format': 'netcdf',
            'target': 'test_operational_ensemble.nc'
        }
        
        print("Requesting operational ensemble forecast data...")
        print(f"Parameters: {request_params}")
        
        # Make the request
        client.retrieve(request_params)
        
        print("✓ Successfully accessed operational ensemble forecast data")
        return True
        
    except Exception as e:
        print(f"✗ Error accessing operational ensemble forecast data: {e}")
        return False

def test_ensemble_members():
    """Test accessing specific ensemble members."""
    
    print("\n=== Testing Ensemble Members ===")
    
    try:
        client = ecmwfapi.ECMWFDataServer()
        
        # Test accessing control forecast (member 0)
        print("Testing control forecast (member 0)...")
        
        control_params = {
            'class': 'od',
            'dataset': 'enfo',
            'expver': '1',
            'stream': 'enfo',
            'type': 'cf',  # Control forecast
            'levtype': 'sfc',
            'param': '151.128',  # Mean sea level pressure
            'date': '2020-11-01',
            'time': '00:00:00',
            'step': '24',  # 24-hour forecast
            'area': [20.0, -85.0, 10.0, -75.0],
            'grid': '0.5/0.5',
            'format': 'netcdf',
            'target': 'test_control_forecast.nc'
        }
        
        client.retrieve(control_params)
        print("✓ Control forecast accessed successfully")
        
        # Test accessing perturbed forecasts (ensemble members 1-50)
        print("Testing perturbed forecasts (ensemble members)...")
        
        perturbed_params = {
            'class': 'od',
            'dataset': 'enfo',
            'expver': '1',
            'stream': 'enfo',
            'type': 'pf',  # Perturbed forecast
            'levtype': 'sfc',
            'param': '151.128',
            'date': '2020-11-01',
            'time': '00:00:00',
            'step': '24',
            'number': '1/2/3/4/5',  # First 5 ensemble members
            'area': [20.0, -85.0, 10.0, -75.0],
            'grid': '0.5/0.5',
            'format': 'netcdf',
            'target': 'test_perturbed_forecasts.nc'
        }
        
        client.retrieve(perturbed_params)
        print("✓ Perturbed forecasts accessed successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Error accessing ensemble members: {e}")
        return False

def test_hurricane_variables():
    """Test accessing hurricane-relevant variables."""
    
    print("\n=== Testing Hurricane-Relevant Variables ===")
    
    try:
        client = ecmwfapi.ECMWFDataServer()
        
        # Test multiple variables relevant for hurricane analysis
        hurricane_vars = {
            '151.128': 'Mean sea level pressure',
            '165.128': '10 metre U wind component',
            '166.128': '10 metre V wind component',
            '228.128': 'Total precipitation',
            '134.128': 'Surface pressure'
        }
        
        for param_id, description in hurricane_vars.items():
            print(f"Testing {description} (param: {param_id})...")
            
            var_params = {
                'class': 'od',
                'dataset': 'enfo',
                'expver': '1',
                'stream': 'enfo',
                'type': 'cf',  # Control forecast
                'levtype': 'sfc',
                'param': param_id,
                'date': '2020-11-01',
                'time': '00:00:00',
                'step': '24',
                'area': [20.0, -85.0, 10.0, -75.0],
                'grid': '0.5/0.5',
                'format': 'netcdf',
                'target': f'test_{param_id.replace(".", "_")}.nc'
            }
            
            client.retrieve(var_params)
            print(f"  ✓ {description} accessed successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Error accessing hurricane variables: {e}")
        return False

if __name__ == "__main__":
    setup_ecmwf_api()
    test_operational_ensemble_access()
    test_ensemble_members()
    test_hurricane_variables() 