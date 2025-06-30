#!/usr/bin/env python3
"""
Test script to access ECMWF MARS for operational ensemble forecasts (ENS).
This is where the real hurricane ensemble track data is stored.
"""

import ecmwfapi
import os
from datetime import datetime, timedelta

def setup_ecmwf_api():
    """Set up ECMWF API configuration for MARS access."""
    
    print("Setting up ECMWF API configuration for MARS...")
    
    # Create .ecmwfapirc file with the provided credentials
    ecmwfapirc_content = """{
    "url"   : "https://api.ecmwf.int/v1",
    "key"   : "b33ee70d55beeb0e00d45ddf7a1a010d",
    "email" : "sidvijay@mit.edu"
}"""
    
    with open(os.path.expanduser('~/.ecmwfapirc'), 'w') as f:
        f.write(ecmwfapirc_content)
    
    print("✓ ECMWF API configuration file created")

def test_mars_ens_access():
    """Test accessing ECMWF MARS for operational ensemble forecasts (ENS)."""
    
    print("\nTesting ECMWF MARS ENS (Atmospheric Model Ensemble) access...")
    
    try:
        # Initialize ECMWF API client
        print("Initializing ECMWF API client for MARS...")
        client = ecmwfapi.ECMWFDataServer()
        print("✓ ECMWF API client initialized successfully")
        
        # Test MARS ENS (operational ensemble forecasts)
        print("\nTesting MARS ENS operational ensemble forecasts...")
        
        # MARS ENS parameters for operational ensemble forecasts
        mars_ens_params = {
            'class': 'od',  # Operational data
            'dataset': 'ens',  # Ensemble forecast dataset
            'expver': '1',  # Experiment version
            'stream': 'enfo',  # Ensemble forecast stream
            'type': 'pf',  # Perturbed forecast (ensemble members)
            'levtype': 'sfc',  # Surface level
            'param': '151.128',  # Mean sea level pressure
            'date': '2020-11-01',  # Hurricane Eta period
            'time': '00:00:00',  # Analysis time
            'step': '24',  # 24-hour forecast
            'number': '1/2/3/4/5',  # First 5 ensemble members
            'area': [20.0, -85.0, 10.0, -75.0],  # Region around Nicaragua
            'grid': '0.5/0.5',  # Grid resolution
            'format': 'netcdf',
            'target': 'test_mars_ens.nc'
        }
        
        print("Requesting MARS ENS operational ensemble forecast data...")
        print(f"Parameters: {mars_ens_params}")
        
        # Make the request
        client.retrieve(mars_ens_params)
        
        print("✓ Successfully accessed MARS ENS operational ensemble forecast data")
        return True
        
    except Exception as e:
        print(f"✗ Error accessing MARS ENS data: {e}")
        return False

def test_mars_ens_control():
    """Test accessing MARS ENS control forecast."""
    
    print("\n=== Testing MARS ENS Control Forecast ===")
    
    try:
        client = ecmwfapi.ECMWFDataServer()
        
        # Test control forecast (member 0)
        print("Testing MARS ENS control forecast...")
        
        control_params = {
            'class': 'od',
            'dataset': 'ens',
            'expver': '1',
            'stream': 'enfo',
            'type': 'cf',  # Control forecast
            'levtype': 'sfc',
            'param': '151.128',
            'date': '2020-11-01',
            'time': '00:00:00',
            'step': '24',
            'area': [20.0, -85.0, 10.0, -75.0],
            'grid': '0.5/0.5',
            'format': 'netcdf',
            'target': 'test_mars_ens_control.nc'
        }
        
        client.retrieve(control_params)
        print("✓ MARS ENS control forecast accessed successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Error accessing MARS ENS control: {e}")
        return False

def test_mars_ens_hurricane_variables():
    """Test accessing hurricane-relevant variables from MARS ENS."""
    
    print("\n=== Testing MARS ENS Hurricane Variables ===")
    
    try:
        client = ecmwfapi.ECMWFDataServer()
        
        # Hurricane-relevant variables for ensemble forecasts
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
                'dataset': 'ens',
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
                'target': f'test_mars_ens_{param_id.replace(".", "_")}.nc'
            }
            
            client.retrieve(var_params)
            print(f"  ✓ {description} accessed successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Error accessing MARS ENS hurricane variables: {e}")
        return False

def test_mars_ens_multiple_lead_times():
    """Test accessing multiple lead times for ensemble forecasts."""
    
    print("\n=== Testing MARS ENS Multiple Lead Times ===")
    
    try:
        client = ecmwfapi.ECMWFDataServer()
        
        # Test multiple lead times for ensemble forecasts
        lead_times = ['0', '6', '12', '18', '24', '48', '72']
        
        for step in lead_times:
            print(f"Testing lead time: {step} hours...")
            
            step_params = {
                'class': 'od',
                'dataset': 'ens',
                'expver': '1',
                'stream': 'enfo',
                'type': 'cf',
                'levtype': 'sfc',
                'param': '151.128',
                'date': '2020-11-01',
                'time': '00:00:00',
                'step': step,
                'area': [20.0, -85.0, 10.0, -75.0],
                'grid': '0.5/0.5',
                'format': 'netcdf',
                'target': f'test_mars_ens_step_{step}.nc'
            }
            
            client.retrieve(step_params)
            print(f"  ✓ Lead time {step}h accessed successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Error accessing MARS ENS lead times: {e}")
        return False

if __name__ == "__main__":
    setup_ecmwf_api()
    test_mars_ens_access()
    test_mars_ens_control()
    test_mars_ens_hurricane_variables()
    test_mars_ens_multiple_lead_times() 