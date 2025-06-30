#!/usr/bin/env python3
"""
Test script to access ECMWF ENS operational ensemble forecasts using ECMWFService("mars").
This uses the correct approach for accessing operational ensemble data.
"""

from ecmwfapi import ECMWFService
import xarray as xr
import os

def setup_ecmwf_api():
    """Set up ECMWF API configuration for MARS access."""
    
    print("Setting up ECMWF API configuration for MARS...")
    
    # Create .ecmwfapirc file with the provided credentials
    ecmwfapirc_content = """{
    "url": "https://api.ecmwf.int/v1",
    "key": "b33ee70d55beeb0e00d45ddf7a1a010d",
    "email": "sidvijay@mit.edu"
}"""
    
    with open(os.path.expanduser('~/.ecmwfapirc'), 'w') as f:
        f.write(ecmwfapirc_content)
    
    print("✓ ECMWF API configuration file created")

def test_mars_ens_access():
    """Test accessing ECMWF MARS for operational ensemble forecasts (ENS)."""
    
    print("\nTesting ECMWF MARS ENS (Atmospheric Model Ensemble) access...")
    
    try:
        # Initialize the service for MARS
        print("Initializing ECMWFService for MARS...")
        server = ECMWFService("mars")
        print("✓ ECMWFService initialized successfully")
        
        # Common parameters for ENS operational data
        common_ens_params = {
            "class": "od",       # Operational Data
            "stream": "oper",    # Operational stream (for real-time ENS)
            "date": "20201101",  # Hurricane Eta period (YYYYMMDD format)
            "time": "12",        # Forecast run time: "12" UTC
            "step": "24",        # 24-hour forecast
            "levtype": "sfc",    # Level type: surface
            "param": "msl",      # Mean Sea Level pressure
            "grid": "1.0/1.0",   # Grid resolution
            "format": "grib",    # Output format
        }
        
        # Test Ensemble Mean (EM)
        print("\nTesting Ensemble Mean (EM)...")
        ens_mean_request = common_ens_params.copy()
        ens_mean_request["type"] = "em"  # Type for Ensemble Mean
        target_ens_mean_file = "test_ens_mean_20201101_12.grib"
        
        print(f"Requesting ECMWF Ensemble Mean for {ens_mean_request['date']} {ens_mean_request['time']} UTC...")
        server.execute(ens_mean_request, target_ens_mean_file)
        print(f"✓ Successfully downloaded Ensemble Mean to {target_ens_mean_file}")
        
        # Load and inspect the data using xarray
        ds_em = xr.open_dataset(target_ens_mean_file, engine="cfgrib")
        print("\n--- Ensemble Mean Data (xarray Dataset) ---")
        print(ds_em)
        
        return True
        
    except Exception as e:
        print(f"✗ Error accessing MARS ENS data: {e}")
        if os.path.exists("test_ens_mean_20201101_12.grib"):
            os.remove("test_ens_mean_20201101_12.grib")
        return False

def test_mars_ens_perturbed_member():
    """Test accessing a specific perturbed forecast member."""
    
    print("\n=== Testing Perturbed Forecast Member ===")
    
    try:
        server = ECMWFService("mars")
        
        # Common parameters for ENS operational data
        common_ens_params = {
            "class": "od",
            "stream": "oper",
            "date": "20201101",
            "time": "12",
            "step": "24",
            "levtype": "sfc",
            "param": "msl",
            "grid": "1.0/1.0",
            "format": "grib",
        }
        
        # Test Perturbed Forecast Member 1
        print("Testing Perturbed Forecast Member 1...")
        perturbed_member_request = common_ens_params.copy()
        perturbed_member_request["type"] = "pf"  # Type for Perturbed Forecast
        perturbed_member_request["number"] = "1"  # Member number 1
        target_pf_file = "test_ens_pf01_20201101_12.grib"
        
        print(f"Requesting ECMWF Perturbed Forecast Member 1...")
        server.execute(perturbed_member_request, target_pf_file)
        print(f"✓ Successfully downloaded Perturbed Forecast Member 1 to {target_pf_file}")
        
        # Load and inspect the data using xarray
        ds_pf = xr.open_dataset(target_pf_file, engine="cfgrib")
        print("\n--- Perturbed Forecast Member 1 Data (xarray Dataset) ---")
        print(ds_pf)
        
        return True
        
    except Exception as e:
        print(f"✗ Error accessing perturbed forecast member: {e}")
        if os.path.exists("test_ens_pf01_20201101_12.grib"):
            os.remove("test_ens_pf01_20201101_12.grib")
        return False

def test_mars_ens_hurricane_variables():
    """Test accessing hurricane-relevant variables from MARS ENS."""
    
    print("\n=== Testing MARS ENS Hurricane Variables ===")
    
    try:
        server = ECMWFService("mars")
        
        # Hurricane-relevant parameters
        hurricane_params = {
            "msl": "Mean sea level pressure",
            "2t": "2 metre temperature",
            "10u": "10 metre U wind component",
            "10v": "10 metre V wind component",
            "tp": "Total precipitation"
        }
        
        for param_id, description in hurricane_params.items():
            print(f"Testing {description} (param: {param_id})...")
            
            var_request = {
                "class": "od",
                "stream": "oper",
                "date": "20201101",
                "time": "12",
                "step": "24",
                "levtype": "sfc",
                "param": param_id,
                "type": "em",  # Ensemble mean
                "grid": "1.0/1.0",
                "format": "grib",
            }
            
            target_file = f"test_ens_{param_id}_20201101_12.grib"
            server.execute(var_request, target_file)
            print(f"  ✓ {description} downloaded successfully")
            
            # Clean up test file
            if os.path.exists(target_file):
                os.remove(target_file)
        
        return True
        
    except Exception as e:
        print(f"✗ Error accessing hurricane variables: {e}")
        return False

def test_mars_ens_multiple_lead_times():
    """Test accessing multiple lead times for ensemble forecasts."""
    
    print("\n=== Testing MARS ENS Multiple Lead Times ===")
    
    try:
        server = ECMWFService("mars")
        
        # Test multiple lead times
        lead_times = ["0", "12", "24", "48", "72"]
        
        for step in lead_times:
            print(f"Testing lead time: {step} hours...")
            
            step_request = {
                "class": "od",
                "stream": "oper",
                "date": "20201101",
                "time": "12",
                "step": step,
                "levtype": "sfc",
                "param": "msl",
                "type": "em",
                "grid": "1.0/1.0",
                "format": "grib",
            }
            
            target_file = f"test_ens_step_{step}_20201101_12.grib"
            server.execute(step_request, target_file)
            print(f"  ✓ Lead time {step}h downloaded successfully")
            
            # Clean up test file
            if os.path.exists(target_file):
                os.remove(target_file)
        
        return True
        
    except Exception as e:
        print(f"✗ Error accessing multiple lead times: {e}")
        return False

if __name__ == "__main__":
    setup_ecmwf_api()
    test_mars_ens_access()
    test_mars_ens_perturbed_member()
    test_mars_ens_hurricane_variables()
    test_mars_ens_multiple_lead_times() 