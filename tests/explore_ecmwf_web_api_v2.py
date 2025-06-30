#!/usr/bin/env python3
"""
Explore ECMWF Web API with correct dataset names based on documentation.
"""

import ecmwfapi
import os

def setup_ecmwf_api():
    """Set up ECMWF Web API configuration."""
    
    print("Setting up ECMWF Web API configuration...")
    
    ecmwfapirc_content = """{
    "url"   : "https://api.ecmwf.int/v1",
    "key"   : "b33ee70d55beeb0e00d45ddf7a1a010d",
    "email" : "sidvijay@mit.edu"
}"""
    
    with open(os.path.expanduser('~/.ecmwfapirc'), 'w') as f:
        f.write(ecmwfapirc_content)
    
    print("✓ ECMWF API configuration file created")

def test_known_datasets():
    """Test known ECMWF Web API datasets."""
    
    print("\n=== Testing Known ECMWF Web API Datasets ===")
    
    try:
        client = ecmwfapi.ECMWFDataServer()
        print("✓ ECMWF API client initialized")
        
        # Based on ECMWF Web API documentation, try these known datasets
        known_datasets = [
            # ERA5 datasets (these should work)
            {'class': 'ei', 'dataset': 'interim', 'stream': 'oper', 'type': 'an'},
            {'class': 'ei', 'dataset': 'interim', 'stream': 'oper', 'type': 'fc'},
            
            # Try different ensemble configurations
            {'class': 'ei', 'dataset': 'interim', 'stream': 'enfo', 'type': 'pf'},
            {'class': 'ei', 'dataset': 'interim', 'stream': 'enfo', 'type': 'cf'},
            
            # Try operational data
            {'class': 'od', 'dataset': 'oper', 'stream': 'oper', 'type': 'an'},
            {'class': 'od', 'dataset': 'oper', 'stream': 'oper', 'type': 'fc'},
            
            # Try ensemble operational data
            {'class': 'od', 'dataset': 'oper', 'stream': 'enfo', 'type': 'pf'},
            {'class': 'od', 'dataset': 'oper', 'stream': 'enfo', 'type': 'cf'},
        ]
        
        for i, config in enumerate(known_datasets):
            print(f"\nTrying config {i+1}: {config}")
            try:
                test_params = {
                    **config,
                    'expver': '1',
                    'levtype': 'sfc',
                    'param': '151.128',  # Mean sea level pressure
                    'date': '2020-11-01',
                    'time': '00:00:00',
                    'step': '0',
                    'area': [20.0, -85.0, 10.0, -75.0],
                    'grid': '0.5/0.5',
                    'format': 'netcdf',
                    'target': f'test_known_{i+1}.nc'
                }
                
                client.retrieve(test_params)
                print(f"  ✓ Config {i+1} works!")
                
            except Exception as e:
                print(f"  ✗ Config {i+1} failed: {str(e)[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_ensemble_specific():
    """Test ensemble-specific configurations."""
    
    print(f"\n=== Testing Ensemble-Specific Configurations ===")
    
    try:
        client = ecmwfapi.ECMWFDataServer()
        
        # Try ensemble-specific parameters
        ensemble_configs = [
            # ERA-Interim ensemble
            {
                'class': 'ei',
                'dataset': 'interim',
                'stream': 'enfo',
                'type': 'pf',
                'number': '1/2/3/4/5'  # Ensemble members
            },
            # Operational ensemble
            {
                'class': 'od',
                'dataset': 'oper',
                'stream': 'enfo',
                'type': 'pf',
                'number': '1/2/3/4/5'
            },
            # Try without dataset specification
            {
                'class': 'ei',
                'stream': 'enfo',
                'type': 'pf',
                'number': '1/2/3/4/5'
            },
            {
                'class': 'od',
                'stream': 'enfo',
                'type': 'pf',
                'number': '1/2/3/4/5'
            }
        ]
        
        for i, config in enumerate(ensemble_configs):
            print(f"\nTrying ensemble config {i+1}: {config}")
            try:
                test_params = {
                    **config,
                    'expver': '1',
                    'levtype': 'sfc',
                    'param': '151.128',
                    'date': '2020-11-01',
                    'time': '00:00:00',
                    'step': '24',
                    'area': [20.0, -85.0, 10.0, -75.0],
                    'grid': '0.5/0.5',
                    'format': 'netcdf',
                    'target': f'test_ensemble_{i+1}.nc'
                }
                
                client.retrieve(test_params)
                print(f"  ✓ Ensemble config {i+1} works!")
                
            except Exception as e:
                print(f"  ✗ Ensemble config {i+1} failed: {str(e)[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_parameter_codes():
    """Test different parameter codes for hurricane-relevant variables."""
    
    print(f"\n=== Testing Parameter Codes ===")
    
    try:
        client = ecmwfapi.ECMWFDataServer()
        
        # Hurricane-relevant parameter codes
        hurricane_params = {
            '151.128': 'Mean sea level pressure',
            '165.128': '10 metre U wind component',
            '166.128': '10 metre V wind component',
            '228.128': 'Total precipitation',
            '134.128': 'Surface pressure',
            '129.128': 'Geopotential',
            '130.128': 'Temperature',
            '157.128': 'Relative humidity'
        }
        
        # Use a working configuration (if we find one)
        base_config = {
            'class': 'ei',
            'dataset': 'interim',
            'stream': 'oper',
            'type': 'an',
            'expver': '1',
            'levtype': 'sfc',
            'date': '2020-11-01',
            'time': '00:00:00',
            'step': '0',
            'area': [20.0, -85.0, 10.0, -75.0],
            'grid': '0.5/0.5',
            'format': 'netcdf'
        }
        
        for param_id, description in hurricane_params.items():
            print(f"\nTesting {description} (param: {param_id})...")
            try:
                test_params = {
                    **base_config,
                    'param': param_id,
                    'target': f'test_param_{param_id.replace(".", "_")}.nc'
                }
                
                client.retrieve(test_params)
                print(f"  ✓ {description} works!")
                
            except Exception as e:
                print(f"  ✗ {description} failed: {str(e)[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    setup_ecmwf_api()
    test_known_datasets()
    test_ensemble_specific()
    test_parameter_codes() 