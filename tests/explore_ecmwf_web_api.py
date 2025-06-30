#!/usr/bin/env python3
"""
Explore available datasets and parameters in ECMWF Web API.
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

def explore_datasets():
    """Explore available datasets in ECMWF Web API."""
    
    print("\n=== Exploring Available Datasets ===")
    
    try:
        client = ecmwfapi.ECMWFDataServer()
        print("✓ ECMWF API client initialized")
        
        # Try different dataset names that might contain ensemble forecasts
        datasets_to_try = [
            'enfo',  # Ensemble forecast
            'ens',   # Ensemble
            'oper',  # Operational
            'fc',    # Forecast
            'an',    # Analysis
            'reanalysis-era5-single-levels',
            'reanalysis-era5-pressure-levels'
        ]
        
        print(f"\nTesting different dataset names:")
        
        for dataset in datasets_to_try:
            print(f"\nTrying dataset: {dataset}")
            try:
                # Try a minimal request to see if the dataset exists
                test_params = {
                    'class': 'od',
                    'dataset': dataset,
                    'expver': '1',
                    'stream': 'oper',
                    'type': 'an',
                    'levtype': 'sfc',
                    'param': '151.128',
                    'date': '2020-11-01',
                    'time': '00:00:00',
                    'area': [20.0, -85.0, 10.0, -75.0],
                    'grid': '0.5/0.5',
                    'format': 'netcdf',
                    'target': f'test_{dataset}.nc'
                }
                
                client.retrieve(test_params)
                print(f"  ✓ Dataset {dataset} is accessible")
                
            except Exception as e:
                print(f"  ✗ Dataset {dataset} failed: {str(e)[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def explore_ensemble_datasets():
    """Explore ensemble-specific datasets."""
    
    print(f"\n=== Exploring Ensemble Datasets ===")
    
    try:
        client = ecmwfapi.ECMWFDataServer()
        
        # Try different ensemble dataset configurations
        ensemble_configs = [
            {'class': 'od', 'dataset': 'enfo', 'stream': 'enfo', 'type': 'pf'},
            {'class': 'od', 'dataset': 'enfo', 'stream': 'oper', 'type': 'pf'},
            {'class': 'od', 'dataset': 'ens', 'stream': 'enfo', 'type': 'pf'},
            {'class': 'od', 'dataset': 'ens', 'stream': 'oper', 'type': 'pf'},
            {'class': 'ei', 'dataset': 'enfo', 'stream': 'enfo', 'type': 'pf'},
            {'class': 'ei', 'dataset': 'enfo', 'stream': 'oper', 'type': 'pf'},
            {'class': 'ei', 'dataset': 'ens', 'stream': 'enfo', 'type': 'pf'},
            {'class': 'ei', 'dataset': 'ens', 'stream': 'oper', 'type': 'pf'},
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

def explore_available_classes():
    """Explore available data classes."""
    
    print(f"\n=== Exploring Available Classes ===")
    
    try:
        client = ecmwfapi.ECMWFDataServer()
        
        # Try different data classes
        classes_to_try = ['od', 'ei', 'rd', 'mc', 'fc', 'an']
        
        for class_name in classes_to_try:
            print(f"\nTrying class: {class_name}")
            try:
                test_params = {
                    'class': class_name,
                    'dataset': 'enfo',
                    'expver': '1',
                    'stream': 'oper',
                    'type': 'an',
                    'levtype': 'sfc',
                    'param': '151.128',
                    'date': '2020-11-01',
                    'time': '00:00:00',
                    'area': [20.0, -85.0, 10.0, -75.0],
                    'grid': '0.5/0.5',
                    'format': 'netcdf',
                    'target': f'test_class_{class_name}.nc'
                }
                
                client.retrieve(test_params)
                print(f"  ✓ Class {class_name} is accessible")
                
            except Exception as e:
                print(f"  ✗ Class {class_name} failed: {str(e)[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    setup_ecmwf_api()
    explore_datasets()
    explore_ensemble_datasets()
    explore_available_classes() 