#!/usr/bin/env python3
"""
Test script for Hurricane Data Download System

This script tests the configuration and basic functionality of the hurricane
data download system without actually downloading data.
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_configuration():
    """Test that the configuration file can be loaded."""
    print("Testing configuration file...")
    
    try:
        from data_prep.hurricane_helper.hurricane_downloader import HurricaneDownloader
        downloader = HurricaneDownloader()
        print("✅ Configuration file loaded successfully")
        
        # Test basic config structure
        required_sections = ['hurricane', 'model', 'download', 'ensemble']
        for section in required_sections:
            if section in downloader.config:
                print(f"✅ {section} section found in config")
            else:
                print(f"❌ {section} section missing from config")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_directory_structure():
    """Test that required directories can be created."""
    print("\nTesting directory structure...")
    
    try:
        from data_prep.hurricane_helper.hurricane_downloader import HurricaneDownloader
        downloader = HurricaneDownloader()
        
        # Check if directories were created
        if os.path.exists(downloader.output_dir):
            print(f"✅ Output directory created: {downloader.output_dir}")
        else:
            print(f"❌ Output directory not created: {downloader.output_dir}")
            return False
        
        if os.path.exists(downloader.processed_dir):
            print(f"✅ Processed directory created: {downloader.processed_dir}")
        else:
            print(f"❌ Processed directory not created: {downloader.processed_dir}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Directory structure test failed: {e}")
        return False

def test_url_generation():
    """Test that download URLs can be generated correctly."""
    print("\nTesting URL generation...")
    
    try:
        from data_prep.hurricane_helper.hurricane_downloader import HurricaneDownloader
        from datetime import datetime
        
        downloader = HurricaneDownloader()
        
        # Test URL generation for a sample date
        test_date = datetime(2024, 10, 15)
        url = downloader._build_download_url(test_date)
        
        print(f"✅ Generated URL: {url}")
        
        # Check if URL has expected structure
        expected_parts = ['FNV3', '2024_10_15', '12_00', 'paired', 'csv']
        for part in expected_parts:
            if part in url:
                print(f"✅ URL contains expected part: {part}")
            else:
                print(f"❌ URL missing expected part: {part}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ URL generation test failed: {e}")
        return False

def test_analyzer_import():
    """Test that the analyzer module can be imported."""
    print("\nTesting analyzer module...")
    
    try:
        from data_prep.hurricane_helper.hurricane_analyzer import HurricaneAnalyzer
        print("✅ Analyzer module imported successfully")
        
        # Test analyzer initialization (will fail if data directory doesn't exist, which is expected)
        try:
            analyzer = HurricaneAnalyzer()
            print("✅ Analyzer initialized successfully")
        except FileNotFoundError:
            print("⚠️  Analyzer initialization expected to fail (no data directory yet)")
        
        return True
        
    except Exception as e:
        print(f"❌ Analyzer import test failed: {e}")
        return False

def test_visualizer_import():
    """Test that the visualizer module can be imported."""
    print("\nTesting visualizer module...")
    
    try:
        from data_prep.hurricane_helper.hurricane_visualizer import HurricaneVisualizer
        print("✅ Visualizer module imported successfully")
        
        # Test visualizer initialization
        visualizer = HurricaneVisualizer()
        print("✅ Visualizer initialized successfully")
        
        # Check if plots directory was created
        if os.path.exists(visualizer.output_dir):
            print(f"✅ Plots directory created: {visualizer.output_dir}")
        else:
            print(f"❌ Plots directory not created: {visualizer.output_dir}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Visualizer import test failed: {e}")
        return False

def test_dependencies():
    """Test that all required dependencies are available."""
    print("\nTesting dependencies...")
    
    required_packages = [
        'yaml', 'requests', 'pandas', 'numpy', 
        'matplotlib', 'seaborn', 'pathlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} available")
        except ImportError:
            print(f"❌ {package} not available")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {missing_packages}")
        print("Install missing packages with: pip install -r requirements.txt")
        return False
    else:
        print("✅ All required dependencies available")
        return True

def main():
    """Run all tests."""
    print("="*60)
    print("HURRICANE DATA DOWNLOAD SYSTEM - TEST SUITE")
    print("="*60)
    
    tests = [
        test_dependencies,
        test_configuration,
        test_directory_structure,
        test_url_generation,
        test_analyzer_import,
        test_visualizer_import
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Review config/hurricane_config.yaml")
        print("2. Run: python src/download_hurricane_data.py")
        print("3. Check the generated data and plots")
    else:
        print("❌ Some tests failed. Please fix the issues before proceeding.")
        print("\nCommon fixes:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Check that config/hurricane_config.yaml exists and is valid")
        print("3. Ensure you have write permissions in the project directory")
    
    print("="*60)

if __name__ == "__main__":
    main() 