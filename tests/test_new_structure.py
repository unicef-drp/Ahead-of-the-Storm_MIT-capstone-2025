#!/usr/bin/env python3
"""
Test script for the new file structure

This script tests that all imports work correctly with the reorganized
hurricane data download system.
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported correctly."""
    print("Testing imports with new file structure...")
    
    try:
        # Test importing from hurricane_helper package
        from data_prep.hurricane_helper import HurricaneDownloader, HurricaneAnalyzer, HurricaneVisualizer
        print("✅ Successfully imported all hurricane helper modules")
        
        # Test individual imports
        from data_prep.hurricane_helper.hurricane_downloader import HurricaneDownloader
        from data_prep.hurricane_helper.hurricane_analyzer import HurricaneAnalyzer
        from data_prep.hurricane_helper.hurricane_visualizer import HurricaneVisualizer
        print("✅ Successfully imported individual modules")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_configuration():
    """Test that configuration can be loaded."""
    print("\nTesting configuration loading...")
    
    try:
        from data_prep.hurricane_helper.hurricane_downloader import HurricaneDownloader
        downloader = HurricaneDownloader()
        print("✅ Configuration loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("="*50)
    print("NEW FILE STRUCTURE TEST")
    print("="*50)
    
    tests = [test_imports, test_configuration]
    passed = 0
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\nTests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("✅ All tests passed! New structure is working correctly.")
        print("\nYou can now run:")
        print("  python src/download_hurricane_data.py")
        print("  python tests/test_hurricane_system.py")
    else:
        print("❌ Some tests failed. Please check the import paths.")
    
    print("="*50)

if __name__ == "__main__":
    main() 