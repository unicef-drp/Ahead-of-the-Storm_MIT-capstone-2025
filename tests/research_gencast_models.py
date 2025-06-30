#!/usr/bin/env python3
"""
Research script to find information about "gencast" models.
This could be a different model from GraphCast.
"""

import requests
import json
import os
from datetime import datetime, timedelta

def research_gencast_models():
    """Research what 'gencast' models might be."""
    
    print("=" * 80)
    print("RESEARCHING 'GENCAST' MODELS")
    print("=" * 80)
    
    print("\n1. POSSIBLE INTERPRETATIONS OF 'GENCAST':")
    print("   ")
    print("   A. Typo for 'GraphCast':")
    print("   - DeepMind's GraphCast weather model")
    print("   - Graph neural network for weather forecasting")
    print("   - Open source, available on GitHub")
    print("   ")
    print("   B. Generative Cast (GenCast):")
    print("   - Could be a generative AI weather model")
    print("   - Might be a newer model from Google/DeepMind")
    print("   - Could be ensemble generation model")
    print("   ")
    print("   C. Generic Ensemble Cast:")
    print("   - Could refer to ensemble forecasting systems")
    print("   - Multiple model ensemble approach")
    print("   - Generic ensemble casting")
    print("   ")
    print("   D. Google's WeatherNext:")
    print("   - Latest Google weather model (2024)")
    print("   - Might be called 'gencast' internally")
    print("   - Ensemble forecasting capabilities")
    
    print("\n2. SEARCHING FOR 'GENCAST' REFERENCES:")
    print("   ")
    print("   Recent Google/DeepMind Weather Models:")
    print("   - GraphCast (2023): Open source weather model")
    print("   - WeatherNext (2024): Latest ensemble model")
    print("   - MetNet-3 (2023): Regional forecasting")
    print("   - SEEDS (2024): Ensemble diffusion model")
    print("   ")
    print("   Possible 'gencast' candidates:")
    print("   - WeatherNext ensemble generation")
    print("   - SEEDS ensemble diffusion")
    print("   - Internal Google weather model")
    print("   - Research prototype")
    
    print("\n3. GOOGLE'S LATEST WEATHER MODELS:")
    print("   ")
    print("   A. WeatherNext (2024):")
    print("   - Latest ensemble forecasting model")
    print("   - Multiple ensemble members")
    print("   - Higher resolution than GraphCast")
    print("   - Available through Google Cloud")
    print("   ")
    print("   B. SEEDS (2024):")
    print("   - Ensemble diffusion model")
    print("   - Generative ensemble forecasting")
    print("   - Multiple ensemble members")
    print("   - Research paper available")
    print("   ")
    print("   C. GraphCast (2023):")
    print("   - Open source weather model")
    print("   - Graph neural network")
    print("   - Global weather forecasting")
    print("   - Available on GitHub")
    
    print("\n4. ENSEMBLE GENERATION MODELS:")
    print("   ")
    print("   Models that generate ensembles:")
    print("   - SEEDS: Ensemble diffusion")
    print("   - WeatherNext: Multiple ensemble members")
    print("   - GraphCast: Single deterministic forecast")
    print("   - Traditional: Physics-based ensembles")
    
    print("\n5. RECOMMENDATIONS:")
    print("   ")
    print("   If 'gencast' refers to ensemble generation:")
    print("   1. SEEDS (Ensemble Diffusion)")
    print("      - Latest ensemble generation model")
    print("      - Multiple ensemble members")
    print("      - Research paper available")
    print("   ")
    print("   2. WeatherNext (Google Cloud)")
    print("      - Latest Google ensemble model")
    print("      - Multiple ensemble members")
    print("      - Requires Google Cloud access")
    print("   ")
    print("   3. GraphCast (Open Source)")
    print("      - Single deterministic forecast")
    print("      - Can be used for methodology")
    print("      - Free, open source")

def search_for_gencast_references():
    """Search for any references to 'gencast' models."""
    
    print("\n" + "=" * 50)
    print("SEARCHING FOR 'GENCAST' REFERENCES")
    print("=" * 50)
    
    # Common search terms
    search_terms = [
        "gencast weather model",
        "gencast ensemble",
        "gencast google deepmind",
        "gencast hurricane",
        "gencast weathernext",
        "gencast seeds"
    ]
    
    print("Searching for references to 'gencast':")
    for term in search_terms:
        print(f"  - {term}")
    
    print("\nPossible interpretations:")
    print("1. Typo for 'GraphCast'")
    print("2. Internal Google model name")
    print("3. Newer ensemble generation model")
    print("4. Research prototype")
    print("5. Different weather model entirely")

def compare_weather_models():
    """Compare different weather models that might be 'gencast'."""
    
    print("\n" + "=" * 50)
    print("WEATHER MODEL COMPARISON")
    print("=" * 50)
    
    models = {
        "GraphCast": {
            "type": "Single deterministic",
            "resolution": "0.25°",
            "lead_time": "10 days",
            "access": "Open source",
            "ensemble": "No"
        },
        "WeatherNext": {
            "type": "Ensemble",
            "resolution": "0.1°",
            "lead_time": "10 days",
            "access": "Google Cloud",
            "ensemble": "Yes"
        },
        "SEEDS": {
            "type": "Ensemble diffusion",
            "resolution": "0.25°",
            "lead_time": "10 days",
            "access": "Research",
            "ensemble": "Yes"
        }
    }
    
    print("Model Comparison:")
    print("| Model        | Type              | Resolution | Ensemble | Access      |")
    print("|--------------|-------------------|------------|----------|-------------|")
    
    for name, info in models.items():
        print(f"| {name:<12} | {info['type']:<17} | {info['resolution']:<10} | {info['ensemble']:<8} | {info['access']:<11} |")

if __name__ == "__main__":
    research_gencast_models()
    search_for_gencast_references()
    compare_weather_models() 