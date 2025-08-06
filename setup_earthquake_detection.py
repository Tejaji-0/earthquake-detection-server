#!/usr/bin/env python3
"""
Setup script for Earthquake Detection ML Pipeline
This script sets up the environment and runs the complete pipeline.
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages from requirements.txt"""
    print("Installing required packages...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✓ Successfully installed all requirements")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing requirements: {e}")
        return False

def check_data_files():
    """Check if required data files exist"""
    print("Checking for required data files...")
    
    required_files = [
        "earthquake_1995-2023.csv"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"✗ Missing required files: {missing_files}")
        return False
    else:
        print("✓ All required data files found")
        return True

def run_ml_pipeline():
    """Run the machine learning pipeline"""
    print("Running ML pipeline...")
    try:
        subprocess.check_call([sys.executable, "earthquake_ml_pipeline.py"])
        print("✓ ML pipeline completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running ML pipeline: {e}")
        return False

def run_realtime_detector():
    """Run the real-time earthquake detector"""
    print("Starting real-time earthquake detector...")
    try:
        subprocess.check_call([sys.executable, "realtime_earthquake_detector.py"])
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error running real-time detector: {e}")
        return False

def main():
    """Main setup function"""
    print("=== Earthquake Detection System Setup ===")
    
    # Check current directory
    if not os.path.exists("earthquake_1995-2023.csv"):
        print("Please run this script from the data directory containing earthquake_1995-2023.csv")
        return
    
    print("Choose setup option:")
    print("1. Full setup (install packages + train models)")
    print("2. Install packages only")
    print("3. Train models only (requires packages)")
    print("4. Run real-time detector only")
    
    try:
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == "1":
            # Full setup
            if install_requirements() and check_data_files():
                run_ml_pipeline()
                
        elif choice == "2":
            # Install packages only
            install_requirements()
            
        elif choice == "3":
            # Train models only
            if check_data_files():
                run_ml_pipeline()
                
        elif choice == "4":
            # Run real-time detector
            run_realtime_detector()
            
        else:
            print("Invalid choice")
            
    except KeyboardInterrupt:
        print("\nSetup cancelled by user")
    except Exception as e:
        print(f"Setup error: {e}")

if __name__ == "__main__":
    main()
