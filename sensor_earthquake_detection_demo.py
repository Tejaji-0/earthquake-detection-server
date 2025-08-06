#!/usr/bin/env python3
"""
Sensor-Based Earthquake Detection Demo
This script demonstrates the complete workflow for training and using
ML models for earthquake detection from accelerometer and gyroscope data.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from accelerometer_feature_extractor import AccelerometerFeatureExtractor
from sensor_earthquake_ml_pipeline import SensorEarthquakeMLPipeline
from realtime_sensor_earthquake_detector import RealTimeEarthquakeDetector

def create_demo_sensor_data():
    """Create demo sensor data files for training and testing"""
    print("=== Creating Demo Sensor Data ===")
    
    extractor = AccelerometerFeatureExtractor()
    
    # Create training data samples
    normal_files = []
    earthquake_files = []
    
    # Generate normal data samples
    print("Generating normal sensor data samples...")
    for i in range(20):
        filename = f"demo_normal_{i:02d}.csv"
        extractor.create_sample_data(filename, duration=np.random.uniform(8, 15))
        normal_files.append((filename, 0))  # Label 0 for normal
    
    # Generate earthquake data samples
    print("Generating earthquake sensor data samples...")
    for i in range(20):
        filename = f"demo_earthquake_{i:02d}.csv"
        extractor.create_sample_data(filename, duration=np.random.uniform(10, 20))
        earthquake_files.append((filename, 1))  # Label 1 for earthquake
    
    all_files = normal_files + earthquake_files
    print(f"✓ Created {len(all_files)} demo sensor data files")
    
    return all_files

def demonstrate_feature_extraction():
    """Demonstrate feature extraction from sensor data"""
    print("\n=== Feature Extraction Demo ===")
    
    # Create sample data
    extractor = AccelerometerFeatureExtractor()
    
    # Create sample normal and earthquake data
    normal_file = extractor.create_sample_data('demo_normal_sample.csv', duration=10)
    earthquake_file = extractor.create_sample_data('demo_earthquake_sample.csv', duration=10)
    
    # Extract features
    normal_features = extractor.process_sensor_file(normal_file, label=0)
    earthquake_features = extractor.process_sensor_file(earthquake_file, label=1)
    
    if normal_features and earthquake_features:
        print(f"✓ Extracted {len(normal_features)-3} features per sample")  # -3 for metadata
        
        # Show key differences
        print("\n📊 Key Feature Comparison:")
        key_features = [
            'pga', 'pgv', 'arias_intensity', 'acc_magnitude_max', 
            'acc_x_energy', 'significant_duration', 'acc_x_freq_dominant'
        ]
        
        print(f"{'Feature':<20} {'Normal':<12} {'Earthquake':<12} {'Ratio':<8}")
        print("-" * 52)
        
        for feature in key_features:
            if feature in normal_features and feature in earthquake_features:
                normal_val = normal_features[feature]
                earthquake_val = earthquake_features[feature]
                ratio = earthquake_val / normal_val if normal_val != 0 else float('inf')
                
                print(f"{feature:<20} {normal_val:<12.3f} {earthquake_val:<12.3f} {ratio:<8.2f}")
    
    # Cleanup
    os.remove(normal_file)
    os.remove(earthquake_file)

def demonstrate_ml_training():
    """Demonstrate ML model training"""
    print("\n=== ML Training Demo ===")
    
    # Initialize pipeline
    pipeline = SensorEarthquakeMLPipeline(output_dir='demo_sensor_models')
    
    # Train models with synthetic data
    print("Training ML models with synthetic sensor data...")
    success = pipeline.run_complete_pipeline(use_synthetic_data=True)
    
    if success:
        print("✓ ML training completed successfully!")
        print(f"📁 Models saved to: {pipeline.output_dir}")
        
        # Show training results
        report_file = os.path.join(pipeline.output_dir, 'training_report.json')
        if os.path.exists(report_file):
            import json
            with open(report_file, 'r') as f:
                report = json.load(f)
            
            print(f"\n📈 Best Model: {report['best_model']['name']} "
                  f"(ROC-AUC: {report['best_model']['roc_auc']:.3f})")
            
            print("\n🎯 Model Performance:")
            for model_name, perf in report['model_performance'].items():
                print(f"  {model_name}: {perf['test_roc_auc']:.3f}")
    
    return success

def demonstrate_real_time_detection():
    """Demonstrate real-time earthquake detection"""
    print("\n=== Real-Time Detection Demo ===")
    
    # Initialize detector
    detector = RealTimeEarthquakeDetector(model_dir='demo_sensor_models')
    
    if detector.model is None:
        print("⚠ No trained model found. Please run ML training first.")
        return False
    
    print("✓ Real-time detector initialized")
    
    # Create sample CSV data for testing
    csv_file = detector.create_sample_csv('demo_realtime_data.csv', duration=30)
    
    print(f"\n🔄 Processing sample data from: {csv_file}")
    print("(This simulates real-time sensor data processing)")
    
    # Process the CSV file
    detector.process_real_time_data('csv')
    
    # Show statistics
    stats = detector.get_detection_statistics()
    if stats:
        print(f"\n📊 Detection Results:")
        print(f"  Total predictions: {stats['total_predictions']}")
        print(f"  Earthquake detections: {stats['earthquake_detections']}")
        print(f"  Detection rate: {stats['detection_rate']:.1%}")
        print(f"  Average probability: {stats['avg_probability']:.3f}")
        print(f"  Max probability: {stats['max_probability']:.3f}")
    
    # Cleanup
    if os.path.exists(csv_file):
        os.remove(csv_file)
    
    return True

def create_sensor_data_format_example():
    """Create an example of the expected sensor data format"""
    print("\n=== Sensor Data Format Example ===")
    
    # Create example CSV file
    example_file = 'sensor_data_format_example.csv'
    
    # Generate 5 seconds of example data
    sampling_rate = 100  # 100 Hz
    duration = 5
    num_samples = duration * sampling_rate
    
    # Generate timestamps
    timestamps = pd.date_range(start='2024-01-01 12:00:00', periods=num_samples, freq='10ms')
    
    # Generate example sensor readings
    data = []
    for i, timestamp in enumerate(timestamps):
        t = i / sampling_rate
        
        # Example accelerometer data (m/s²)
        acc_x = 0.1 * np.sin(2 * np.pi * 1 * t) + np.random.normal(0, 0.05)
        acc_y = 0.15 * np.cos(2 * np.pi * 0.8 * t) + np.random.normal(0, 0.05)
        acc_z = 9.81 + 0.05 * np.sin(2 * np.pi * 0.5 * t) + np.random.normal(0, 0.02)
        
        # Example gyroscope data (rad/s)
        gyro_x = 0.01 * np.sin(2 * np.pi * 2 * t) + np.random.normal(0, 0.005)
        gyro_y = 0.01 * np.cos(2 * np.pi * 1.5 * t) + np.random.normal(0, 0.005)
        gyro_z = np.random.normal(0, 0.002)
        
        data.append([timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z])
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=[
        'timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'
    ])
    
    # Save example file
    df.to_csv(example_file, index=False)
    
    print(f"✓ Created sensor data format example: {example_file}")
    print("\n📋 Required CSV format:")
    print("  - timestamp: ISO format (YYYY-MM-DD HH:MM:SS.mmm)")
    print("  - acc_x, acc_y, acc_z: Accelerometer readings (m/s²)")
    print("  - gyro_x, gyro_y, gyro_z: Gyroscope readings (rad/s) [optional]")
    print(f"\n📄 Sample data (first 5 rows):")
    print(df.head().to_string(index=False))
    
    return example_file

def cleanup_demo_files():
    """Clean up demo files"""
    print("\n=== Cleaning Up Demo Files ===")
    
    # List of file patterns to clean
    patterns = [
        'demo_normal_*.csv',
        'demo_earthquake_*.csv',
        'demo_*.csv',
        'sample_*.csv'
    ]
    
    import glob
    removed_count = 0
    
    for pattern in patterns:
        files = glob.glob(pattern)
        for file in files:
            try:
                os.remove(file)
                removed_count += 1
            except:
                pass
    
    print(f"✓ Removed {removed_count} demo files")

def main():
    """Main demo function"""
    print("🌊 SENSOR-BASED EARTHQUAKE DETECTION DEMO 🌊")
    print("=" * 60)
    print("This demo shows how to use accelerometer and gyroscope data")
    print("for real-time earthquake detection and prediction.")
    print("=" * 60)
    
    try:
        # Step 1: Create sensor data format example
        example_file = create_sensor_data_format_example()
        
        # Step 2: Demonstrate feature extraction
        demonstrate_feature_extraction()
        
        # Step 3: Demonstrate ML training
        training_success = demonstrate_ml_training()
        
        if training_success:
            # Step 4: Demonstrate real-time detection
            demonstrate_real_time_detection()
        
        print("\n" + "=" * 60)
        print("🎉 DEMO COMPLETED SUCCESSFULLY! 🎉")
        print("=" * 60)
        
        print("\n📚 What you learned:")
        print("1. How to format sensor data for earthquake detection")
        print("2. Feature extraction from accelerometer/gyroscope data")
        print("3. ML model training for earthquake classification")
        print("4. Real-time earthquake detection from sensor streams")
        
        print(f"\n📁 Generated files:")
        print(f"  - {example_file} (sensor data format example)")
        print(f"  - demo_sensor_models/ (trained ML models)")
        print(f"  - Visualization plots and training reports")
        
        print("\n🚀 Next steps:")
        print("1. Replace synthetic data with real sensor data")
        print("2. Fine-tune detection thresholds for your environment")
        print("3. Integrate with your IoT sensor setup")
        print("4. Add alert mechanisms (SMS, email, sirens, etc.)")
        print("5. Deploy for real-time earthquake monitoring")
        
        # Ask if user wants to keep demo files
        keep_files = input("\nKeep demo files? (y/n): ").strip().lower()
        if keep_files != 'y':
            cleanup_demo_files()
        
    except KeyboardInterrupt:
        print("\n\n⏹ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n👋 Demo finished!")

if __name__ == "__main__":
    main()
