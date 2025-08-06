#!/usr/bin/env python3
"""
Test script for Enhanced Earthquake Detection Model
This script tests the enhanced ML model to ensure it works correctly.
"""

import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from enhanced_realtime_detector import EnhancedEarthquakeDetector, EnhancedFeatureExtractor
    print("✓ Enhanced detector modules imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

def test_feature_extraction():
    """Test the enhanced feature extraction"""
    print("\n=== Testing Enhanced Feature Extraction ===")
    
    extractor = EnhancedFeatureExtractor()
    
    # Generate test data
    np.random.seed(42)
    test_data = {
        'acc_x': np.random.normal(0, 0.1, 1000),
        'acc_y': np.random.normal(0, 0.1, 1000),
        'acc_z': np.random.normal(1.0, 0.1, 1000)
    }
    
    # Add some earthquake-like signal
    test_data['acc_x'][400:600] += np.random.normal(0, 0.5, 200)
    test_data['acc_y'][400:600] += np.random.normal(0, 0.5, 200)
    test_data['acc_z'][400:600] += np.random.normal(0, 0.6, 200)
    
    try:
        features = extractor.extract_advanced_features(test_data)
        print(f"✓ Extracted {len(features)} features successfully")
        print(f"  Sample features: {list(features.keys())[:10]}")
        return True
    except Exception as e:
        print(f"✗ Feature extraction failed: {e}")
        return False

def test_enhanced_detector():
    """Test the enhanced earthquake detector"""
    print("\n=== Testing Enhanced Earthquake Detector ===")
    
    try:
        # Create detector (will use default ensemble if no trained model exists)
        detector = EnhancedEarthquakeDetector()
        print("✓ Enhanced detector created successfully")
        
        # Test with normal data
        print("\nTesting with normal data...")
        np.random.seed(42)
        for i in range(50):
            timestamp = f"2024-01-01T00:00:{i:02d}"
            acc_x = np.random.normal(0, 0.05)
            acc_y = np.random.normal(0, 0.05)
            acc_z = np.random.normal(1.0, 0.05)
            detector.add_sensor_data(timestamp, acc_x, acc_y, acc_z)
        
        # Wait for buffer to fill
        for i in range(1000):
            timestamp = f"2024-01-01T00:10:{i%60:02d}"
            acc_x = np.random.normal(0, 0.05)
            acc_y = np.random.normal(0, 0.05)
            acc_z = np.random.normal(1.0, 0.05)
            detector.add_sensor_data(timestamp, acc_x, acc_y, acc_z)
        
        # Test prediction with normal data
        prediction = detector.predict_earthquake()
        if prediction:
            print(f"✓ Normal data prediction: {prediction['probability']:.3f} (should be low)")
        else:
            print("⚠ No prediction returned")
        
        # Test with earthquake-like data
        print("\nTesting with earthquake-like data...")
        for i in range(100):
            timestamp = f"2024-01-01T00:20:{i%60:02d}"
            # Simulate earthquake signal
            acc_x = np.random.normal(0, 0.5)
            acc_y = np.random.normal(0, 0.5)
            acc_z = np.random.normal(1.0, 0.6)
            detector.add_sensor_data(timestamp, acc_x, acc_y, acc_z)
        
        # Test prediction with earthquake data
        prediction = detector.predict_earthquake()
        if prediction:
            print(f"✓ Earthquake data prediction: {prediction['probability']:.3f} (should be higher)")
            print(f"  Confidence: {prediction.get('confidence', 'N/A')}")
            print(f"  Adaptive threshold: {prediction.get('adaptive_threshold', 'N/A'):.3f}")
        else:
            print("⚠ No prediction returned for earthquake data")
        
        # Test performance summary
        summary = detector.get_performance_summary()
        if isinstance(summary, dict):
            print(f"✓ Performance summary: {summary}")
        else:
            print(f"✓ Performance summary: {summary}")
        
        return True
        
    except Exception as e:
        print(f"✗ Enhanced detector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_capability():
    """Test if training script can be imported and run"""
    print("\n=== Testing Training Capability ===")
    
    try:
        from train_enhanced_model import EnhancedModelTrainer
        print("✓ Training module imported successfully")
        
        # Test trainer creation
        trainer = EnhancedModelTrainer()
        print("✓ Model trainer created successfully")
        
        # Test synthetic data generation (small sample)
        print("Testing synthetic data generation...")
        X, y = trainer.generate_synthetic_data(n_samples=100)
        print(f"✓ Generated {len(X)} samples with {len(X.columns)} features")
        print(f"  Class distribution: Normal={np.sum(y==0)}, Earthquake={np.sum(y==1)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Training capability test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("ENHANCED EARTHQUAKE DETECTION MODEL TESTS")
    print("="*60)
    
    tests = [
        ("Feature Extraction", test_feature_extraction),
        ("Enhanced Detector", test_enhanced_detector),
        ("Training Capability", test_training_capability)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\nRunning {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    passed = 0
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:.<30} {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Enhanced model is ready to use.")
    else:
        print(f"\n⚠ {len(results) - passed} test(s) failed. Check the errors above.")
    
    print("\nNext steps:")
    print("1. Run 'python train_enhanced_model.py' to train the enhanced model")
    print("2. Use 'python graph.py' to test real-time detection with enhanced model")

if __name__ == "__main__":
    main()
