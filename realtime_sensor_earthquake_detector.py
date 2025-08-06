#!/usr/bin/env python3
"""
Real-Time Earthquake Detector
This script provides real-time earthquake detection using trained ML models
and live accelerometer/gyroscope sensor data.
"""

import numpy as np
import pandas as pd
import joblib
import json
import os
import time
from datetime import datetime, timedelta
from collections import deque
import threading
import queue
from accelerometer_feature_extractor import AccelerometerFeatureExtractor
import warnings
warnings.filterwarnings('ignore')

class RealTimeEarthquakeDetector:
    def __init__(self, model_dir='sensor_ml_models', model_name='ensemble'):
        self.model_dir = model_dir
        self.model_name = model_name.lower()
        
        # Load trained model and preprocessing objects
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.feature_info = None
        
        # Initialize feature extractor
        self.feature_extractor = AccelerometerFeatureExtractor()
        
        # Data buffers for windowed analysis
        self.window_size = 10  # seconds
        self.overlap = 0.5     # 50% overlap
        self.sampling_rate = 100  # Hz
        self.buffer_size = int(self.window_size * self.sampling_rate)
        
        # Data queues
        self.data_buffer = {
            'timestamp': deque(maxlen=self.buffer_size),
            'acc_x': deque(maxlen=self.buffer_size),
            'acc_y': deque(maxlen=self.buffer_size),
            'acc_z': deque(maxlen=self.buffer_size),
            'gyro_x': deque(maxlen=self.buffer_size),
            'gyro_y': deque(maxlen=self.buffer_size),
            'gyro_z': deque(maxlen=self.buffer_size)
        }
        
        # Detection settings
        self.detection_threshold = 0.5  # Probability threshold
        self.alert_threshold = 0.8      # High confidence threshold
        self.min_alert_interval = 10    # Minimum seconds between alerts
        
        # State tracking
        self.last_alert_time = None
        self.detection_history = deque(maxlen=100)  # Last 100 predictions
        self.is_running = False
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load the trained ML model and preprocessing objects"""
        print("=== Loading Trained Model ===")
        
        try:
            # Load model
            model_file = os.path.join(self.model_dir, f'{self.model_name}_earthquake_detector.joblib')
            if os.path.exists(model_file):
                self.model = joblib.load(model_file)
                print(f"✓ Loaded model: {model_file}")
            else:
                print(f"⚠ Model file not found: {model_file}")
                return False
            
            # Load scaler
            scaler_file = os.path.join(self.model_dir, 'feature_scaler.joblib')
            if os.path.exists(scaler_file):
                self.scaler = joblib.load(scaler_file)
                print(f"✓ Loaded scaler: {scaler_file}")
            else:
                print(f"⚠ Scaler file not found: {scaler_file}")
                return False
            
            # Load feature selector
            selector_file = os.path.join(self.model_dir, 'feature_selector.joblib')
            if os.path.exists(selector_file):
                self.feature_selector = joblib.load(selector_file)
                print(f"✓ Loaded feature selector: {selector_file}")
            else:
                print(f"⚠ Feature selector file not found: {selector_file}")
                return False
            
            # Load feature info
            feature_file = os.path.join(self.model_dir, 'feature_info.json')
            if os.path.exists(feature_file):
                with open(feature_file, 'r') as f:
                    self.feature_info = json.load(f)
                print(f"✓ Loaded feature info: {self.feature_info['n_features']} features")
            else:
                print(f"⚠ Feature info file not found: {feature_file}")
                return False
            
            print("✓ Model loading completed successfully")
            return True
            
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return False
    
    def add_sensor_data(self, timestamp, acc_x, acc_y, acc_z, gyro_x=0, gyro_y=0, gyro_z=0):
        """Add new sensor data point to the buffer"""
        self.data_buffer['timestamp'].append(timestamp)
        self.data_buffer['acc_x'].append(acc_x)
        self.data_buffer['acc_y'].append(acc_y)
        self.data_buffer['acc_z'].append(acc_z)
        self.data_buffer['gyro_x'].append(gyro_x)
        self.data_buffer['gyro_y'].append(gyro_y)
        self.data_buffer['gyro_z'].append(gyro_z)
    
    def get_buffer_data(self):
        """Get current buffer data as sensor data dictionary"""
        if len(self.data_buffer['acc_x']) < self.buffer_size:
            return None
        
        return {
            'acc_x': list(self.data_buffer['acc_x']),
            'acc_y': list(self.data_buffer['acc_y']),
            'acc_z': list(self.data_buffer['acc_z']),
            'gyro_x': list(self.data_buffer['gyro_x']),
            'gyro_y': list(self.data_buffer['gyro_y']),
            'gyro_z': list(self.data_buffer['gyro_z'])
        }
    
    def extract_features_from_buffer(self):
        """Extract features from current buffer data"""
        sensor_data = self.get_buffer_data()
        if sensor_data is None:
            return None
        
        try:
            # Extract features
            features = self.feature_extractor.extract_all_features(sensor_data, self.sampling_rate)
            
            # Convert to DataFrame with correct feature order
            feature_cols = self.feature_info['all_feature_names']
            feature_values = []
            
            for col in feature_cols:
                if col in features:
                    feature_values.append(features[col])
                else:
                    feature_values.append(0)  # Default value for missing features
            
            feature_df = pd.DataFrame([feature_values], columns=feature_cols)
            
            # Handle missing values
            feature_df = feature_df.fillna(feature_df.median())
            
            return feature_df
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def predict_earthquake(self):
        """Predict earthquake probability from current buffer"""
        if self.model is None:
            return None
        
        # Extract features
        features_df = self.extract_features_from_buffer()
        if features_df is None:
            return None
        
        try:
            # Apply preprocessing
            features_scaled = self.scaler.transform(features_df)
            features_selected = self.feature_selector.transform(features_scaled)
            
            # Make prediction
            probability = self.model.predict_proba(features_selected)[0, 1]  # Probability of earthquake
            prediction = int(probability >= self.detection_threshold)
            
            # Store prediction
            prediction_data = {
                'timestamp': datetime.now(),
                'probability': probability,
                'prediction': prediction,
                'confidence': 'high' if probability >= self.alert_threshold else 'medium' if probability >= self.detection_threshold else 'low'
            }
            
            self.detection_history.append(prediction_data)
            
            return prediction_data
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
    
    def check_alert_conditions(self, prediction_data):
        """Check if alert should be triggered"""
        if prediction_data is None:
            return False
        
        # Check probability threshold
        if prediction_data['probability'] < self.alert_threshold:
            return False
        
        # Check minimum time interval
        current_time = prediction_data['timestamp']
        if self.last_alert_time:
            time_diff = (current_time - self.last_alert_time).total_seconds()
            if time_diff < self.min_alert_interval:
                return False
        
        # Check recent detection history (require multiple detections)
        recent_detections = [p for p in self.detection_history if 
                           (current_time - p['timestamp']).total_seconds() <= 5 and p['prediction'] == 1]
        
        if len(recent_detections) < 2:  # Require at least 2 detections in 5 seconds
            return False
        
        return True
    
    def trigger_alert(self, prediction_data):
        """Trigger earthquake alert"""
        self.last_alert_time = prediction_data['timestamp']
        
        alert_message = f"""
🚨 EARTHQUAKE ALERT 🚨
Time: {prediction_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
Probability: {prediction_data['probability']:.3f}
Confidence: {prediction_data['confidence'].upper()}
Action: Take immediate safety measures!
"""
        
        print(alert_message)
        
        # Log alert
        alert_log = {
            'timestamp': prediction_data['timestamp'].isoformat(),
            'probability': prediction_data['probability'],
            'confidence': prediction_data['confidence'],
            'type': 'earthquake_alert'
        }
        
        self.log_event(alert_log, 'earthquake_alerts.log')
        
        # Here you could add additional alert mechanisms:
        # - Send SMS/email notifications
        # - Trigger IoT devices (sirens, lights)
        # - Send to emergency services
        # - Update mobile app
        
        return True
    
    def log_event(self, event_data, log_file='earthquake_detection.log'):
        """Log events to file"""
        log_path = os.path.join(self.model_dir, log_file)
        
        try:
            with open(log_path, 'a') as f:
                f.write(json.dumps(event_data) + '\n')
        except Exception as e:
            print(f"Error logging event: {e}")
    
    def process_real_time_data(self, data_source='simulation'):
        """Process real-time sensor data"""
        print(f"=== Starting Real-Time Detection ({data_source}) ===")
        print(f"Window size: {self.window_size}s, Sampling rate: {self.sampling_rate}Hz")
        print(f"Detection threshold: {self.detection_threshold}, Alert threshold: {self.alert_threshold}")
        print("Press Ctrl+C to stop\n")
        
        self.is_running = True
        
        try:
            if data_source == 'simulation':
                self.simulate_sensor_stream()
            elif data_source == 'csv':
                self.process_csv_stream()
            else:
                print(f"Unknown data source: {data_source}")
                
        except KeyboardInterrupt:
            print("\n⏹ Detection stopped by user")
        except Exception as e:
            print(f"\n✗ Error in real-time processing: {e}")
        finally:
            self.is_running = False
    
    def simulate_sensor_stream(self):
        """Simulate real-time sensor data stream"""
        print("🔄 Simulating sensor data stream...")
        
        step_size = int(self.overlap * self.buffer_size)  # 50% overlap
        sample_count = 0
        
        while self.is_running:
            # Generate synthetic sensor data
            current_time = datetime.now()
            
            # Normal background + occasional earthquake simulation
            earthquake_probability = 0.01  # 1% chance per window
            is_earthquake = np.random.random() < earthquake_probability
            
            for i in range(step_size):
                # Timestamp
                timestamp = current_time + timedelta(seconds=i/self.sampling_rate)
                
                if is_earthquake and i > step_size * 0.3:  # Earthquake in middle of window
                    # Simulate earthquake signals
                    acc_x = np.random.normal(0, 0.2) + 3 * np.sin(2 * np.pi * 8 * i/self.sampling_rate)
                    acc_y = np.random.normal(0, 0.2) + 2 * np.sin(2 * np.pi * 5 * i/self.sampling_rate)
                    acc_z = 9.81 + np.random.normal(0, 0.2) + 4 * np.sin(2 * np.pi * 3 * i/self.sampling_rate)
                    gyro_x = np.random.normal(0, 0.1) + 0.5 * np.sin(2 * np.pi * 6 * i/self.sampling_rate)
                    gyro_y = np.random.normal(0, 0.1) + 0.3 * np.sin(2 * np.pi * 7 * i/self.sampling_rate)
                    gyro_z = np.random.normal(0, 0.05) + 0.2 * np.sin(2 * np.pi * 4 * i/self.sampling_rate)
                else:
                    # Normal background noise
                    acc_x = np.random.normal(0, 0.1)
                    acc_y = np.random.normal(0, 0.1)
                    acc_z = 9.81 + np.random.normal(0, 0.1)
                    gyro_x = np.random.normal(0, 0.02)
                    gyro_y = np.random.normal(0, 0.02)
                    gyro_z = np.random.normal(0, 0.01)
                
                # Add data to buffer
                self.add_sensor_data(timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)
                sample_count += 1
            
            # Make prediction if buffer is full
            if len(self.data_buffer['acc_x']) >= self.buffer_size:
                prediction = self.predict_earthquake()
                
                if prediction:
                    status = "🟡 DETECTION" if prediction['prediction'] == 1 else "🟢 Normal"
                    print(f"{status} | Time: {prediction['timestamp'].strftime('%H:%M:%S')} | "
                          f"Probability: {prediction['probability']:.3f} | "
                          f"Confidence: {prediction['confidence']}")
                    
                    # Check for alert
                    if self.check_alert_conditions(prediction):
                        self.trigger_alert(prediction)
                
                # Log prediction
                if prediction:
                    self.log_event({
                        'timestamp': prediction['timestamp'].isoformat(),
                        'probability': prediction['probability'],
                        'prediction': prediction['prediction'],
                        'confidence': prediction['confidence'],
                        'type': 'prediction'
                    })
            
            # Simulate real-time delay
            time.sleep(step_size / self.sampling_rate)
    
    def process_csv_stream(self, csv_file='real_sensor_data.csv'):
        """Process sensor data from CSV file"""
        print(f"📁 Processing sensor data from: {csv_file}")
        
        if not os.path.exists(csv_file):
            print(f"⚠ CSV file not found: {csv_file}")
            return
        
        try:
            # Read CSV data
            df = pd.read_csv(csv_file)
            
            required_cols = ['timestamp', 'acc_x', 'acc_y', 'acc_z']
            if not all(col in df.columns for col in required_cols):
                print(f"⚠ CSV must contain columns: {required_cols}")
                return
            
            print(f"✓ Loaded {len(df)} data points")
            
            # Process data in real-time simulation
            for idx, row in df.iterrows():
                if not self.is_running:
                    break
                
                # Parse timestamp
                timestamp = pd.to_datetime(row['timestamp'])
                
                # Get sensor values
                acc_x = row['acc_x']
                acc_y = row['acc_y']
                acc_z = row['acc_z']
                gyro_x = row.get('gyro_x', 0)
                gyro_y = row.get('gyro_y', 0)
                gyro_z = row.get('gyro_z', 0)
                
                # Add to buffer
                self.add_sensor_data(timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)
                
                # Make prediction when buffer is full
                if len(self.data_buffer['acc_x']) >= self.buffer_size:
                    prediction = self.predict_earthquake()
                    
                    if prediction:
                        status = "🟡 DETECTION" if prediction['prediction'] == 1 else "🟢 Normal"
                        print(f"{status} | Sample: {idx+1}/{len(df)} | "
                              f"Probability: {prediction['probability']:.3f}")
                        
                        # Check for alert
                        if self.check_alert_conditions(prediction):
                            self.trigger_alert(prediction)
                
                # Simulate processing delay
                time.sleep(0.01)  # 10ms delay
                
        except Exception as e:
            print(f"Error processing CSV: {e}")
    
    def get_detection_statistics(self):
        """Get detection statistics"""
        if not self.detection_history:
            return None
        
        total_predictions = len(self.detection_history)
        earthquake_predictions = sum(1 for p in self.detection_history if p['prediction'] == 1)
        
        probabilities = [p['probability'] for p in self.detection_history]
        
        stats = {
            'total_predictions': total_predictions,
            'earthquake_detections': earthquake_predictions,
            'detection_rate': earthquake_predictions / total_predictions,
            'avg_probability': np.mean(probabilities),
            'max_probability': np.max(probabilities),
            'min_probability': np.min(probabilities),
            'recent_detections': earthquake_predictions
        }
        
        return stats
    
    def create_sample_csv(self, filename='sample_real_sensor_data.csv', duration=60):
        """Create sample CSV file for testing"""
        print(f"Creating sample CSV file: {filename}")
        
        # Generate sample data
        sampling_rate = 100
        total_samples = duration * sampling_rate
        
        timestamps = pd.date_range(start='2024-01-01 12:00:00', periods=total_samples, freq='10ms')
        
        # Generate realistic sensor data with some earthquake events
        data = []
        earthquake_events = [
            (20, 25),  # Earthquake from 20-25 seconds
            (45, 48)   # Earthquake from 45-48 seconds
        ]
        
        for i, timestamp in enumerate(timestamps):
            t_seconds = i / sampling_rate
            
            # Check if in earthquake period
            in_earthquake = any(start <= t_seconds <= end for start, end in earthquake_events)
            
            if in_earthquake:
                # Earthquake signals
                acc_x = np.random.normal(0, 0.3) + 5 * np.sin(2 * np.pi * 8 * t_seconds)
                acc_y = np.random.normal(0, 0.3) + 3 * np.sin(2 * np.pi * 6 * t_seconds)
                acc_z = 9.81 + np.random.normal(0, 0.3) + 6 * np.sin(2 * np.pi * 4 * t_seconds)
                gyro_x = np.random.normal(0, 0.1) + 0.8 * np.sin(2 * np.pi * 5 * t_seconds)
                gyro_y = np.random.normal(0, 0.1) + 0.6 * np.sin(2 * np.pi * 7 * t_seconds)
                gyro_z = np.random.normal(0, 0.1) + 0.4 * np.sin(2 * np.pi * 3 * t_seconds)
            else:
                # Normal background
                acc_x = np.random.normal(0, 0.1)
                acc_y = np.random.normal(0, 0.1)
                acc_z = 9.81 + np.random.normal(0, 0.1)
                gyro_x = np.random.normal(0, 0.02)
                gyro_y = np.random.normal(0, 0.02)
                gyro_z = np.random.normal(0, 0.01)
            
            data.append([timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z])
        
        # Create DataFrame and save
        df = pd.DataFrame(data, columns=['timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'])
        df.to_csv(filename, index=False)
        
        print(f"✓ Sample CSV created: {filename}")
        print(f"Duration: {duration}s, Samples: {total_samples}, Earthquake events: {len(earthquake_events)}")
        
        return filename


def main():
    """Main function for real-time earthquake detection"""
    print("🌊 Real-Time Earthquake Detector 🌊")
    print("=" * 50)
    
    # Initialize detector
    detector = RealTimeEarthquakeDetector()
    
    if detector.model is None:
        print("❌ Failed to load model. Please train the model first by running:")
        print("   python sensor_earthquake_ml_pipeline.py")
        return
    
    print("\nDetector initialized successfully!")
    print("Available options:")
    print("1. Simulate real-time sensor stream")
    print("2. Process CSV file")
    print("3. Create sample CSV file")
    print("4. Show detection statistics")
    
    while True:
        try:
            choice = input("\nSelect option (1-4, or 'q' to quit): ").strip()
            
            if choice == 'q':
                break
            elif choice == '1':
                detector.process_real_time_data('simulation')
            elif choice == '2':
                csv_file = input("Enter CSV file path (or press Enter for default): ").strip()
                if not csv_file:
                    csv_file = 'sample_real_sensor_data.csv'
                detector.process_real_time_data('csv')
            elif choice == '3':
                filename = input("Enter filename (or press Enter for default): ").strip()
                if not filename:
                    filename = 'sample_real_sensor_data.csv'
                duration = input("Enter duration in seconds (default 60): ").strip()
                duration = int(duration) if duration else 60
                detector.create_sample_csv(filename, duration)
            elif choice == '4':
                stats = detector.get_detection_statistics()
                if stats:
                    print(f"\n📊 Detection Statistics:")
                    print(f"Total predictions: {stats['total_predictions']}")
                    print(f"Earthquake detections: {stats['earthquake_detections']}")
                    print(f"Detection rate: {stats['detection_rate']:.1%}")
                    print(f"Average probability: {stats['avg_probability']:.3f}")
                    print(f"Max probability: {stats['max_probability']:.3f}")
                else:
                    print("No detection history available")
            else:
                print("Invalid option. Please select 1-4 or 'q'.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
