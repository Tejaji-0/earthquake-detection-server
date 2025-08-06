#!/usr/bin/env python3
"""
Real-time Seismic Data Fetcher for Earthquake Detection
This script fetches real-time seismic data and applies ML models for earthquake detection.
"""

import pandas as pd
import numpy as np
import json
import os
import requests
import time
from datetime import datetime, timedelta
import pickle
import warnings
warnings.filterwarnings('ignore')

class RealTimeEarthquakeDetector:
    def __init__(self, models_dir='ml_models'):
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # If models_dir is relative, make it relative to the script directory
        if not os.path.isabs(models_dir):
            self.models_dir = os.path.join(script_dir, models_dir)
        else:
            self.models_dir = models_dir
            
        self.loaded_models = {}
        self.api_endpoints = {
            'usgs_latest': 'https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_hour.geojson',
            'usgs_significant': 'https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_week.geojson',
            'emsc_latest': 'https://www.seismicportal.eu/fdsnws/event/1/query?format=json&limit=100'
        }
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Earthquake-Detection-System/1.0'})
    
    def load_trained_models(self, task='major_earthquake'):
        """Load pre-trained ML models for earthquake detection."""
        print(f"Loading trained models for task: {task}")
        
        task_dir = os.path.join(self.models_dir, task)
        if not os.path.exists(task_dir):
            print(f"Model directory not found: {task_dir}")
            return False
        
        try:
            # Load metadata
            metadata_file = os.path.join(task_dir, 'metadata.json')
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Load scaler
            scaler_file = os.path.join(task_dir, 'scaler.pkl')
            with open(scaler_file, 'rb') as f:
                scaler = pickle.load(f)
            
            # Load feature selector
            selector_file = os.path.join(task_dir, 'feature_selector.pkl')
            with open(selector_file, 'rb') as f:
                feature_selector = pickle.load(f)
            
            # Load optimized model if available, otherwise load best performing model
            optimized_model_file = os.path.join(task_dir, 'optimized_model.pkl')
            if os.path.exists(optimized_model_file):
                with open(optimized_model_file, 'rb') as f:
                    model = pickle.load(f)
                print("Loaded optimized model")
            else:
                # Find best model based on metadata
                best_model_name = max(metadata['model_performances'].keys(),
                                    key=lambda k: metadata['model_performances'][k]['test_roc_auc'] or 0)
                model_file = os.path.join(task_dir, f'{best_model_name.lower().replace(" ", "_")}_model.pkl')
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
                print(f"Loaded best model: {best_model_name}")
            
            self.loaded_models[task] = {
                'model': model,
                'scaler': scaler,
                'feature_selector': feature_selector,
                'metadata': metadata
            }
            
            print(f"Successfully loaded models for {task}")
            return True
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def fetch_latest_earthquakes(self, source='usgs_latest'):
        """Fetch latest earthquake data from real-time APIs."""
        try:
            url = self.api_endpoints[source]
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if source.startswith('usgs'):
                    return self.parse_usgs_data(data)
                elif source.startswith('emsc'):
                    return self.parse_emsc_data(data)
            else:
                print(f"API request failed: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"Error fetching earthquake data: {e}")
            return []
    
    def parse_usgs_data(self, geojson_data):
        """Parse USGS GeoJSON earthquake data."""
        earthquakes = []
        
        for feature in geojson_data.get('features', []):
            props = feature['properties']
            coords = feature['geometry']['coordinates']
            
            earthquake = {
                'id': props.get('ids', '').split(',')[0] if props.get('ids') else props.get('code', ''),
                'title': props.get('title', ''),
                'magnitude': props.get('mag', 0),
                'date_time': datetime.fromtimestamp(props.get('time', 0) / 1000),
                'latitude': coords[1],
                'longitude': coords[0],
                'depth': coords[2] if len(coords) > 2 else 0,
                'location': props.get('place', ''),
                'significance': props.get('sig', 0),
                'alert': props.get('alert', ''),
                'tsunami': props.get('tsunami', 0),
                'cdi': props.get('cdi', 0),
                'mmi': props.get('mmi', 0),
                'gap': props.get('gap', 0),
                'dmin': props.get('dmin', 0),
                'nst': props.get('nst', 0),
                'magType': props.get('magType', ''),
                'net': props.get('net', ''),
                'source': 'USGS'
            }
            
            earthquakes.append(earthquake)
        
        return earthquakes
    
    def parse_emsc_data(self, json_data):
        """Parse EMSC JSON earthquake data."""
        earthquakes = []
        
        for event in json_data.get('features', []):
            props = event['properties']
            coords = event['geometry']['coordinates']
            
            earthquake = {
                'id': props.get('eventid', ''),
                'title': f"M {props.get('mag', 0)} - {props.get('region', '')}",
                'magnitude': props.get('mag', 0),
                'date_time': datetime.fromisoformat(props.get('time', '').replace('Z', '+00:00')),
                'latitude': coords[1],
                'longitude': coords[0],
                'depth': coords[2] if len(coords) > 2 else 0,
                'location': props.get('region', ''),
                'significance': 0,  # EMSC doesn't provide significance
                'alert': '',
                'tsunami': 0,
                'cdi': 0,
                'mmi': 0,
                'gap': 0,
                'dmin': 0,
                'nst': 0,
                'magType': props.get('magtype', ''),
                'net': 'EMSC',
                'source': 'EMSC'
            }
            
            earthquakes.append(earthquake)
        
        return earthquakes
    
    def engineer_features_for_detection(self, earthquake_data):
        """Engineer features for a single earthquake for real-time detection."""
        features = {}
        
        # Basic magnitude features
        mag = earthquake_data.get('magnitude', 0)
        if mag is None or pd.isna(mag):
            mag = 0
        features['magnitude'] = mag
        features['magnitude_squared'] = mag ** 2
        features['magnitude_log'] = np.log(mag + 1)
        
        # Depth features
        depth = earthquake_data.get('depth', 10)
        if depth is None or pd.isna(depth) or depth <= 0:
            depth = 10  # Default depth
        features['depth'] = depth
        features['depth_log'] = np.log(depth + 1)
        features['depth_normalized'] = depth / 700.0  # Normalize by typical max depth
        
        # Geographic features
        lat = earthquake_data.get('latitude', 0)
        lon = earthquake_data.get('longitude', 0)
        if lat is None or pd.isna(lat):
            lat = 0
        if lon is None or pd.isna(lon):
            lon = 0
        features['latitude'] = lat
        features['longitude'] = lon
        features['lat_abs'] = abs(lat)
        features['lon_abs'] = abs(lon)
        features['distance_from_equator'] = abs(lat)
        features['distance_from_prime_meridian'] = abs(lon)
        
        # Significance and other features
        sig = earthquake_data.get('sig', earthquake_data.get('significance', 0))
        if sig is None or pd.isna(sig):
            sig = 0
        features['significance'] = sig
        features['significance_log'] = np.log(sig + 1)
        
        # Alert encoding (simple mapping)
        alert_map = {'': 0, 'green': 1, 'yellow': 2, 'orange': 3, 'red': 4}
        alert_value = earthquake_data.get('alert', '')
        if pd.isna(alert_value):
            alert_value = ''
        features['alert_encoded'] = alert_map.get(alert_value, 0)
        
        # Tsunami flag
        tsunami = earthquake_data.get('tsunami', 0)
        if tsunami is None or pd.isna(tsunami):
            tsunami = 0
        features['tsunami'] = tsunami
        
        # Network features
        nst = earthquake_data.get('nst', 0)
        if nst is None or pd.isna(nst):
            nst = 0
        features['num_stations'] = nst
        
        gap = earthquake_data.get('gap', 180)
        if gap is None or pd.isna(gap) or gap <= 0:
            gap = 180
        features['gap'] = gap
        
        dmin = earthquake_data.get('dmin', 1.0)
        if dmin is None or pd.isna(dmin) or dmin <= 0:
            dmin = 1.0
        features['dmin'] = dmin
        
        # Temporal features
        dt = earthquake_data.get('date_time')
        if dt is None:
            dt = datetime.now()
        elif isinstance(dt, str):
            dt = pd.to_datetime(dt)
        
        features['year'] = dt.year
        features['month'] = dt.month
        features['day'] = dt.day
        features['hour'] = dt.hour
        features['day_of_year'] = dt.timetuple().tm_yday
        features['day_of_week'] = dt.weekday()
        
        # Seasonal features
        features['sin_month'] = np.sin(2 * np.pi * features['month'] / 12)
        features['cos_month'] = np.cos(2 * np.pi * features['month'] / 12)
        features['sin_hour'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['cos_hour'] = np.cos(2 * np.pi * features['hour'] / 24)
        
        # For real-time detection, we can't calculate recent activity
        # So we'll set these to default values
        features['recent_activity_30d'] = 0
        features['recent_activity_7d'] = 0
        features['magnitude_trend_7d'] = 0
        features['magnitude_trend_30d'] = 0
        
        return features
    
    def predict_earthquake_class(self, earthquake_data, task='major_earthquake'):
        """Predict earthquake classification using trained ML model."""
        if task not in self.loaded_models:
            print(f"Model for task '{task}' not loaded")
            return None
        
        # Engineer features
        features = self.engineer_features_for_detection(earthquake_data)
        
        # Get model components
        model_data = self.loaded_models[task]
        model = model_data['model']
        scaler = model_data['scaler']
        feature_selector = model_data['feature_selector']
        selected_features = model_data['metadata']['selected_features']
        
        # Create feature vector in correct order
        feature_vector = []
        for feature_name in selected_features:
            feature_vector.append(features.get(feature_name, 0))
        
        # Convert to numpy array and reshape
        X = np.array(feature_vector).reshape(1, -1)
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0] if hasattr(model, 'predict_proba') else None
        
        return {
            'prediction': int(prediction),
            'probability': probability.tolist() if probability is not None else None,
            'confidence': max(probability) if probability is not None else None
        }
    
    def monitor_earthquakes(self, interval_minutes=15, duration_hours=24):
        """Monitor earthquakes in real-time and apply ML detection."""
        print("=== Real-time Earthquake Monitoring ===")
        print(f"Monitoring interval: {interval_minutes} minutes")
        print(f"Duration: {duration_hours} hours")
        
        # Load models
        tasks = ['major_earthquake', 'significant_earthquake', 'tsunami_generating']
        loaded_tasks = []
        for task in tasks:
            if self.load_trained_models(task):
                loaded_tasks.append(task)
        
        if not loaded_tasks:
            print("No models loaded successfully. Exiting.")
            return
        
        print(f"Loaded models for: {loaded_tasks}")
        
        # Create output directory for monitoring results
        monitoring_dir = f"monitoring_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(monitoring_dir, exist_ok=True)
        
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)
        
        detected_earthquakes = []
        last_processed_ids = set()
        
        print(f"\nStarting monitoring at {start_time}")
        print("Press Ctrl+C to stop monitoring\n")
        
        try:
            while datetime.now() < end_time:
                current_time = datetime.now()
                print(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] Checking for new earthquakes...")
                
                # Fetch latest earthquakes from multiple sources
                all_earthquakes = []
                for source in ['usgs_latest', 'usgs_significant']:
                    earthquakes = self.fetch_latest_earthquakes(source)
                    all_earthquakes.extend(earthquakes)
                
                # Remove duplicates and process new earthquakes
                new_earthquakes = []
                for eq in all_earthquakes:
                    if eq['id'] not in last_processed_ids:
                        new_earthquakes.append(eq)
                        last_processed_ids.add(eq['id'])
                
                if new_earthquakes:
                    print(f"Found {len(new_earthquakes)} new earthquakes")
                    
                    # Process each new earthquake
                    for eq in new_earthquakes:
                        print(f"  Processing: {eq['title']}")
                        
                        # Apply ML models
                        eq['ml_predictions'] = {}
                        for task in loaded_tasks:
                            prediction = self.predict_earthquake_class(eq, task)
                            eq['ml_predictions'][task] = prediction
                            
                            if prediction and prediction['prediction'] == 1:
                                confidence = prediction['confidence']
                                print(f"    ⚠️  {task}: POSITIVE (confidence: {confidence:.3f})")
                            else:
                                print(f"    ✓ {task}: negative")
                        
                        detected_earthquakes.append(eq)
                        
                        # Check for high-confidence detections
                        high_confidence_alerts = []
                        for task in loaded_tasks:
                            pred = eq['ml_predictions'][task]
                            if pred and pred['prediction'] == 1 and pred['confidence'] > 0.8:
                                high_confidence_alerts.append(task)
                        
                        if high_confidence_alerts:
                            print(f"    🚨 HIGH CONFIDENCE ALERT: {', '.join(high_confidence_alerts)}")
                            self.send_alert(eq, high_confidence_alerts)
                
                else:
                    print("  No new earthquakes found")
                
                # Save monitoring results
                monitoring_file = os.path.join(monitoring_dir, 'monitoring_results.json')
                monitoring_data = {
                    'monitoring_start': start_time.isoformat(),
                    'last_check': current_time.isoformat(),
                    'total_earthquakes_processed': len(detected_earthquakes),
                    'loaded_models': loaded_tasks,
                    'earthquakes': detected_earthquakes
                }
                
                with open(monitoring_file, 'w') as f:
                    json.dump(monitoring_data, f, indent=2, default=str)
                
                # Wait for next check
                print(f"  Waiting {interval_minutes} minutes until next check...\n")
                time.sleep(interval_minutes * 60)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        
        print(f"Monitoring completed. Results saved to: {monitoring_dir}")
        return detected_earthquakes
    
    def send_alert(self, earthquake, alert_types):
        """Send alert for high-confidence earthquake detection."""
        # In a real system, this would send notifications via email, SMS, etc.
        print(f"\n{'='*60}")
        print("🚨 EARTHQUAKE ALERT 🚨")
        print(f"Event: {earthquake['title']}")
        print(f"Magnitude: {earthquake['magnitude']}")
        print(f"Location: {earthquake['location']}")
        print(f"Time: {earthquake['date_time']}")
        print(f"Alert Types: {', '.join(alert_types)}")
        print(f"{'='*60}\n")
        
        # Save alert to file
        alert_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        alert_file = f"earthquake_alert_{alert_time}.json"
        
        alert_data = {
            'alert_time': datetime.now().isoformat(),
            'earthquake': earthquake,
            'alert_types': alert_types
        }
        
        with open(alert_file, 'w') as f:
            json.dump(alert_data, f, indent=2, default=str)
        
        print(f"Alert saved to: {alert_file}")
    
    def batch_predict_csv(self, csv_file, output_file=None):
        """Apply ML models to a batch of earthquakes from CSV file."""
        print(f"Processing batch predictions for: {csv_file}")
        
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # If csv_file is relative, make it relative to the script directory
        if not os.path.isabs(csv_file):
            csv_file = os.path.join(script_dir, csv_file)
        
        if not os.path.exists(csv_file):
            print(f"CSV file not found: {csv_file}")
            return
        
        # Load earthquake data
        df = pd.read_csv(csv_file)
        df['date_time'] = pd.to_datetime(df['date_time'], format='%d-%m-%Y %H:%M')
        
        # Load models
        tasks = ['major_earthquake', 'significant_earthquake', 'tsunami_generating']
        loaded_tasks = []
        for task in tasks:
            if self.load_trained_models(task):
                loaded_tasks.append(task)
        
        if not loaded_tasks:
            print("No models loaded successfully.")
            return
        
        print(f"Loaded models for: {loaded_tasks}")
        
        # Process each earthquake
        results = []
        for idx, row in df.iterrows():
            earthquake_data = row.to_dict()
            
            # Apply ML models
            predictions = {}
            for task in loaded_tasks:
                prediction = self.predict_earthquake_class(earthquake_data, task)
                predictions[task] = prediction
            
            result = earthquake_data.copy()
            result['ml_predictions'] = predictions
            results.append(result)
            
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(df)} earthquakes")
        
        # Save results
        if output_file is None:
            output_file = f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Batch predictions saved to: {output_file}")
        
        # Print summary
        print(f"\n=== Batch Prediction Summary ===")
        for task in loaded_tasks:
            positive_predictions = sum(1 for r in results 
                                     if r['ml_predictions'][task]['prediction'] == 1)
            print(f"{task}: {positive_predictions}/{len(results)} positive predictions "
                  f"({positive_predictions/len(results)*100:.1f}%)")

def main():
    """Main function for real-time earthquake detection."""
    detector = RealTimeEarthquakeDetector()
    
    print("=== Earthquake Detection System ===")
    print("Choose an option:")
    print("1. Real-time monitoring")
    print("2. Batch prediction from CSV")
    print("3. Test latest earthquakes")
    
    try:
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == '1':
            # Real-time monitoring
            interval = int(input("Monitoring interval (minutes, default 15): ") or 15)
            duration = int(input("Duration (hours, default 24): ") or 24)
            detector.monitor_earthquakes(interval, duration)
            
        elif choice == '2':
            # Batch prediction
            csv_file = input("Enter CSV file path (default: earthquake_1995-2023.csv): ") or "earthquake_1995-2023.csv"
            detector.batch_predict_csv(csv_file)
                
        elif choice == '3':
            # Test latest earthquakes
            print("Fetching latest earthquakes...")
            earthquakes = detector.fetch_latest_earthquakes()
            print(f"Found {len(earthquakes)} recent earthquakes")
            
            if earthquakes:
                # Load models and test on first few earthquakes
                tasks = ['major_earthquake', 'significant_earthquake', 'tsunami_generating']
                for task in tasks:
                    detector.load_trained_models(task)
                
                for i, eq in enumerate(earthquakes[:5]):  # Test first 5
                    print(f"\nTesting earthquake {i+1}: {eq['title']}")
                    for task in tasks:
                        if task in detector.loaded_models:
                            prediction = detector.predict_earthquake_class(eq, task)
                            if prediction:
                                pred_label = "POSITIVE" if prediction['prediction'] == 1 else "negative"
                                conf = prediction['confidence']
                                print(f"  {task}: {pred_label} (confidence: {conf:.3f})")
        
        else:
            print("Invalid choice")
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
