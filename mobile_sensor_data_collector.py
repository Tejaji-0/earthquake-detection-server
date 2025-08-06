#!/usr/bin/env python3
"""
Mobile Sensor Data Collector
This script helps collect accelerometer and gyroscope data from various sources
including mobile phones, web APIs, and sensor datasets.
"""

import requests
import pandas as pd
import numpy as np
import json
import os
import time
from datetime import datetime, timedelta
import zipfile
import io
from urllib.parse import urljoin
import warnings
warnings.filterwarnings('ignore')

class MobileSensorDataCollector:
    def __init__(self, output_dir='collected_sensor_data'):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        print(f"Mobile Sensor Data Collector initialized")
        print(f"Output directory: {self.output_dir}")
    
    def download_earthquake_accelerometer_datasets(self):
        """Download publicly available earthquake accelerometer datasets"""
        print("\n=== Downloading Earthquake Accelerometer Datasets ===")
        
        datasets = []
        
        # Try to download from various earthquake data sources
        try:
            # 1. Strong Motion Database (simulated URLs - replace with real ones)
            print("Attempting to download from Strong Motion databases...")
            
            # CESMD (Center for Engineering Strong Motion Data) - example structure
            cesmd_events = [
                "19940117123055",  # Northridge earthquake
                "19891018000415",  # Loma Prieta earthquake
                "20190706033419",  # Ridgecrest earthquake
            ]
            
            for event_id in cesmd_events:
                try:
                    # This is a placeholder - actual CESMD API would be different
                    print(f"  Searching for event: {event_id}")
                    # In real implementation, you'd use CESMD API here
                    datasets.append(f"cesmd_event_{event_id}")
                except Exception as e:
                    print(f"    Could not download {event_id}: {e}")
            
            # 2. PEER Strong Motion Database
            print("Attempting PEER database access...")
            # peer_url = "https://ngawest2.berkeley.edu/api/..."  # Example
            
        except Exception as e:
            print(f"Error accessing earthquake databases: {e}")
        
        # Create sample earthquake data since real APIs require authentication
        self.create_sample_earthquake_accelerometer_data()
        
        return datasets
    
    def create_sample_earthquake_accelerometer_data(self):
        """Create realistic earthquake accelerometer data based on real earthquake characteristics"""
        print("\n=== Creating Sample Earthquake Accelerometer Data ===")
        
        # Based on real earthquake characteristics
        earthquake_events = [
            {
                "name": "Northridge_1994_like",
                "magnitude": 6.7,
                "depth": 18.4,
                "duration": 45,
                "peak_acceleration": 8.5,  # m/s²
                "dominant_frequency": 5.2,  # Hz
                "characteristics": "thrust_fault"
            },
            {
                "name": "Loma_Prieta_1989_like", 
                "magnitude": 6.9,
                "depth": 15.0,
                "duration": 60,
                "peak_acceleration": 4.7,
                "dominant_frequency": 3.8,
                "characteristics": "strike_slip"
            },
            {
                "name": "Tohoku_2011_like",
                "magnitude": 9.1,
                "depth": 32.0,
                "duration": 180,
                "peak_acceleration": 12.3,
                "dominant_frequency": 2.1,
                "characteristics": "megathrust"
            },
            {
                "name": "Turkey_2023_like",
                "magnitude": 7.8,
                "depth": 17.9,
                "duration": 90,
                "peak_acceleration": 15.2,
                "dominant_frequency": 4.5,
                "characteristics": "strike_slip"
            }
        ]
        
        for event in earthquake_events:
            print(f"  Creating data for: {event['name']} (M{event['magnitude']})")
            
            sampling_rate = 100  # Hz
            duration = event["duration"]
            total_samples = duration * sampling_rate
            
            # Time vector
            t = np.linspace(0, duration, total_samples)
            
            # Initialize with background noise
            noise_level = 0.05
            acc_x = np.random.normal(0, noise_level, total_samples)
            acc_y = np.random.normal(0, noise_level, total_samples)
            acc_z = np.random.normal(9.81, noise_level, total_samples)
            
            # P-wave arrival (first 10-20% of duration)
            p_start = int(0.1 * total_samples)
            p_end = int(0.3 * total_samples)
            p_freq = event["dominant_frequency"] * 2  # P-waves higher frequency
            p_amplitude = event["peak_acceleration"] * 0.3  # P-waves lower amplitude
            
            # S-wave arrival (main shock, 30-80% of duration)
            s_start = int(0.3 * total_samples)
            s_end = int(0.8 * total_samples)
            s_freq = event["dominant_frequency"]
            s_amplitude = event["peak_acceleration"]
            
            # Add P-wave
            for i in range(p_start, p_end):
                if i < total_samples:
                    # Envelope function for realistic wave decay
                    envelope = np.exp(-(i - p_start) / (p_end - p_start) * 2)
                    p_signal = p_amplitude * envelope * np.sin(2 * np.pi * p_freq * t[i])
                    
                    acc_x[i] += p_signal * 0.8
                    acc_y[i] += p_signal * 0.6
                    acc_z[i] += p_signal * 0.4
            
            # Add S-wave (main shaking)
            for i in range(s_start, s_end):
                if i < total_samples:
                    # Complex envelope with buildup and decay
                    buildup = min(1.0, (i - s_start) / (0.2 * (s_end - s_start)))
                    decay = np.exp(-(i - s_start) / (s_end - s_start) * 1.5)
                    envelope = buildup * decay
                    
                    # Multiple frequency components
                    s_signal = (s_amplitude * envelope * np.sin(2 * np.pi * s_freq * t[i]) +
                               s_amplitude * 0.5 * envelope * np.sin(2 * np.pi * s_freq * 0.7 * t[i]) +
                               s_amplitude * 0.3 * envelope * np.sin(2 * np.pi * s_freq * 1.3 * t[i]))
                    
                    # Different amplitudes for different axes
                    acc_x[i] += s_signal * 1.0
                    acc_y[i] += s_signal * 0.8
                    acc_z[i] += s_signal * 0.6
            
            # Add surface waves (long-period, end of record)
            surface_start = int(0.7 * total_samples)
            surface_freq = event["dominant_frequency"] * 0.5
            surface_amplitude = event["peak_acceleration"] * 0.4
            
            for i in range(surface_start, total_samples):
                envelope = np.exp(-(i - surface_start) / (total_samples - surface_start) * 1.0)
                surface_signal = surface_amplitude * envelope * np.sin(2 * np.pi * surface_freq * t[i])
                
                acc_x[i] += surface_signal * 0.6
                acc_y[i] += surface_signal * 0.8
                acc_z[i] += surface_signal * 1.0
            
            # Generate realistic gyroscope data (correlated with acceleration changes)
            gyro_x = np.gradient(acc_y) * 2 + np.random.normal(0, 0.02, total_samples)
            gyro_y = -np.gradient(acc_x) * 2 + np.random.normal(0, 0.02, total_samples)
            gyro_z = np.gradient(acc_z) * 1 + np.random.normal(0, 0.01, total_samples)
            
            # Create timestamp
            start_time = datetime(2024, 1, 1, 12, 0, 0)
            timestamps = [start_time + timedelta(seconds=i/sampling_rate) for i in range(total_samples)]
            
            # Create DataFrame
            earthquake_data = pd.DataFrame({
                'timestamp': timestamps,
                'acc_x': acc_x,
                'acc_y': acc_y,
                'acc_z': acc_z,
                'gyro_x': gyro_x,
                'gyro_y': gyro_y,
                'gyro_z': gyro_z,
                'event_name': event['name'],
                'magnitude': event['magnitude'],
                'depth_km': event['depth'],
                'label': 1  # Earthquake
            })
            
            # Save to file
            filename = f"earthquake_{event['name']}.csv"
            filepath = os.path.join(self.output_dir, filename)
            earthquake_data.to_csv(filepath, index=False)
            
            print(f"    ✓ Saved: {filename} ({len(earthquake_data)} samples, {duration}s)")
        
        print(f"✓ Created {len(earthquake_events)} earthquake accelerometer datasets")
    
    def create_normal_background_data(self):
        """Create normal background accelerometer data (non-earthquake)"""
        print("\n=== Creating Normal Background Data ===")
        
        scenarios = [
            {"name": "building_ambient", "description": "Normal building vibrations", "duration": 300},
            {"name": "traffic_vibrations", "description": "Traffic-induced vibrations", "duration": 240},
            {"name": "construction_nearby", "description": "Nearby construction activity", "duration": 180},
            {"name": "quiet_environment", "description": "Very quiet environment", "duration": 600},
            {"name": "windy_conditions", "description": "Wind-induced building sway", "duration": 300},
            {"name": "machinery_operations", "description": "Industrial machinery vibrations", "duration": 200}
        ]
        
        for scenario in scenarios:
            print(f"  Creating: {scenario['name']}")
            
            sampling_rate = 100
            duration = scenario["duration"]
            total_samples = duration * sampling_rate
            t = np.linspace(0, duration, total_samples)
            
            if scenario["name"] == "building_ambient":
                # Typical building natural frequency around 0.1-1 Hz
                acc_x = 0.02 * np.sin(2 * np.pi * 0.3 * t) + np.random.normal(0, 0.05, total_samples)
                acc_y = 0.03 * np.sin(2 * np.pi * 0.25 * t) + np.random.normal(0, 0.05, total_samples)
                acc_z = 9.81 + 0.01 * np.sin(2 * np.pi * 0.2 * t) + np.random.normal(0, 0.03, total_samples)
                
            elif scenario["name"] == "traffic_vibrations":
                # Traffic frequencies typically 2-20 Hz
                acc_x = (0.1 * np.sin(2 * np.pi * 8 * t) + 0.05 * np.sin(2 * np.pi * 15 * t) + 
                        np.random.normal(0, 0.08, total_samples))
                acc_y = (0.08 * np.sin(2 * np.pi * 12 * t) + 0.03 * np.sin(2 * np.pi * 6 * t) + 
                        np.random.normal(0, 0.06, total_samples))
                acc_z = 9.81 + 0.05 * np.sin(2 * np.pi * 10 * t) + np.random.normal(0, 0.04, total_samples)
                
            elif scenario["name"] == "construction_nearby":
                # Construction activities - intermittent high frequency
                base_x = np.random.normal(0, 0.1, total_samples)
                base_y = np.random.normal(0, 0.1, total_samples)
                base_z = 9.81 + np.random.normal(0, 0.05, total_samples)
                
                # Add construction bursts
                for burst_start in range(0, total_samples, 500):
                    burst_end = min(burst_start + 200, total_samples)
                    burst_freq = np.random.uniform(20, 50)
                    burst_amp = np.random.uniform(0.5, 1.5)
                    
                    for i in range(burst_start, burst_end):
                        envelope = np.exp(-(i - burst_start) / 100)
                        signal = burst_amp * envelope * np.sin(2 * np.pi * burst_freq * t[i])
                        base_x[i] += signal * 0.8
                        base_y[i] += signal * 0.6
                        base_z[i] += signal * 0.4
                
                acc_x, acc_y, acc_z = base_x, base_y, base_z
                
            elif scenario["name"] == "quiet_environment":
                # Very low noise environment
                acc_x = np.random.normal(0, 0.02, total_samples)
                acc_y = np.random.normal(0, 0.02, total_samples) 
                acc_z = 9.81 + np.random.normal(0, 0.01, total_samples)
                
            elif scenario["name"] == "windy_conditions":
                # Building sway due to wind (very low frequency)
                acc_x = 0.05 * np.sin(2 * np.pi * 0.1 * t) + 0.03 * np.sin(2 * np.pi * 0.15 * t) + np.random.normal(0, 0.04, total_samples)
                acc_y = 0.04 * np.sin(2 * np.pi * 0.12 * t) + 0.02 * np.sin(2 * np.pi * 0.08 * t) + np.random.normal(0, 0.04, total_samples)
                acc_z = 9.81 + 0.02 * np.sin(2 * np.pi * 0.09 * t) + np.random.normal(0, 0.02, total_samples)
                
            else:  # machinery_operations
                # Regular machinery vibrations
                acc_x = 0.3 * np.sin(2 * np.pi * 25 * t) + 0.1 * np.sin(2 * np.pi * 50 * t) + np.random.normal(0, 0.1, total_samples)
                acc_y = 0.2 * np.sin(2 * np.pi * 30 * t) + 0.05 * np.sin(2 * np.pi * 60 * t) + np.random.normal(0, 0.08, total_samples)
                acc_z = 9.81 + 0.15 * np.sin(2 * np.pi * 35 * t) + np.random.normal(0, 0.06, total_samples)
            
            # Generate correlated gyroscope data
            gyro_x = np.gradient(acc_y) * 0.5 + np.random.normal(0, 0.01, total_samples)
            gyro_y = -np.gradient(acc_x) * 0.5 + np.random.normal(0, 0.01, total_samples)
            gyro_z = np.gradient(acc_z) * 0.3 + np.random.normal(0, 0.005, total_samples)
            
            # Create timestamps
            start_time = datetime(2024, 1, 1, 12, 0, 0)
            timestamps = [start_time + timedelta(seconds=i/sampling_rate) for i in range(total_samples)]
            
            # Create DataFrame
            normal_data = pd.DataFrame({
                'timestamp': timestamps,
                'acc_x': acc_x,
                'acc_y': acc_y,
                'acc_z': acc_z,
                'gyro_x': gyro_x,
                'gyro_y': gyro_y,
                'gyro_z': gyro_z,
                'scenario': scenario['name'],
                'description': scenario['description'],
                'label': 0  # Normal (non-earthquake)
            })
            
            # Save to file
            filename = f"normal_{scenario['name']}.csv"
            filepath = os.path.join(self.output_dir, filename)
            normal_data.to_csv(filepath, index=False)
            
            print(f"    ✓ Saved: {filename} ({len(normal_data)} samples, {duration}s)")
        
        print(f"✓ Created {len(scenarios)} normal background datasets")
    
    def create_mobile_phone_simulation_data(self):
        """Create data that simulates mobile phone accelerometer readings"""
        print("\n=== Creating Mobile Phone Simulation Data ===")
        
        phone_scenarios = [
            {"name": "phone_on_table", "movement": "minimal", "duration": 120},
            {"name": "phone_in_pocket_walking", "movement": "walking", "duration": 180},
            {"name": "phone_in_car", "movement": "driving", "duration": 300},
            {"name": "phone_on_desk_typing", "movement": "desk_vibrations", "duration": 240},
            {"name": "phone_stationary_building", "movement": "building_motion", "duration": 600}
        ]
        
        for scenario in phone_scenarios:
            print(f"  Creating: {scenario['name']}")
            
            sampling_rate = 50  # Typical mobile phone sampling rate
            duration = scenario["duration"]
            total_samples = duration * sampling_rate
            t = np.linspace(0, duration, total_samples)
            
            if scenario["movement"] == "minimal":
                # Phone lying flat on table
                acc_x = np.random.normal(0, 0.1, total_samples)
                acc_y = np.random.normal(0, 0.1, total_samples)
                acc_z = np.random.normal(9.81, 0.1, total_samples)
                
            elif scenario["movement"] == "walking":
                # Walking pattern ~1.5-2 Hz
                step_freq = 1.8
                acc_x = 2 * np.sin(2 * np.pi * step_freq * t) + np.random.normal(0, 0.5, total_samples)
                acc_y = 1.5 * np.sin(2 * np.pi * step_freq * t + np.pi/3) + np.random.normal(0, 0.4, total_samples)
                acc_z = 9.81 + 3 * np.sin(2 * np.pi * step_freq * t + np.pi/2) + np.random.normal(0, 0.6, total_samples)
                
            elif scenario["movement"] == "driving":
                # Car vibrations and turns
                engine_freq = 25  # Engine vibration
                acc_x = (0.3 * np.sin(2 * np.pi * engine_freq * t) + 
                        0.5 * np.sin(2 * np.pi * 0.1 * t) +  # Gentle turns
                        np.random.normal(0, 0.4, total_samples))
                acc_y = (0.2 * np.sin(2 * np.pi * engine_freq * t + np.pi/4) + 
                        0.8 * np.sin(2 * np.pi * 0.08 * t) +  # Lane changes
                        np.random.normal(0, 0.3, total_samples))
                acc_z = (9.81 + 0.4 * np.sin(2 * np.pi * engine_freq * t) + 
                        0.3 * np.sin(2 * np.pi * 0.05 * t) +  # Road bumps
                        np.random.normal(0, 0.2, total_samples))
                
            elif scenario["movement"] == "desk_vibrations":
                # Typing and desk movements
                acc_x = 0.2 * np.sin(2 * np.pi * 3 * t) + np.random.normal(0, 0.15, total_samples)
                acc_y = 0.15 * np.sin(2 * np.pi * 2.5 * t) + np.random.normal(0, 0.12, total_samples)
                acc_z = 9.81 + 0.1 * np.sin(2 * np.pi * 4 * t) + np.random.normal(0, 0.08, total_samples)
                
            else:  # building_motion
                # Phone stationary in building
                acc_x = 0.05 * np.sin(2 * np.pi * 0.3 * t) + np.random.normal(0, 0.08, total_samples)
                acc_y = 0.04 * np.sin(2 * np.pi * 0.25 * t) + np.random.normal(0, 0.08, total_samples)
                acc_z = 9.81 + 0.03 * np.sin(2 * np.pi * 0.2 * t) + np.random.normal(0, 0.05, total_samples)
            
            # Generate gyroscope data (phones have gyroscopes)
            gyro_x = np.gradient(acc_y) * 0.3 + np.random.normal(0, 0.02, total_samples)
            gyro_y = -np.gradient(acc_x) * 0.3 + np.random.normal(0, 0.02, total_samples)
            gyro_z = np.gradient(acc_z) * 0.2 + np.random.normal(0, 0.01, total_samples)
            
            # Create timestamps
            start_time = datetime(2024, 1, 1, 12, 0, 0)
            timestamps = [start_time + timedelta(seconds=i/sampling_rate) for i in range(total_samples)]
            
            # Create DataFrame
            phone_data = pd.DataFrame({
                'timestamp': timestamps,
                'acc_x': acc_x,
                'acc_y': acc_y,
                'acc_z': acc_z,
                'gyro_x': gyro_x,
                'gyro_y': gyro_y,
                'gyro_z': gyro_z,
                'device_type': 'mobile_phone',
                'scenario': scenario['name'],
                'sampling_rate': sampling_rate,
                'label': 0  # Normal (non-earthquake)
            })
            
            # Save to file
            filename = f"mobile_{scenario['name']}.csv"
            filepath = os.path.join(self.output_dir, filename)
            phone_data.to_csv(filepath, index=False)
            
            print(f"    ✓ Saved: {filename} ({len(phone_data)} samples, {duration}s)")
        
        print(f"✓ Created {len(phone_scenarios)} mobile phone simulation datasets")
    
    def download_public_datasets(self):
        """Download publicly available accelerometer datasets"""
        print("\n=== Downloading Public Accelerometer Datasets ===")
        
        try:
            # UCI Human Activity Recognition Dataset
            print("Downloading UCI HAR Dataset...")
            uci_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
            
            response = requests.get(uci_url, timeout=30)
            if response.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                    zip_file.extractall(self.output_dir)
                print("✓ UCI HAR Dataset downloaded successfully")
            else:
                print(f"⚠ Could not download UCI HAR Dataset: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"⚠ Error downloading UCI HAR Dataset: {e}")
        
        # Try other public datasets
        public_datasets = [
            {
                "name": "WISDM Activity Recognition",
                "description": "Smartphone accelerometer data for activity recognition",
                "note": "Available at: https://www.cis.fordham.edu/wisdm/dataset.php"
            },
            {
                "name": "Opportunity Dataset", 
                "description": "Multi-sensor data including accelerometers",
                "note": "Available at: https://archive.ics.uci.edu/ml/datasets/opportunity+activity+recognition"
            },
            {
                "name": "PAMAP2 Dataset",
                "description": "Physical activity monitoring with accelerometers",
                "note": "Available at: https://archive.ics.uci.edu/ml/datasets/pamap2+physical+activity+monitoring"
            }
        ]
        
        print("\n📚 Additional Public Datasets Available:")
        for dataset in public_datasets:
            print(f"  • {dataset['name']}: {dataset['description']}")
            print(f"    {dataset['note']}")
        
        return public_datasets
    
    def create_data_collection_guide(self):
        """Create a guide for collecting real sensor data"""
        print("\n=== Creating Data Collection Guide ===")
        
        guide_content = """# Real Sensor Data Collection Guide

## 📱 Mobile Phone Data Collection

### Android Apps:
1. **Sensor Kinetics** - Real-time sensor data logging
2. **Accelerometer Monitor** - Simple data export
3. **Physics Toolbox Sensor Suite** - Comprehensive sensor logging
4. **Sensor Data Collector** - CSV export functionality

### iOS Apps:
1. **SensorLog** - Professional sensor data logging
2. **Accelerometer** - Basic acceleration monitoring
3. **Motion Data** - Export accelerometer/gyroscope data

### Web-based Collection:
Create a simple HTML page with JavaScript to collect data:

```html
<!DOCTYPE html>
<html>
<head><title>Sensor Data Collector</title></head>
<body>
    <button onclick="startCollection()">Start Collecting</button>
    <button onclick="stopCollection()">Stop</button>
    <button onclick="downloadData()">Download CSV</button>
    
    <script>
    let sensorData = [];
    let collecting = false;
    
    function startCollection() {
        collecting = true;
        
        window.addEventListener('devicemotion', function(event) {
            if (collecting) {
                sensorData.push({
                    timestamp: new Date().toISOString(),
                    acc_x: event.acceleration.x,
                    acc_y: event.acceleration.y,
                    acc_z: event.acceleration.z,
                    gyro_x: event.rotationRate.alpha,
                    gyro_y: event.rotationRate.beta,
                    gyro_z: event.rotationRate.gamma
                });
            }
        });
    }
    
    function downloadData() {
        let csv = 'timestamp,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z\\n';
        sensorData.forEach(row => {
            csv += `${row.timestamp},${row.acc_x},${row.acc_y},${row.acc_z},${row.gyro_x},${row.gyro_y},${row.gyro_z}\\n`;
        });
        
        let blob = new Blob([csv], { type: 'text/csv' });
        let url = window.URL.createObjectURL(blob);
        let a = document.createElement('a');
        a.setAttribute('hidden', '');
        a.setAttribute('href', url);
        a.setAttribute('download', 'sensor_data.csv');
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }
    </script>
</body>
</html>
```

## 🏢 Building Sensor Networks

### IoT Sensor Nodes:
1. **Arduino + MPU6050** - Basic accelerometer/gyroscope
2. **Raspberry Pi + ADXL345** - More processing power
3. **ESP32 + MPU9250** - WiFi-enabled sensor node
4. **Seeed Studio Grove sensors** - Plug-and-play options

### Professional Seismic Stations:
1. **IRIS Consortium** - Global seismic data
2. **USGS Earthquake Hazards Program** - US seismic data
3. **European-Mediterranean Seismological Centre** - EU data
4. **Japan Meteorological Agency** - High-quality earthquake data

## 🌐 API Data Sources

### Real-time Earthquake Data:
- USGS Earthquake API: https://earthquake.usgs.gov/fdsnws/event/1/
- EMSC Real-time: https://www.seismicportal.eu/
- ISC Bulletin: http://www.isc.ac.uk/

### Strong Motion Data:
- CESMD: https://strongmotioncenter.org/
- PEER NGA Database: https://ngawest2.berkeley.edu/
- ESM Database: https://esm.mi.ingv.it/

## 📊 Data Collection Best Practices

### For Earthquake Detection:
1. **Sampling Rate**: 100+ Hz for earthquake signals
2. **Duration**: Minimum 10-second windows
3. **Location**: Fixed position, stable mounting
4. **Calibration**: Known reference (gravity = 9.81 m/s²)
5. **Multiple Sensors**: Different locations for validation

### Data Quality:
- Consistent sampling rate
- Proper timestamp format
- Sensor orientation documentation
- Environmental conditions notes
- Battery level monitoring

### Labeling Guidelines:
- 0 = Normal conditions
- 1 = Earthquake event
- Include magnitude, distance, depth when available
- Note other vibration sources (traffic, construction, etc.)

## 🔧 Hardware Recommendations

### Budget Option ($20-50):
- Arduino Nano + MPU6050
- SD card module for data logging
- Battery pack for portable operation

### Professional Option ($200-500):
- Raspberry Pi 4 + high-quality MEMS sensor
- GPS module for precise timing
- 4G/WiFi connectivity for real-time transmission
- Weather-resistant enclosure

### Research Grade ($1000+):
- Dedicated seismograph
- Broadband seismometer
- Professional data acquisition system
"""
        
        guide_file = os.path.join(self.output_dir, "DATA_COLLECTION_GUIDE.md")
        with open(guide_file, 'w') as f:
            f.write(guide_content)
        
        print(f"✓ Data collection guide saved: {guide_file}")
    
    def create_dataset_summary(self):
        """Create summary of all collected/created datasets"""
        print("\n=== Creating Dataset Summary ===")
        
        # List all files in output directory
        files = [f for f in os.listdir(self.output_dir) if f.endswith('.csv')]
        
        summary = {
            "collection_date": datetime.now().isoformat(),
            "total_files": len(files),
            "earthquake_datasets": [],
            "normal_datasets": [],
            "mobile_datasets": [],
            "total_samples": 0,
            "total_duration_hours": 0
        }
        
        for file in files:
            filepath = os.path.join(self.output_dir, file)
            try:
                df = pd.read_csv(filepath)
                
                file_info = {
                    "filename": file,
                    "samples": len(df),
                    "duration_seconds": len(df) / 100,  # Assuming 100Hz
                    "has_gyroscope": all(col in df.columns for col in ['gyro_x', 'gyro_y', 'gyro_z']),
                    "label": df['label'].iloc[0] if 'label' in df.columns else 'unknown'
                }
                
                summary["total_samples"] += file_info["samples"]
                summary["total_duration_hours"] += file_info["duration_seconds"] / 3600
                
                if 'earthquake' in file:
                    summary["earthquake_datasets"].append(file_info)
                elif 'mobile' in file:
                    summary["mobile_datasets"].append(file_info)
                elif 'normal' in file:
                    summary["normal_datasets"].append(file_info)
                    
            except Exception as e:
                print(f"⚠ Error reading {file}: {e}")
        
        # Save summary
        summary_file = os.path.join(self.output_dir, "dataset_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create human-readable summary
        readable_file = os.path.join(self.output_dir, "DATASET_SUMMARY.md")
        with open(readable_file, 'w') as f:
            f.write("# Collected Sensor Dataset Summary\n\n")
            f.write(f"**Collection Date:** {summary['collection_date']}\n")
            f.write(f"**Total Files:** {summary['total_files']}\n")
            f.write(f"**Total Samples:** {summary['total_samples']:,}\n")
            f.write(f"**Total Duration:** {summary['total_duration_hours']:.1f} hours\n\n")
            
            f.write("## Earthquake Datasets\n")
            for dataset in summary["earthquake_datasets"]:
                f.write(f"- **{dataset['filename']}**: {dataset['samples']:,} samples, {dataset['duration_seconds']:.0f}s\n")
            
            f.write("\n## Normal Datasets\n")
            for dataset in summary["normal_datasets"]:
                f.write(f"- **{dataset['filename']}**: {dataset['samples']:,} samples, {dataset['duration_seconds']:.0f}s\n")
            
            f.write("\n## Mobile Datasets\n")
            for dataset in summary["mobile_datasets"]:
                f.write(f"- **{dataset['filename']}**: {dataset['samples']:,} samples, {dataset['duration_seconds']:.0f}s\n")
        
        print(f"✓ Dataset summary saved: {summary_file}")
        print(f"✓ Readable summary saved: {readable_file}")
        
        return summary
    
    def collect_all_data(self):
        """Run complete data collection process"""
        print("🌊 MOBILE SENSOR DATA COLLECTION 🌊")
        print("=" * 50)
        
        # Download public datasets
        self.download_public_datasets()
        
        # Create earthquake datasets
        self.download_earthquake_accelerometer_datasets()
        
        # Create normal background data
        self.create_normal_background_data()
        
        # Create mobile phone simulation data
        self.create_mobile_phone_simulation_data()
        
        # Create data collection guide
        self.create_data_collection_guide()
        
        # Create dataset summary
        summary = self.create_dataset_summary()
        
        print("\n" + "=" * 50)
        print("🎉 DATA COLLECTION COMPLETED! 🎉")
        print("=" * 50)
        
        print(f"\n📊 Collection Summary:")
        print(f"  • Total files: {summary['total_files']}")
        print(f"  • Total samples: {summary['total_samples']:,}")
        print(f"  • Total duration: {summary['total_duration_hours']:.1f} hours")
        print(f"  • Earthquake datasets: {len(summary['earthquake_datasets'])}")
        print(f"  • Normal datasets: {len(summary['normal_datasets'])}")
        print(f"  • Mobile datasets: {len(summary['mobile_datasets'])}")
        
        print(f"\n📁 All data saved to: {self.output_dir}")
        print(f"📖 Check DATA_COLLECTION_GUIDE.md for real sensor data collection")
        
        return summary


def main():
    """Main function to collect sensor data"""
    collector = MobileSensorDataCollector()
    
    print("Mobile Sensor Data Collector")
    print("This tool helps you collect accelerometer and gyroscope data for earthquake detection")
    
    # Run complete data collection
    summary = collector.collect_all_data()
    
    print(f"\n✅ Data collection completed!")
    print(f"🎯 Next steps:")
    print(f"  1. Review the collected datasets in '{collector.output_dir}'")
    print(f"  2. Train ML models using: python sensor_earthquake_ml_pipeline.py")
    print(f"  3. Collect real sensor data using the guide in DATA_COLLECTION_GUIDE.md")
    print(f"  4. Replace synthetic data with real data for better accuracy")


if __name__ == "__main__":
    main()
