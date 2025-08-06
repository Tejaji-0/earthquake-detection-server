#!/usr/bin/env python3
"""
Enhanced Machine Learning Pipeline for Earthquake Detection
This script combines seismic waveform data with earthquake catalog data to train 
more sophisticated ML models for earthquake detection and prediction.
"""

import pandas as pd
import numpy as np
import json
import os
import pickle
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import requests
import time

# Seismic data processing
try:
    from obspy import read, UTCDateTime
    from obspy.signal import filter
    from obspy.signal.trigger import classic_sta_lta, trigger_onset
    from scipy.signal import welch
    OBSPY_AVAILABLE = True
    print("✓ ObsPy available for seismic data processing")
except ImportError:
    OBSPY_AVAILABLE = False
    print("⚠ ObsPy not available - seismic features will be limited")

warnings.filterwarnings('ignore')

class EnhancedEarthquakeMLPipeline:
    def __init__(self, 
                 earthquake_csv='data/database.csv', 
                 seismic_data_dir='earthquake_seismic_data',
                 output_dir='enhanced_ml_models'):
        
        self.earthquake_csv = earthquake_csv
        self.seismic_data_dir = seismic_data_dir
        self.output_dir = output_dir
        
        # Data containers
        self.earthquake_df = None
        self.seismic_features = None
        self.combined_features = None
        self.labels = None
        self.models = {}
        
        # Preprocessing components
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.label_encoders = {}
        
        # API endpoints for additional data
        self.api_endpoints = {
            'usgs_latest': 'https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.geojson',
            'usgs_significant': 'https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/significant_month.geojson',
            'emsc': 'https://www.seismicportal.eu/fdsnws/event/1/query?format=json&limit=1000'
        }
        
        # Create output directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        print(f"Enhanced ML Pipeline initialized")
        print(f"Output directory: {self.output_dir}")
    
    def fetch_additional_data(self):
        """Fetch additional earthquake data from multiple APIs"""
        print("\n=== Fetching Additional Earthquake Data ===")
        
        additional_data = []
        
        # USGS Latest Earthquakes
        print("Fetching USGS latest earthquakes...")
        try:
            response = requests.get(self.api_endpoints['usgs_latest'], timeout=30)
            if response.status_code == 200:
                usgs_data = response.json()
                for feature in usgs_data['features']:
                    props = feature['properties']
                    geom = feature['geometry']['coordinates']
                    
                    earthquake = {
                        'Date': datetime.fromtimestamp(props['time']/1000).strftime('%d/%m/%Y'),
                        'Time': datetime.fromtimestamp(props['time']/1000).strftime('%H:%M:%S'),
                        'Latitude': geom[1],
                        'Longitude': geom[0], 
                        'Depth': geom[2] if len(geom) > 2 else 10,
                        'Magnitude': props.get('mag', 0),
                        'Type': 'Earthquake',
                        'Location Source': 'USGS_API',
                        'Country': props.get('place', 'Unknown'),
                        'Status': 'Automatic',
                        'ID': props.get('id', ''),
                        'significance': props.get('sig', 0),
                        'alert': props.get('alert', ''),
                        'tsunami': props.get('tsunami', 0),
                        'nst': props.get('nst', 0),
                        'gap': props.get('gap', 180),
                        'dmin': props.get('dmin', 1.0)
                    }
                    additional_data.append(earthquake)
                
                print(f"✓ Fetched {len(usgs_data['features'])} earthquakes from USGS latest")
        except Exception as e:
            print(f"✗ Failed to fetch USGS latest data: {e}")
        
        # USGS Significant Earthquakes
        print("Fetching USGS significant earthquakes...")
        try:
            response = requests.get(self.api_endpoints['usgs_significant'], timeout=30)
            if response.status_code == 200:
                usgs_sig_data = response.json()
                for feature in usgs_sig_data['features']:
                    props = feature['properties']
                    geom = feature['geometry']['coordinates']
                    
                    earthquake = {
                        'Date': datetime.fromtimestamp(props['time']/1000).strftime('%d/%m/%Y'),
                        'Time': datetime.fromtimestamp(props['time']/1000).strftime('%H:%M:%S'),
                        'Latitude': geom[1],
                        'Longitude': geom[0], 
                        'Depth': geom[2] if len(geom) > 2 else 10,
                        'Magnitude': props.get('mag', 0),
                        'Type': 'Earthquake',
                        'Location Source': 'USGS_API_SIG',
                        'Country': props.get('place', 'Unknown'),
                        'Status': 'Automatic',
                        'ID': props.get('id', ''),
                        'significance': props.get('sig', 0),
                        'alert': props.get('alert', ''),
                        'tsunami': props.get('tsunami', 0),
                        'nst': props.get('nst', 0),
                        'gap': props.get('gap', 180),
                        'dmin': props.get('dmin', 1.0)
                    }
                    additional_data.append(earthquake)
                
                print(f"✓ Fetched {len(usgs_sig_data['features'])} significant earthquakes from USGS")
        except Exception as e:
            print(f"✗ Failed to fetch USGS significant data: {e}")
        
        # EMSC Data
        print("Fetching EMSC earthquake data...")
        try:
            response = requests.get(self.api_endpoints['emsc'], timeout=30)
            if response.status_code == 200:
                emsc_data = response.json()
                for feature in emsc_data['features']:
                    props = feature['properties']
                    geom = feature['geometry']['coordinates']
                    
                    # Convert time
                    time_str = props.get('time', '')
                    if time_str:
                        dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                        earthquake = {
                            'Date': dt.strftime('%d/%m/%Y'),
                            'Time': dt.strftime('%H:%M:%S'),
                            'Latitude': geom[1],
                            'Longitude': geom[0], 
                            'Depth': geom[2] if len(geom) > 2 else 10,
                            'Magnitude': props.get('mag', 0),
                            'Type': 'Earthquake',
                            'Location Source': 'EMSC_API',
                            'Country': props.get('place', 'Unknown'),
                            'Status': 'Automatic',
                            'ID': props.get('id', ''),
                            'significance': 0,  # EMSC doesn't provide significance
                            'alert': '',
                            'tsunami': 0,
                            'nst': 0,
                            'gap': 180,
                            'dmin': 1.0
                        }
                        additional_data.append(earthquake)
                
                print(f"✓ Fetched {len(emsc_data['features'])} earthquakes from EMSC")
        except Exception as e:
            print(f"✗ Failed to fetch EMSC data: {e}")
        
        # Convert to DataFrame and save
        if additional_data:
            additional_df = pd.DataFrame(additional_data)
            
            # Remove duplicates based on time, location, and magnitude
            additional_df['datetime'] = pd.to_datetime(additional_df['Date'] + ' ' + additional_df['Time'], 
                                                     format='%d/%m/%Y %H:%M:%S')
            additional_df = additional_df.drop_duplicates(
                subset=['datetime', 'Latitude', 'Longitude', 'Magnitude']
            )
            
            # Save additional data
            additional_file = os.path.join(self.output_dir, 'additional_earthquake_data.csv')
            additional_df.to_csv(additional_file, index=False)
            print(f"✓ Saved {len(additional_df)} unique additional earthquakes to {additional_file}")
            
            return additional_df
        else:
            print("⚠ No additional data fetched")
            return pd.DataFrame()
    
    def load_earthquake_data(self):
        """Load and combine earthquake data from multiple sources"""
        print("\n=== Loading Earthquake Data ===")
        
        # Load main database
        print(f"Loading main earthquake database: {self.earthquake_csv}")
        self.earthquake_df = pd.read_csv(self.earthquake_csv)
        print(f"✓ Loaded {len(self.earthquake_df)} earthquakes from main database")
        
        # Fetch additional data
        additional_df = self.fetch_additional_data()
        
        # Combine datasets if additional data was fetched
        if not additional_df.empty:
            # Standardize column names and formats
            print("Combining datasets...")
            
            # Add missing columns to additional data with default values
            for col in self.earthquake_df.columns:
                if col not in additional_df.columns:
                    if col in ['Depth Error', 'Magnitude Error', 'Horizontal Distance', 'Horizontal Error']:
                        additional_df[col] = np.nan
                    elif col in ['Depth Seismic Stations', 'Magnitude Seismic Stations', 'Azimuthal Gap']:
                        additional_df[col] = 0
                    elif col == 'Magnitude Type':
                        additional_df[col] = 'ML'
                    elif col == 'Root Mean Square':
                        additional_df[col] = 1.0
                    elif col == 'Source':
                        additional_df[col] = 'API'
                    elif col == 'Magnitude Source':
                        additional_df[col] = 'API'
                    else:
                        additional_df[col] = ''
            
            # Reorder columns to match
            additional_df = additional_df[self.earthquake_df.columns]
            
            # Combine
            combined_df = pd.concat([self.earthquake_df, additional_df], ignore_index=True)
            print(f"✓ Combined dataset: {len(combined_df)} earthquakes total")
            
            self.earthquake_df = combined_df
        
        # Create datetime column
        self.earthquake_df['datetime'] = pd.to_datetime(
            self.earthquake_df['Date'] + ' ' + self.earthquake_df['Time'], 
            format='%d/%m/%Y %H:%M:%S',
            errors='coerce'
        )
        
        # Sort by datetime
        self.earthquake_df = self.earthquake_df.sort_values('datetime').reset_index(drop=True)
        
        print(f"✓ Final dataset: {len(self.earthquake_df)} earthquakes")
        print(f"Date range: {self.earthquake_df['datetime'].min()} to {self.earthquake_df['datetime'].max()}")
        
        return self.earthquake_df
    
    def extract_seismic_features(self):
        """Extract features from seismic waveform data"""
        print("\n=== Extracting Seismic Features ===")
        
        if not OBSPY_AVAILABLE:
            print("⚠ ObsPy not available - skipping seismic feature extraction")
            return pd.DataFrame()
        
        # Load seismic metadata
        metadata_file = os.path.join(self.seismic_data_dir, 'earthquake_seismic_metadata.json')
        if not os.path.exists(metadata_file):
            print(f"⚠ Seismic metadata not found: {metadata_file}")
            return pd.DataFrame()
        
        with open(metadata_file, 'r') as f:
            seismic_metadata = json.load(f)
        
        print(f"✓ Loaded metadata for {len(seismic_metadata)} seismic files")
        
        seismic_features = []
        processed_files = 0
        
        for metadata in seismic_metadata:
            try:
                filename = metadata['filename']
                filepath = os.path.join(self.seismic_data_dir, filename)
                
                if not os.path.exists(filepath):
                    continue
                
                # Read seismic data
                st = read(filepath)
                
                # Initialize feature dictionary
                features = {
                    'earthquake_id': metadata['earthquake_id'],
                    'magnitude': metadata['magnitude'],
                    'station': metadata['station'],
                    'network': metadata['network']
                }
                
                # Process each trace in the stream
                for i, tr in enumerate(st):
                    prefix = f"trace_{i}_"
                    
                    # Basic statistics
                    data = tr.data
                    features[f"{prefix}mean"] = np.mean(data)
                    features[f"{prefix}std"] = np.std(data)
                    features[f"{prefix}max"] = np.max(data)
                    features[f"{prefix}min"] = np.min(data)
                    features[f"{prefix}range"] = np.max(data) - np.min(data)
                    features[f"{prefix}rms"] = np.sqrt(np.mean(data**2))
                    features[f"{prefix}median"] = np.median(data)
                    features[f"{prefix}skewness"] = self.calculate_skewness(data)
                    features[f"{prefix}kurtosis"] = self.calculate_kurtosis(data)
                    
                    # Frequency domain features
                    try:
                        # Power spectral density
                        freqs, psd = welch(data, fs=tr.stats.sampling_rate, nperseg=min(1024, len(data)//4))
                        
                        # Dominant frequency
                        dominant_freq_idx = np.argmax(psd)
                        features[f"{prefix}dominant_frequency"] = freqs[dominant_freq_idx]
                        features[f"{prefix}peak_power"] = psd[dominant_freq_idx]
                        
                        # Frequency band powers
                        low_freq_mask = freqs <= 1.0  # 0-1 Hz
                        mid_freq_mask = (freqs > 1.0) & (freqs <= 10.0)  # 1-10 Hz
                        high_freq_mask = freqs > 10.0  # >10 Hz
                        
                        features[f"{prefix}low_freq_power"] = np.sum(psd[low_freq_mask])
                        features[f"{prefix}mid_freq_power"] = np.sum(psd[mid_freq_mask])
                        features[f"{prefix}high_freq_power"] = np.sum(psd[high_freq_mask])
                        
                        # Spectral centroid
                        features[f"{prefix}spectral_centroid"] = np.sum(freqs * psd) / np.sum(psd)
                        
                    except:
                        # Default values if frequency analysis fails
                        features[f"{prefix}dominant_frequency"] = 0
                        features[f"{prefix}peak_power"] = 0
                        features[f"{prefix}low_freq_power"] = 0
                        features[f"{prefix}mid_freq_power"] = 0
                        features[f"{prefix}high_freq_power"] = 0
                        features[f"{prefix}spectral_centroid"] = 0
                    
                    # Time domain analysis
                    try:
                        # STA/LTA ratio for onset detection
                        cft = classic_sta_lta(data, int(tr.stats.sampling_rate), 
                                            int(tr.stats.sampling_rate * 10))
                        features[f"{prefix}max_sta_lta"] = np.max(cft) if len(cft) > 0 else 0
                        features[f"{prefix}mean_sta_lta"] = np.mean(cft) if len(cft) > 0 else 0
                        
                        # Number of trigger onsets
                        triggers = trigger_onset(cft, 1.5, 0.5)
                        features[f"{prefix}num_triggers"] = len(triggers)
                        
                    except:
                        features[f"{prefix}max_sta_lta"] = 0
                        features[f"{prefix}mean_sta_lta"] = 0
                        features[f"{prefix}num_triggers"] = 0
                
                seismic_features.append(features)
                processed_files += 1
                
                if processed_files % 10 == 0:
                    print(f"Processed {processed_files} seismic files...")
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
        
        if seismic_features:
            self.seismic_features = pd.DataFrame(seismic_features)
            print(f"✓ Extracted features from {len(self.seismic_features)} seismic files")
            
            # Aggregate features by earthquake_id
            self.aggregate_seismic_features()
            
            return self.seismic_features
        else:
            print("⚠ No seismic features extracted")
            return pd.DataFrame()
    
    def calculate_skewness(self, data):
        """Calculate skewness of data"""
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0
            return np.mean(((data - mean) / std) ** 3)
        except:
            return 0
    
    def calculate_kurtosis(self, data):
        """Calculate kurtosis of data"""
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0
            return np.mean(((data - mean) / std) ** 4) - 3
        except:
            return 0
    
    def aggregate_seismic_features(self):
        """Aggregate seismic features by earthquake ID"""
        if self.seismic_features is None or self.seismic_features.empty:
            return
        
        print("Aggregating seismic features by earthquake...")
        
        # Group by earthquake_id and calculate statistics
        numeric_cols = self.seismic_features.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['magnitude']]
        
        aggregated_features = []
        
        for earthquake_id, group in self.seismic_features.groupby('earthquake_id'):
            agg_features = {'earthquake_id': earthquake_id}
            
            # Keep magnitude (should be the same for all stations)
            agg_features['magnitude'] = group['magnitude'].iloc[0]
            
            # Calculate aggregation statistics across stations
            for col in numeric_cols:
                if col in group.columns:
                    agg_features[f"{col}_mean"] = group[col].mean()
                    agg_features[f"{col}_std"] = group[col].std()
                    agg_features[f"{col}_max"] = group[col].max()
                    agg_features[f"{col}_min"] = group[col].min()
            
            # Station count
            agg_features['num_stations'] = len(group)
            
            aggregated_features.append(agg_features)
        
        self.seismic_features = pd.DataFrame(aggregated_features)
        print(f"✓ Aggregated features for {len(self.seismic_features)} earthquakes")
    
    def engineer_catalog_features(self):
        """Engineer features from earthquake catalog data"""
        print("\n=== Engineering Catalog Features ===")
        
        features_df = pd.DataFrame()
        
        # Basic magnitude features
        features_df['magnitude'] = self.earthquake_df['Magnitude'].fillna(0)
        features_df['magnitude_squared'] = features_df['magnitude'] ** 2
        features_df['magnitude_log'] = np.log(features_df['magnitude'] + 1)
        features_df['magnitude_cubed'] = features_df['magnitude'] ** 3
        
        # Depth features
        depth = self.earthquake_df['Depth'].fillna(self.earthquake_df['Depth'].median())
        features_df['depth'] = depth
        features_df['depth_log'] = np.log(depth + 1)
        features_df['depth_sqrt'] = np.sqrt(depth)
        features_df['depth_normalized'] = depth / depth.max()
        features_df['depth_category'] = pd.cut(depth, bins=[0, 35, 70, 300, 700], 
                                              labels=['shallow', 'intermediate', 'deep', 'very_deep'])
        
        # Geographic features
        features_df['latitude'] = self.earthquake_df['Latitude']
        features_df['longitude'] = self.earthquake_df['Longitude']
        features_df['lat_abs'] = np.abs(features_df['latitude'])
        features_df['lon_abs'] = np.abs(features_df['longitude'])
        features_df['distance_from_equator'] = np.abs(features_df['latitude'])
        features_df['distance_from_prime_meridian'] = np.abs(features_df['longitude'])
        
        # Magnitude-depth interaction
        features_df['mag_depth_ratio'] = features_df['magnitude'] / (features_df['depth'] + 1)
        features_df['mag_depth_product'] = features_df['magnitude'] * features_df['depth']
        
        # Temporal features
        dt = self.earthquake_df['datetime']
        features_df['year'] = dt.dt.year
        features_df['month'] = dt.dt.month
        features_df['day'] = dt.dt.day
        features_df['hour'] = dt.dt.hour
        features_df['day_of_year'] = dt.dt.dayofyear
        features_df['day_of_week'] = dt.dt.dayofweek
        features_df['is_weekend'] = features_df['day_of_week'].isin([5, 6]).astype(int)
        
        # Seasonal features
        features_df['sin_month'] = np.sin(2 * np.pi * features_df['month'] / 12)
        features_df['cos_month'] = np.cos(2 * np.pi * features_df['month'] / 12)
        features_df['sin_hour'] = np.sin(2 * np.pi * features_df['hour'] / 24)
        features_df['cos_hour'] = np.cos(2 * np.pi * features_df['hour'] / 24)
        features_df['sin_day_of_year'] = np.sin(2 * np.pi * features_df['day_of_year'] / 365)
        features_df['cos_day_of_year'] = np.cos(2 * np.pi * features_df['day_of_year'] / 365)
        
        # Regional seismic activity
        window_days = [7, 30, 90]
        for window in window_days:
            features_df[f'recent_activity_{window}d'] = self.calculate_recent_activity(window)
            features_df[f'magnitude_trend_{window}d'] = self.calculate_magnitude_trend(window)
        
        # Encode categorical depth
        depth_encoder = LabelEncoder()
        features_df['depth_category_encoded'] = depth_encoder.fit_transform(
            features_df['depth_category'].astype(str)
        )
        
        # Additional features if available
        if 'significance' in self.earthquake_df.columns:
            sig = self.earthquake_df['significance'].fillna(0)
            features_df['significance'] = sig
            features_df['significance_log'] = np.log(sig + 1)
        
        if 'alert' in self.earthquake_df.columns:
            alert_encoder = LabelEncoder()
            alert_values = self.earthquake_df['alert'].fillna('none')
            features_df['alert_encoded'] = alert_encoder.fit_transform(alert_values)
        
        if 'tsunami' in self.earthquake_df.columns:
            features_df['tsunami'] = self.earthquake_df['tsunami'].fillna(0)
        
        if 'nst' in self.earthquake_df.columns:
            features_df['num_stations'] = self.earthquake_df['nst'].fillna(0)
        
        if 'gap' in self.earthquake_df.columns:
            features_df['gap'] = self.earthquake_df['gap'].fillna(180)
        
        if 'dmin' in self.earthquake_df.columns:
            features_df['dmin'] = self.earthquake_df['dmin'].fillna(1.0)
        
        # Remove any infinite or NaN values
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(features_df.median())
        
        print(f"✓ Engineered {len(features_df.columns)} catalog features")
        return features_df
    
    def calculate_recent_activity(self, window_days):
        """Calculate recent seismic activity in a sliding window"""
        activity = []
        
        for i, row in self.earthquake_df.iterrows():
            current_time = row['datetime']
            if pd.isna(current_time):
                activity.append(0)
                continue
                
            # Define window
            start_time = current_time - timedelta(days=window_days)
            end_time = current_time
            
            # Count earthquakes in window (excluding current earthquake)
            mask = (
                (self.earthquake_df['datetime'] >= start_time) & 
                (self.earthquake_df['datetime'] < end_time) &
                (self.earthquake_df.index != i)
            )
            
            # Weight by magnitude and proximity
            window_earthquakes = self.earthquake_df[mask]
            if len(window_earthquakes) > 0:
                # Simple count for now - could add distance weighting
                activity_score = len(window_earthquakes)
            else:
                activity_score = 0
                
            activity.append(activity_score)
        
        return activity
    
    def calculate_magnitude_trend(self, window_days):
        """Calculate magnitude trend in recent earthquakes"""
        trends = []
        
        for i, row in self.earthquake_df.iterrows():
            current_time = row['datetime']
            if pd.isna(current_time):
                trends.append(0)
                continue
                
            # Define window
            start_time = current_time - timedelta(days=window_days)
            end_time = current_time
            
            # Get earthquakes in window
            mask = (
                (self.earthquake_df['datetime'] >= start_time) & 
                (self.earthquake_df['datetime'] < end_time) &
                (self.earthquake_df.index != i)
            )
            
            window_earthquakes = self.earthquake_df[mask]
            if len(window_earthquakes) >= 2:
                # Calculate trend using linear regression
                timestamps = (window_earthquakes['datetime'] - start_time).dt.total_seconds()
                magnitudes = window_earthquakes['Magnitude'].fillna(0)
                
                if len(timestamps) > 1 and timestamps.std() > 0:
                    correlation = np.corrcoef(timestamps, magnitudes)[0, 1]
                    trend = correlation if not np.isnan(correlation) else 0
                else:
                    trend = 0
            else:
                trend = 0
                
            trends.append(trend)
        
        return trends
    
    def combine_features(self):
        """Combine catalog and seismic features"""
        print("\n=== Combining Features ===")
        
        # Get catalog features
        catalog_features = self.engineer_catalog_features()
        
        # Add earthquake IDs for merging
        if 'ID' in self.earthquake_df.columns:
            catalog_features['earthquake_id'] = self.earthquake_df['ID']
        else:
            # Create synthetic IDs
            catalog_features['earthquake_id'] = range(len(catalog_features))
        
        # Combine with seismic features if available
        if self.seismic_features is not None and not self.seismic_features.empty:
            print("Merging catalog and seismic features...")
            
            # Merge on earthquake_id
            self.combined_features = catalog_features.merge(
                self.seismic_features, 
                on='earthquake_id', 
                how='left',
                suffixes=('_catalog', '_seismic')
            )
            
            # Fill missing seismic features with 0
            seismic_cols = [col for col in self.combined_features.columns if 'seismic' in col or 'trace_' in col]
            for col in seismic_cols:
                self.combined_features[col] = self.combined_features[col].fillna(0)
            
            print(f"✓ Combined features: {len(self.combined_features.columns)} total features")
        else:
            print("Using catalog features only (no seismic data)")
            self.combined_features = catalog_features
        
        # Remove earthquake_id from features (keep for reference)
        feature_cols = [col for col in self.combined_features.columns if col != 'earthquake_id']
        self.feature_matrix = self.combined_features[feature_cols]
        
        print(f"✓ Final feature matrix: {self.feature_matrix.shape}")
        return self.feature_matrix
    
    def create_enhanced_labels(self):
        """Create enhanced classification labels"""
        print("\n=== Creating Enhanced Labels ===")
        
        labels = {}
        
        # Traditional labels
        magnitude = self.earthquake_df['Magnitude'].fillna(0)
        
        # Major earthquake (M >= 7.0)
        labels['major_earthquake'] = (magnitude >= 7.0).astype(int)
        
        # Very major earthquake (M >= 8.0)
        labels['very_major_earthquake'] = (magnitude >= 8.0).astype(int)
        
        # Significant earthquake (based on significance score if available)
        if 'significance' in self.earthquake_df.columns:
            significance = self.earthquake_df['significance'].fillna(0)
            labels['significant_earthquake'] = (significance >= 600).astype(int)
        else:
            # Use magnitude as proxy
            labels['significant_earthquake'] = (magnitude >= 6.0).astype(int)
        
        # Tsunami generating earthquake
        if 'tsunami' in self.earthquake_df.columns:
            labels['tsunami_generating'] = self.earthquake_df['tsunami'].fillna(0).astype(int)
        else:
            # Use depth and magnitude as proxy
            depth = self.earthquake_df['Depth'].fillna(50)
            labels['tsunami_generating'] = ((magnitude >= 6.5) & (depth <= 50)).astype(int)
        
        # Depth-based labels
        depth = self.earthquake_df['Depth'].fillna(50)
        labels['shallow_earthquake'] = (depth <= 35).astype(int)
        labels['deep_earthquake'] = (depth >= 300).astype(int)
        
        # Multi-class magnitude categories
        mag_categories = pd.cut(magnitude, 
                               bins=[0, 4, 5, 6, 7, 8, 12], 
                               labels=[0, 1, 2, 3, 4, 5])
        labels['magnitude_category'] = mag_categories.astype(int)
        
        self.labels = labels
        
        # Print label statistics
        print("Label statistics:")
        for label_name, label_values in labels.items():
            if label_name != 'magnitude_category':
                positive_count = label_values.sum()
                total_count = len(label_values)
                percentage = (positive_count / total_count) * 100
                print(f"  {label_name}: {positive_count}/{total_count} ({percentage:.1f}%)")
            else:
                print(f"  {label_name}: {label_values.value_counts().to_dict()}")
        
        return labels
    
    def advanced_feature_selection(self, task='major_earthquake', k=50):
        """Advanced feature selection using multiple methods"""
        print(f"\n=== Advanced Feature Selection for {task} ===")
        
        X = self.feature_matrix
        y = self.labels[task]
        
        # Remove constant features
        constant_features = X.columns[X.std() == 0]
        if len(constant_features) > 0:
            print(f"Removing {len(constant_features)} constant features")
            X = X.drop(columns=constant_features)
        
        selected_features = []
        
        # Method 1: Statistical tests (f_classif)
        print("Selecting features using statistical tests...")
        selector_stats = SelectKBest(score_func=f_classif, k=min(k//2, len(X.columns)))
        X_stats = selector_stats.fit_transform(X, y)
        stats_features = X.columns[selector_stats.get_support()].tolist()
        selected_features.extend(stats_features)
        
        # Method 2: Mutual information
        print("Selecting features using mutual information...")
        selector_mi = SelectKBest(score_func=mutual_info_classif, k=min(k//2, len(X.columns)))
        X_mi = selector_mi.fit_transform(X, y)
        mi_features = X.columns[selector_mi.get_support()].tolist()
        selected_features.extend(mi_features)
        
        # Method 3: Recursive Feature Elimination with Random Forest
        if len(X.columns) <= 100:  # Only for smaller feature sets
            print("Selecting features using RFE...")
            rf = RandomForestClassifier(n_estimators=50, random_state=42)
            selector_rfe = RFE(rf, n_features_to_select=min(k//3, len(X.columns)))
            selector_rfe.fit(X, y)
            rfe_features = X.columns[selector_rfe.get_support()].tolist()
            selected_features.extend(rfe_features)
        
        # Combine and remove duplicates
        selected_features = list(set(selected_features))
        
        # Limit to k features
        if len(selected_features) > k:
            # Use Random Forest feature importance to rank
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X[selected_features], y)
            
            importance_df = pd.DataFrame({
                'feature': selected_features,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            selected_features = importance_df['feature'].head(k).tolist()
        
        print(f"✓ Selected {len(selected_features)} features")
        
        return X[selected_features], selected_features
    
    def train_enhanced_models(self, task='major_earthquake'):
        """Train enhanced ML models with hyperparameter tuning"""
        print(f"\n=== Training Enhanced Models for {task} ===")
        
        # Prepare data
        X, selected_features = self.advanced_feature_selection(task)
        y = self.labels[task]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Positive class ratio: {y_train.mean():.3f}")
        
        # Enhanced model definitions with hyperparameter grids
        models_config = {
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 15, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                }
            },
            'Extra Trees': {
                'model': ExtraTreesClassifier(random_state=42, class_weight='balanced'),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 15, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [4, 6, 8],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'SVM': {
                'model': SVC(random_state=42, probability=True, class_weight='balanced'),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                    'kernel': ['rbf', 'poly']
                }
            },
            'Neural Network': {
                'model': MLPClassifier(random_state=42, max_iter=1000),
                'params': {
                    'hidden_layer_sizes': [(100,), (100, 50), (200, 100), (100, 50, 25)],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                }
            },
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2', 'elasticnet'],
                    'solver': ['liblinear', 'saga']
                }
            }
        }
        
        results = {}
        
        for name, config in models_config.items():
            print(f"\nTraining {name}...")
            
            try:
                # Grid search with cross-validation
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                grid_search = GridSearchCV(
                    config['model'], 
                    config['params'], 
                    cv=cv, 
                    scoring='roc_auc',
                    n_jobs=-1,
                    verbose=0
                )
                
                # Fit with limited parameter combinations for speed
                grid_search.fit(X_train, y_train)
                
                best_model = grid_search.best_estimator_
                
                # Predictions
                y_pred = best_model.predict(X_test)
                y_pred_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None
                
                # Metrics
                results[name] = {
                    'model': best_model,
                    'best_params': grid_search.best_params_,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'classification_report': classification_report(y_test, y_pred),
                    'confusion_matrix': confusion_matrix(y_test, y_pred),
                    'best_cv_score': grid_search.best_score_,
                    'roc_auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
                }
                
                print(f"✓ {name} - Best CV Score: {grid_search.best_score_:.3f}")
                if results[name]['roc_auc']:
                    print(f"  Test ROC-AUC: {results[name]['roc_auc']:.3f}")
                
            except Exception as e:
                print(f"✗ Failed to train {name}: {e}")
                continue
        
        # Store results
        self.models[task] = {
            'results': results,
            'test_data': (X_test, y_test),
            'selected_features': selected_features,
            'scaler': self.scaler
        }
        
        print(f"\n✓ Trained {len(results)} models for {task}")
        return results
    
    def create_visualizations(self, task='major_earthquake'):
        """Create enhanced visualizations"""
        print(f"\n=== Creating Visualizations for {task} ===")
        
        if task not in self.models:
            print(f"No models found for task: {task}")
            return
        
        results = self.models[task]['results']
        X_test, y_test = self.models[task]['test_data']
        
        # 1. ROC Curves
        plt.figure(figsize=(12, 8))
        for name, result in results.items():
            if result['probabilities'] is not None:
                fpr, tpr, _ = roc_curve(y_test, result['probabilities'])
                roc_auc = result['roc_auc']
                plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves - {task}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'enhanced_roc_curves_{task}.png'), dpi=300)
        plt.close()
        
        # 2. Feature Importance (for tree-based models)
        selected_features = self.models[task]['selected_features']
        
        for name, result in results.items():
            if hasattr(result['model'], 'feature_importances_'):
                plt.figure(figsize=(12, 8))
                importances = result['model'].feature_importances_
                
                # Sort features by importance
                feature_importance = pd.DataFrame({
                    'feature': selected_features,
                    'importance': importances
                }).sort_values('importance', ascending=True)
                
                # Plot top 20 features
                top_features = feature_importance.tail(20)
                plt.barh(range(len(top_features)), top_features['importance'])
                plt.yticks(range(len(top_features)), top_features['feature'])
                plt.xlabel('Feature Importance')
                plt.title(f'Feature Importance - {name} ({task})')
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f'enhanced_feature_importance_{name}_{task}.png'), dpi=300)
                plt.close()
        
        # 3. Confusion Matrices
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, (name, result) in enumerate(results.items()):
            if i >= len(axes):
                break
                
            cm = result['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues')
            axes[i].set_title(f'{name}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        # Hide empty subplots
        for i in range(len(results), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Confusion Matrices - {task}', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'enhanced_confusion_matrices_{task}.png'), dpi=300)
        plt.close()
        
        print(f"✓ Visualizations saved to {self.output_dir}")
    
    def save_enhanced_models(self, task='major_earthquake'):
        """Save enhanced models and metadata"""
        print(f"\n=== Saving Enhanced Models for {task} ===")
        
        task_dir = os.path.join(self.output_dir, task)
        if not os.path.exists(task_dir):
            os.makedirs(task_dir)
        
        results = self.models[task]['results']
        
        # Save each model
        for model_name, result in results.items():
            model_file = os.path.join(task_dir, f'{model_name.lower().replace(" ", "_")}_model.pkl')
            with open(model_file, 'wb') as f:
                pickle.dump(result['model'], f)
        
        # Save preprocessing components
        scaler_file = os.path.join(task_dir, 'scaler.pkl')
        with open(scaler_file, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save metadata
        metadata = {
            'task': task,
            'selected_features': self.models[task]['selected_features'],
            'model_performances': {
                name: {
                    'best_cv_score': result['best_cv_score'],
                    'test_roc_auc': result['roc_auc'],
                    'best_params': result['best_params']
                }
                for name, result in results.items()
            },
            'training_date': datetime.now().isoformat(),
            'data_size': len(self.earthquake_df),
            'feature_count': len(self.feature_matrix.columns),
            'selected_feature_count': len(self.models[task]['selected_features']),
            'seismic_features_included': self.seismic_features is not None and not self.seismic_features.empty
        }
        
        metadata_file = os.path.join(task_dir, 'enhanced_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Enhanced models saved to: {task_dir}")
    
    def run_enhanced_pipeline(self, tasks=None):
        """Run the complete enhanced ML pipeline"""
        print("="*80)
        print("ENHANCED EARTHQUAKE ML PIPELINE")
        print("="*80)
        
        if tasks is None:
            tasks = ['major_earthquake', 'very_major_earthquake', 'significant_earthquake', 
                    'tsunami_generating', 'shallow_earthquake']
        
        # Load and prepare data
        self.load_earthquake_data()
        self.extract_seismic_features()
        self.combine_features()
        self.create_enhanced_labels()
        
        # Train models for each task
        for task in tasks:
            if task in self.labels:
                try:
                    self.train_enhanced_models(task)
                    self.create_visualizations(task)
                    self.save_enhanced_models(task)
                except Exception as e:
                    print(f"Error processing task {task}: {e}")
                    continue
        
        print("\n" + "="*80)
        print("ENHANCED PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)


def main():
    """Main function to run the enhanced earthquake ML pipeline"""
    pipeline = EnhancedEarthquakeMLPipeline()
    
    # Run full enhanced pipeline
    pipeline.run_enhanced_pipeline()


if __name__ == "__main__":
    main()
