#!/usr/bin/env python3
"""
Accelerometer and Gyroscope Feature Extractor for Earthquake Detection
This script extracts features from accelerometer (x, y, z) and gyroscope data
for real-time earthquake detection and prediction.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from scipy import signal
from scipy.stats import kurtosis, skew
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')

class AccelerometerFeatureExtractor:
    def __init__(self, output_dir='sensor_ml_models'):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def extract_time_domain_features(self, data, axis_name):
        """Extract time-domain features from sensor data"""
        features = {}
        prefix = f"{axis_name}_"
        
        # Basic statistical features
        features[f'{prefix}mean'] = np.mean(data)
        features[f'{prefix}std'] = np.std(data)
        features[f'{prefix}var'] = np.var(data)
        features[f'{prefix}max'] = np.max(data)
        features[f'{prefix}min'] = np.min(data)
        features[f'{prefix}range'] = np.max(data) - np.min(data)
        features[f'{prefix}median'] = np.median(data)
        features[f'{prefix}rms'] = np.sqrt(np.mean(data**2))
        
        # Higher order moments
        features[f'{prefix}skewness'] = skew(data)
        features[f'{prefix}kurtosis'] = kurtosis(data)
        
        # Peak-to-peak and percentiles
        features[f'{prefix}peak_to_peak'] = np.ptp(data)
        features[f'{prefix}q25'] = np.percentile(data, 25)
        features[f'{prefix}q75'] = np.percentile(data, 75)
        features[f'{prefix}iqr'] = features[f'{prefix}q75'] - features[f'{prefix}q25']
        
        # Energy and power
        features[f'{prefix}energy'] = np.sum(data**2)
        features[f'{prefix}power'] = features[f'{prefix}energy'] / len(data)
        
        # Zero crossing rate
        zero_crossings = np.sum(np.diff(np.sign(data)) != 0)
        features[f'{prefix}zero_crossing_rate'] = zero_crossings / len(data)
        
        # Signal magnitude area (SMA)
        features[f'{prefix}sma'] = np.sum(np.abs(data)) / len(data)
        
        # Crest factor and form factor
        if features[f'{prefix}rms'] > 0:
            features[f'{prefix}crest_factor'] = features[f'{prefix}max'] / features[f'{prefix}rms']
        else:
            features[f'{prefix}crest_factor'] = 0
        
        mean_abs = np.mean(np.abs(data))
        if mean_abs > 0:
            features[f'{prefix}form_factor'] = features[f'{prefix}rms'] / mean_abs
        else:
            features[f'{prefix}form_factor'] = 0
        
        # Autocorrelation features
        autocorr = np.correlate(data, data, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]  # Normalize
        
        if len(autocorr) > 1:
            features[f'{prefix}autocorr_max'] = np.max(autocorr[1:])  # Exclude lag 0
            features[f'{prefix}autocorr_min'] = np.min(autocorr[1:])
        else:
            features[f'{prefix}autocorr_max'] = 0
            features[f'{prefix}autocorr_min'] = 0
        
        return features
    
    def extract_frequency_domain_features(self, data, sampling_rate, axis_name):
        """Extract frequency-domain features"""
        features = {}
        prefix = f"{axis_name}_freq_"
        
        try:
            # FFT
            n = len(data)
            fft_vals = fft(data)
            freqs = fftfreq(n, 1/sampling_rate)
            
            # Take only positive frequencies
            positive_freq_idx = freqs > 0
            freqs = freqs[positive_freq_idx]
            fft_magnitude = np.abs(fft_vals[positive_freq_idx])
            
            if len(fft_magnitude) == 0:
                return {f'{prefix}{key}': 0 for key in ['dominant', 'peak_power', 'centroid', 
                                                      'low_power', 'mid_power', 'high_power',
                                                      'bandwidth', 'spectral_entropy']}
            
            # Power spectral density
            psd = fft_magnitude**2 / n
            
            # Dominant frequency
            dominant_freq_idx = np.argmax(psd)
            features[f'{prefix}dominant'] = freqs[dominant_freq_idx]
            features[f'{prefix}peak_power'] = psd[dominant_freq_idx]
            
            # Spectral centroid
            features[f'{prefix}centroid'] = np.sum(freqs * psd) / np.sum(psd)
            
            # Frequency band powers (for earthquake detection)
            # Low: 0-1 Hz, Mid: 1-10 Hz, High: 10+ Hz
            low_freq_mask = (freqs >= 0) & (freqs <= 1.0)
            mid_freq_mask = (freqs > 1.0) & (freqs <= 10.0)
            high_freq_mask = freqs > 10.0
            
            features[f'{prefix}low_power'] = np.sum(psd[low_freq_mask]) if np.any(low_freq_mask) else 0
            features[f'{prefix}mid_power'] = np.sum(psd[mid_freq_mask]) if np.any(mid_freq_mask) else 0
            features[f'{prefix}high_power'] = np.sum(psd[high_freq_mask]) if np.any(high_freq_mask) else 0
            
            # Total power and ratios
            total_power = features[f'{prefix}low_power'] + features[f'{prefix}mid_power'] + features[f'{prefix}high_power']
            if total_power > 0:
                features[f'{prefix}low_ratio'] = features[f'{prefix}low_power'] / total_power
                features[f'{prefix}mid_ratio'] = features[f'{prefix}mid_power'] / total_power
                features[f'{prefix}high_ratio'] = features[f'{prefix}high_power'] / total_power
            else:
                features[f'{prefix}low_ratio'] = 0
                features[f'{prefix}mid_ratio'] = 0
                features[f'{prefix}high_ratio'] = 0
            
            # Spectral bandwidth
            spectral_spread = np.sqrt(np.sum(((freqs - features[f'{prefix}centroid'])**2) * psd) / np.sum(psd))
            features[f'{prefix}bandwidth'] = spectral_spread
            
            # Spectral entropy
            psd_norm = psd / np.sum(psd)
            psd_norm = psd_norm[psd_norm > 0]  # Remove zeros for log
            spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm))
            features[f'{prefix}spectral_entropy'] = spectral_entropy
            
        except Exception as e:
            print(f"Error in frequency analysis for {axis_name}: {e}")
            features = {f'{prefix}{key}': 0 for key in ['dominant', 'peak_power', 'centroid', 
                                                      'low_power', 'mid_power', 'high_power',
                                                      'low_ratio', 'mid_ratio', 'high_ratio',
                                                      'bandwidth', 'spectral_entropy']}
        
        return features
    
    def extract_magnitude_vector_features(self, acc_x, acc_y, acc_z, gyro_x=None, gyro_y=None, gyro_z=None):
        """Extract features from magnitude vectors"""
        features = {}
        
        # Accelerometer magnitude vector
        acc_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        
        features['acc_magnitude_mean'] = np.mean(acc_magnitude)
        features['acc_magnitude_std'] = np.std(acc_magnitude)
        features['acc_magnitude_max'] = np.max(acc_magnitude)
        features['acc_magnitude_min'] = np.min(acc_magnitude)
        features['acc_magnitude_range'] = features['acc_magnitude_max'] - features['acc_magnitude_min']
        
        # Jerk (rate of change of acceleration)
        acc_jerk = np.sqrt(np.diff(acc_x)**2 + np.diff(acc_y)**2 + np.diff(acc_z)**2)
        if len(acc_jerk) > 0:
            features['acc_jerk_mean'] = np.mean(acc_jerk)
            features['acc_jerk_std'] = np.std(acc_jerk)
            features['acc_jerk_max'] = np.max(acc_jerk)
        else:
            features['acc_jerk_mean'] = 0
            features['acc_jerk_std'] = 0
            features['acc_jerk_max'] = 0
        
        # Gyroscope magnitude vector (if available)
        if gyro_x is not None and gyro_y is not None and gyro_z is not None:
            gyro_magnitude = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
            
            features['gyro_magnitude_mean'] = np.mean(gyro_magnitude)
            features['gyro_magnitude_std'] = np.std(gyro_magnitude)
            features['gyro_magnitude_max'] = np.max(gyro_magnitude)
            features['gyro_magnitude_min'] = np.min(gyro_magnitude)
            features['gyro_magnitude_range'] = features['gyro_magnitude_max'] - features['gyro_magnitude_min']
            
            # Angular jerk
            gyro_jerk = np.sqrt(np.diff(gyro_x)**2 + np.diff(gyro_y)**2 + np.diff(gyro_z)**2)
            if len(gyro_jerk) > 0:
                features['gyro_jerk_mean'] = np.mean(gyro_jerk)
                features['gyro_jerk_std'] = np.std(gyro_jerk)
                features['gyro_jerk_max'] = np.max(gyro_jerk)
            else:
                features['gyro_jerk_mean'] = 0
                features['gyro_jerk_std'] = 0
                features['gyro_jerk_max'] = 0
        
        return features
    
    def extract_correlation_features(self, acc_x, acc_y, acc_z, gyro_x=None, gyro_y=None, gyro_z=None):
        """Extract correlation features between axes"""
        features = {}
        
        # Accelerometer cross-correlations
        features['acc_xy_correlation'] = np.corrcoef(acc_x, acc_y)[0, 1] if len(acc_x) > 1 else 0
        features['acc_xz_correlation'] = np.corrcoef(acc_x, acc_z)[0, 1] if len(acc_x) > 1 else 0
        features['acc_yz_correlation'] = np.corrcoef(acc_y, acc_z)[0, 1] if len(acc_y) > 1 else 0
        
        # Handle NaN correlations
        for key in ['acc_xy_correlation', 'acc_xz_correlation', 'acc_yz_correlation']:
            if np.isnan(features[key]):
                features[key] = 0
        
        # Gyroscope cross-correlations (if available)
        if gyro_x is not None and gyro_y is not None and gyro_z is not None:
            features['gyro_xy_correlation'] = np.corrcoef(gyro_x, gyro_y)[0, 1] if len(gyro_x) > 1 else 0
            features['gyro_xz_correlation'] = np.corrcoef(gyro_x, gyro_z)[0, 1] if len(gyro_x) > 1 else 0
            features['gyro_yz_correlation'] = np.corrcoef(gyro_y, gyro_z)[0, 1] if len(gyro_y) > 1 else 0
            
            # Handle NaN correlations
            for key in ['gyro_xy_correlation', 'gyro_xz_correlation', 'gyro_yz_correlation']:
                if np.isnan(features[key]):
                    features[key] = 0
            
            # Accelerometer-Gyroscope correlations
            features['acc_gyro_x_correlation'] = np.corrcoef(acc_x, gyro_x)[0, 1] if len(acc_x) > 1 else 0
            features['acc_gyro_y_correlation'] = np.corrcoef(acc_y, gyro_y)[0, 1] if len(acc_y) > 1 else 0
            features['acc_gyro_z_correlation'] = np.corrcoef(acc_z, gyro_z)[0, 1] if len(acc_z) > 1 else 0
            
            # Handle NaN correlations
            for key in ['acc_gyro_x_correlation', 'acc_gyro_y_correlation', 'acc_gyro_z_correlation']:
                if np.isnan(features[key]):
                    features[key] = 0
        
        return features
    
    def extract_earthquake_specific_features(self, acc_x, acc_y, acc_z, sampling_rate, gyro_x=None, gyro_y=None, gyro_z=None):
        """Extract features specifically relevant to earthquake detection"""
        features = {}
        
        # Peak Ground Acceleration (PGA) - maximum acceleration
        acc_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        features['pga'] = np.max(acc_magnitude)
        
        # Peak Ground Velocity (PGV) - estimated from acceleration
        # Simple integration using cumsum
        velocity_x = np.cumsum(acc_x) / sampling_rate
        velocity_y = np.cumsum(acc_y) / sampling_rate
        velocity_z = np.cumsum(acc_z) / sampling_rate
        velocity_magnitude = np.sqrt(velocity_x**2 + velocity_y**2 + velocity_z**2)
        features['pgv'] = np.max(velocity_magnitude)
        
        # Arias Intensity (measure of earthquake strength)
        arias_intensity = np.pi / (2 * 9.81) * np.sum(acc_magnitude**2) / sampling_rate
        features['arias_intensity'] = arias_intensity
        
        # Significant duration (time between 5% and 95% of Arias intensity)
        cumulative_arias = np.cumsum(acc_magnitude**2)
        total_arias = cumulative_arias[-1]
        
        if total_arias > 0:
            t5_idx = np.where(cumulative_arias >= 0.05 * total_arias)[0]
            t95_idx = np.where(cumulative_arias >= 0.95 * total_arias)[0]
            
            if len(t5_idx) > 0 and len(t95_idx) > 0:
                features['significant_duration'] = (t95_idx[0] - t5_idx[0]) / sampling_rate
            else:
                features['significant_duration'] = 0
        else:
            features['significant_duration'] = 0
        
        # Seismic intensity indicators
        # Strong motion duration (time acceleration > 0.05g)
        g = 9.81  # gravity
        strong_motion_threshold = 0.05 * g
        strong_motion_mask = acc_magnitude > strong_motion_threshold
        features['strong_motion_duration'] = np.sum(strong_motion_mask) / sampling_rate
        
        # Response spectrum approximation (dominant period)
        try:
            # Find dominant period from frequency analysis
            n = len(acc_magnitude)
            freqs = fftfreq(n, 1/sampling_rate)
            fft_magnitude = np.abs(fft(acc_magnitude))
            
            positive_freq_idx = freqs > 0
            freqs_pos = freqs[positive_freq_idx]
            fft_pos = fft_magnitude[positive_freq_idx]
            
            if len(fft_pos) > 0:
                dominant_freq = freqs_pos[np.argmax(fft_pos)]
                features['dominant_period'] = 1.0 / dominant_freq if dominant_freq > 0 else 0
            else:
                features['dominant_period'] = 0
        except:
            features['dominant_period'] = 0
        
        # Vertical to horizontal ratio (important for earthquake characterization)
        horizontal_acc = np.sqrt(acc_x**2 + acc_y**2)
        vertical_acc = np.abs(acc_z)
        
        mean_horizontal = np.mean(horizontal_acc)
        mean_vertical = np.mean(vertical_acc)
        
        if mean_horizontal > 0:
            features['vh_ratio'] = mean_vertical / mean_horizontal
        else:
            features['vh_ratio'] = 0
        
        return features
    
    def extract_all_features(self, sensor_data, sampling_rate=100):
        """
        Extract all features from sensor data
        
        Parameters:
        sensor_data: dict with keys 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'
        sampling_rate: sampling rate in Hz
        """
        features = {}
        
        # Extract accelerometer data
        acc_x = np.array(sensor_data['acc_x'])
        acc_y = np.array(sensor_data['acc_y'])
        acc_z = np.array(sensor_data['acc_z'])
        
        # Extract gyroscope data (if available)
        gyro_x = np.array(sensor_data.get('gyro_x', [])) if 'gyro_x' in sensor_data else None
        gyro_y = np.array(sensor_data.get('gyro_y', [])) if 'gyro_y' in sensor_data else None
        gyro_z = np.array(sensor_data.get('gyro_z', [])) if 'gyro_z' in sensor_data else None
        
        # Basic metadata
        features['sampling_rate'] = sampling_rate
        features['data_length'] = len(acc_x)
        features['duration'] = len(acc_x) / sampling_rate
        
        # Time domain features for each axis
        features.update(self.extract_time_domain_features(acc_x, 'acc_x'))
        features.update(self.extract_time_domain_features(acc_y, 'acc_y'))
        features.update(self.extract_time_domain_features(acc_z, 'acc_z'))
        
        if gyro_x is not None and len(gyro_x) > 0:
            features.update(self.extract_time_domain_features(gyro_x, 'gyro_x'))
            features.update(self.extract_time_domain_features(gyro_y, 'gyro_y'))
            features.update(self.extract_time_domain_features(gyro_z, 'gyro_z'))
        
        # Frequency domain features
        features.update(self.extract_frequency_domain_features(acc_x, sampling_rate, 'acc_x'))
        features.update(self.extract_frequency_domain_features(acc_y, sampling_rate, 'acc_y'))
        features.update(self.extract_frequency_domain_features(acc_z, sampling_rate, 'acc_z'))
        
        if gyro_x is not None and len(gyro_x) > 0:
            features.update(self.extract_frequency_domain_features(gyro_x, sampling_rate, 'gyro_x'))
            features.update(self.extract_frequency_domain_features(gyro_y, sampling_rate, 'gyro_y'))
            features.update(self.extract_frequency_domain_features(gyro_z, sampling_rate, 'gyro_z'))
        
        # Magnitude vector features
        features.update(self.extract_magnitude_vector_features(acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z))
        
        # Correlation features
        features.update(self.extract_correlation_features(acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z))
        
        # Earthquake-specific features
        features.update(self.extract_earthquake_specific_features(acc_x, acc_y, acc_z, sampling_rate, gyro_x, gyro_y, gyro_z))
        
        return features
    
    def process_sensor_file(self, filepath, label=0):
        """
        Process a sensor data file
        
        Parameters:
        filepath: path to CSV file with columns: timestamp, acc_x, acc_y, acc_z, [gyro_x, gyro_y, gyro_z]
        label: 0 for normal, 1 for earthquake
        """
        try:
            # Read sensor data
            df = pd.read_csv(filepath)
            
            # Determine sampling rate
            if 'timestamp' in df.columns and len(df) > 1:
                time_diff = pd.to_datetime(df['timestamp'].iloc[1]) - pd.to_datetime(df['timestamp'].iloc[0])
                sampling_rate = 1.0 / time_diff.total_seconds()
            else:
                sampling_rate = 100  # Default 100 Hz
            
            # Prepare sensor data dictionary
            sensor_data = {
                'acc_x': df['acc_x'].values,
                'acc_y': df['acc_y'].values,
                'acc_z': df['acc_z'].values
            }
            
            # Add gyroscope data if available
            if 'gyro_x' in df.columns:
                sensor_data['gyro_x'] = df['gyro_x'].values
                sensor_data['gyro_y'] = df['gyro_y'].values
                sensor_data['gyro_z'] = df['gyro_z'].values
            
            # Extract features
            features = self.extract_all_features(sensor_data, sampling_rate)
            
            # Add metadata
            features['filename'] = os.path.basename(filepath)
            features['label'] = label
            features['timestamp'] = datetime.now().isoformat()
            
            return features
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            return None
    
    def create_sample_data(self, output_file='sample_sensor_data.csv', duration=10, sampling_rate=100):
        """Create sample sensor data for testing"""
        
        print(f"Creating sample sensor data: {output_file}")
        
        # Generate time vector
        time_points = int(duration * sampling_rate)
        t = np.linspace(0, duration, time_points)
        
        # Generate synthetic earthquake-like data
        # Normal background noise
        noise_level = 0.1
        acc_x = np.random.normal(0, noise_level, time_points)
        acc_y = np.random.normal(0, noise_level, time_points)
        acc_z = np.random.normal(9.81, noise_level, time_points)  # Gravity + noise
        
        # Add earthquake signal (simulated)
        if 'earthquake' in output_file:
            # Add earthquake characteristics
            earthquake_start = int(0.3 * time_points)
            earthquake_end = int(0.7 * time_points)
            
            # Primary wave (P-wave) - higher frequency, lower amplitude
            p_wave_freq = 8  # Hz
            p_wave_amplitude = 2.0
            p_wave = p_wave_amplitude * np.sin(2 * np.pi * p_wave_freq * t[earthquake_start:earthquake_end])
            
            # Secondary wave (S-wave) - lower frequency, higher amplitude
            s_wave_freq = 3  # Hz
            s_wave_amplitude = 5.0
            s_wave_delay = int(0.1 * (earthquake_end - earthquake_start))
            
            if s_wave_delay < len(p_wave):
                s_wave = s_wave_amplitude * np.sin(2 * np.pi * s_wave_freq * t[earthquake_start:earthquake_end-s_wave_delay])
                
                # Combine waves
                acc_x[earthquake_start:earthquake_end] += p_wave
                acc_y[earthquake_start:earthquake_end] += p_wave * 0.7  # Different amplitude
                acc_z[earthquake_start:earthquake_end-s_wave_delay] += s_wave
        
        # Generate gyroscope data (correlated with acceleration changes)
        gyro_x = np.gradient(acc_y) * 10 + np.random.normal(0, 0.05, time_points)
        gyro_y = -np.gradient(acc_x) * 10 + np.random.normal(0, 0.05, time_points)
        gyro_z = np.random.normal(0, 0.02, time_points)
        
        # Create DataFrame
        sample_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=time_points, freq=f'{1000//sampling_rate}ms'),
            'acc_x': acc_x,
            'acc_y': acc_y,
            'acc_z': acc_z,
            'gyro_x': gyro_x,
            'gyro_y': gyro_y,
            'gyro_z': gyro_z
        })
        
        # Save to file
        sample_data.to_csv(output_file, index=False)
        print(f"✓ Sample data saved to: {output_file}")
        
        return output_file


def main():
    """Main function to demonstrate accelerometer feature extraction"""
    extractor = AccelerometerFeatureExtractor()
    
    # Create sample data files
    normal_file = extractor.create_sample_data('sample_normal_data.csv')
    earthquake_file = extractor.create_sample_data('sample_earthquake_data.csv')
    
    # Process sample files
    print("\n=== Processing Sample Data ===")
    
    normal_features = extractor.process_sensor_file(normal_file, label=0)
    earthquake_features = extractor.process_sensor_file(earthquake_file, label=1)
    
    if normal_features and earthquake_features:
        # Create feature comparison
        feature_comparison = pd.DataFrame([normal_features, earthquake_features])
        comparison_file = os.path.join(extractor.output_dir, 'feature_comparison.csv')
        feature_comparison.to_csv(comparison_file, index=False)
        
        print(f"✓ Feature comparison saved to: {comparison_file}")
        print(f"Total features extracted: {len(normal_features) - 3}")  # Exclude metadata
        
        # Show key differences
        print("\n=== Key Feature Differences ===")
        key_features = ['pga', 'pgv', 'arias_intensity', 'acc_magnitude_max', 'acc_x_energy', 'significant_duration']
        
        for feature in key_features:
            if feature in normal_features and feature in earthquake_features:
                normal_val = normal_features[feature]
                earthquake_val = earthquake_features[feature]
                ratio = earthquake_val / normal_val if normal_val != 0 else float('inf')
                print(f"{feature}: Normal={normal_val:.3f}, Earthquake={earthquake_val:.3f}, Ratio={ratio:.2f}")
    
    print("\n✓ Accelerometer feature extraction demonstration completed!")


if __name__ == "__main__":
    main()
