#!/usr/bin/env python3
"""
Advanced Seismic Feature Extractor
This script extracts sophisticated features from seismic waveform data (.mseed files)
for use in earthquake detection machine learning models.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from obspy import read, UTCDateTime
    from obspy.signal import filter
    from obspy.signal.trigger import classic_sta_lta, trigger_onset, z_detect
    from scipy.signal import welch
    from obspy.signal.util import smooth
    from scipy import signal
    from scipy.stats import kurtosis, skew
    import pywt  # PyWavelets for modern wavelet analysis
    OBSPY_AVAILABLE = True
    print("✓ ObsPy, scipy, and PyWavelets available for advanced seismic processing")
except ImportError as e:
    OBSPY_AVAILABLE = False
    print(f"⚠ Advanced seismic processing not available: {e}")
    print("Please install: pip install obspy scipy PyWavelets")

class AdvancedSeismicFeatureExtractor:
    def __init__(self, seismic_data_dir='earthquake_seismic_data', output_dir='enhanced_ml_models'):
        self.seismic_data_dir = seismic_data_dir
        self.output_dir = output_dir
        self.metadata = None
        self.features_df = None
        
        if not OBSPY_AVAILABLE:
            print("Cannot proceed without ObsPy. Please install: pip install obspy")
            return
            
        # Load metadata
        self.load_metadata()
        
    def load_metadata(self):
        """Load seismic data metadata"""
        metadata_file = os.path.join(self.seismic_data_dir, 'earthquake_seismic_metadata.json')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
            print(f"✓ Loaded metadata for {len(self.metadata)} seismic files")
        else:
            print(f"⚠ Metadata file not found: {metadata_file}")
    
    def extract_time_domain_features(self, trace):
        """Extract comprehensive time-domain features from a seismic trace"""
        data = trace.data
        features = {}
        
        # Basic statistical features
        features['mean'] = np.mean(data)
        features['std'] = np.std(data)
        features['variance'] = np.var(data)
        features['max'] = np.max(data)
        features['min'] = np.min(data)
        features['range'] = np.max(data) - np.min(data)
        features['median'] = np.median(data)
        features['q25'] = np.percentile(data, 25)
        features['q75'] = np.percentile(data, 75)
        features['iqr'] = features['q75'] - features['q25']
        
        # Higher order moments
        features['skewness'] = skew(data)
        features['kurtosis'] = kurtosis(data)
        
        # Energy-based features
        features['rms'] = np.sqrt(np.mean(data**2))
        features['energy'] = np.sum(data**2)
        features['power'] = features['energy'] / len(data)
        
        # Zero crossing rate
        zero_crossings = np.sum(np.diff(np.sign(data)) != 0)
        features['zero_crossing_rate'] = zero_crossings / len(data)
        
        # Envelope features
        analytic_signal = signal.hilbert(data)
        envelope = np.abs(analytic_signal)
        features['envelope_mean'] = np.mean(envelope)
        features['envelope_std'] = np.std(envelope)
        features['envelope_max'] = np.max(envelope)
        
        # Crest factor
        if features['rms'] > 0:
            features['crest_factor'] = features['max'] / features['rms']
        else:
            features['crest_factor'] = 0
        
        # Form factor
        if np.mean(np.abs(data)) > 0:
            features['form_factor'] = features['rms'] / np.mean(np.abs(data))
        else:
            features['form_factor'] = 0
        
        # Peak factor
        if features['std'] > 0:
            features['peak_factor'] = features['max'] / features['std']
        else:
            features['peak_factor'] = 0
        
        return features
    
    def extract_frequency_domain_features(self, trace):
        """Extract frequency-domain features"""
        data = trace.data
        sampling_rate = trace.stats.sampling_rate
        features = {}
        
        try:
            # Power spectral density using Welch's method
            nperseg = min(1024, len(data) // 4)
            if nperseg < 8:
                nperseg = len(data) // 2
            
            freqs, psd = welch(data, fs=sampling_rate, nperseg=nperseg)
            
            # Remove DC component
            if len(freqs) > 1:
                freqs = freqs[1:]
                psd = psd[1:]
            
            if len(psd) == 0:
                return {f'freq_{key}': 0 for key in ['dominant', 'peak_power', 'centroid', 
                                                   'low_power', 'mid_power', 'high_power',
                                                   'bandwidth', 'spectral_spread', 'spectral_flux']}
            
            # Dominant frequency
            dominant_freq_idx = np.argmax(psd)
            features['freq_dominant'] = freqs[dominant_freq_idx]
            features['freq_peak_power'] = psd[dominant_freq_idx]
            
            # Spectral centroid
            features['freq_centroid'] = np.sum(freqs * psd) / np.sum(psd)
            
            # Frequency band powers
            low_freq_mask = freqs <= 1.0  # Low frequency: 0-1 Hz
            mid_freq_mask = (freqs > 1.0) & (freqs <= 10.0)  # Mid frequency: 1-10 Hz
            high_freq_mask = freqs > 10.0  # High frequency: >10 Hz
            
            features['freq_low_power'] = np.sum(psd[low_freq_mask]) if np.any(low_freq_mask) else 0
            features['freq_mid_power'] = np.sum(psd[mid_freq_mask]) if np.any(mid_freq_mask) else 0
            features['freq_high_power'] = np.sum(psd[high_freq_mask]) if np.any(high_freq_mask) else 0
            
            # Spectral bandwidth (95% of energy)
            cumsum_psd = np.cumsum(psd)
            total_power = cumsum_psd[-1]
            
            if total_power > 0:
                # Find frequencies containing 95% of power
                low_cutoff = np.where(cumsum_psd >= 0.025 * total_power)[0]
                high_cutoff = np.where(cumsum_psd >= 0.975 * total_power)[0]
                
                if len(low_cutoff) > 0 and len(high_cutoff) > 0:
                    features['freq_bandwidth'] = freqs[high_cutoff[0]] - freqs[low_cutoff[0]]
                else:
                    features['freq_bandwidth'] = freqs[-1] - freqs[0]
            else:
                features['freq_bandwidth'] = 0
            
            # Spectral spread
            features['freq_spectral_spread'] = np.sqrt(np.sum(((freqs - features['freq_centroid'])**2) * psd) / np.sum(psd))
            
            # Spectral flux (change in spectrum)
            if len(psd) > 1:
                spectral_flux = np.sum(np.diff(psd)**2)
                features['freq_spectral_flux'] = spectral_flux
            else:
                features['freq_spectral_flux'] = 0
            
        except Exception as e:
            print(f"Error in frequency analysis: {e}")
            # Return default values
            features = {f'freq_{key}': 0 for key in ['dominant', 'peak_power', 'centroid', 
                                                   'low_power', 'mid_power', 'high_power',
                                                   'bandwidth', 'spectral_spread', 'spectral_flux']}
        
        return features
    
    def extract_trigger_features(self, trace):
        """Extract trigger and onset detection features"""
        data = trace.data
        sampling_rate = trace.stats.sampling_rate
        features = {}
        
        try:
            # STA/LTA trigger
            sta_len = int(sampling_rate)  # 1 second
            lta_len = int(sampling_rate * 10)  # 10 seconds
            
            if len(data) > lta_len:
                cft = classic_sta_lta(data, sta_len, lta_len)
                
                features['trigger_max_sta_lta'] = np.max(cft)
                features['trigger_mean_sta_lta'] = np.mean(cft)
                features['trigger_std_sta_lta'] = np.std(cft)
                
                # Count trigger events
                triggers = trigger_onset(cft, 2.0, 1.0)  # trigger on/off
                features['trigger_count'] = len(triggers)
                
                # Maximum trigger duration
                if len(triggers) > 0:
                    durations = [trig[1] - trig[0] for trig in triggers if len(trig) >= 2]
                    features['trigger_max_duration'] = max(durations) if durations else 0
                    features['trigger_mean_duration'] = np.mean(durations) if durations else 0
                else:
                    features['trigger_max_duration'] = 0
                    features['trigger_mean_duration'] = 0
            else:
                features.update({
                    'trigger_max_sta_lta': 0,
                    'trigger_mean_sta_lta': 0,
                    'trigger_std_sta_lta': 0,
                    'trigger_count': 0,
                    'trigger_max_duration': 0,
                    'trigger_mean_duration': 0
                })
            
            # Z-detector (alternative trigger algorithm)
            try:
                z_cft = z_detect(data, int(sampling_rate * 10))
                features['z_detect_max'] = np.max(z_cft) if len(z_cft) > 0 else 0
                features['z_detect_mean'] = np.mean(z_cft) if len(z_cft) > 0 else 0
            except:
                features['z_detect_max'] = 0
                features['z_detect_mean'] = 0
                
        except Exception as e:
            print(f"Error in trigger analysis: {e}")
            features = {
                'trigger_max_sta_lta': 0,
                'trigger_mean_sta_lta': 0,
                'trigger_std_sta_lta': 0,
                'trigger_count': 0,
                'trigger_max_duration': 0,
                'trigger_mean_duration': 0,
                'z_detect_max': 0,
                'z_detect_mean': 0
            }
        
        return features
    
    def extract_wavelet_features(self, trace):
        """Extract wavelet-based features using PyWavelets"""
        data = trace.data
        sampling_rate = trace.stats.sampling_rate
        features = {}
        
        try:
            # Continuous wavelet transform using Morlet wavelet
            # Limit data length for computational efficiency
            if len(data) > 10000:
                data_sample = data[::len(data)//10000]  # Downsample
            else:
                data_sample = data
            
            # Normalize the data
            data_sample = (data_sample - np.mean(data_sample)) / np.std(data_sample)
            
            # Use PyWavelets for continuous wavelet transform
            scales = np.arange(1, 31)  # Scale parameters
            wavelet = 'morl'  # Morlet wavelet
            
            # Perform continuous wavelet transform
            coefficients, frequencies = pywt.cwt(data_sample, scales, wavelet, sampling_period=1.0/sampling_rate)
            
            # Extract features from wavelet coefficients
            features['wavelet_energy'] = np.sum(np.abs(coefficients)**2)
            features['wavelet_max'] = np.max(np.abs(coefficients))
            features['wavelet_mean'] = np.mean(np.abs(coefficients))
            features['wavelet_std'] = np.std(np.abs(coefficients))
            
            # Energy at different scales (frequency ranges)
            low_scale_energy = np.sum(np.abs(coefficients[:10, :])**2)  # Low scales (high freq)
            mid_scale_energy = np.sum(np.abs(coefficients[10:20, :])**2)  # Mid scales
            high_scale_energy = np.sum(np.abs(coefficients[20:, :])**2)  # High scales (low freq)
            
            total_energy = low_scale_energy + mid_scale_energy + high_scale_energy
            if total_energy > 0:
                features['wavelet_low_scale_ratio'] = low_scale_energy / total_energy
                features['wavelet_mid_scale_ratio'] = mid_scale_energy / total_energy
                features['wavelet_high_scale_ratio'] = high_scale_energy / total_energy
            else:
                features['wavelet_low_scale_ratio'] = 0
                features['wavelet_mid_scale_ratio'] = 0
                features['wavelet_high_scale_ratio'] = 0
            
            # Additional wavelet features
            # Peak frequency (scale with maximum energy)
            scale_energies = np.sum(np.abs(coefficients)**2, axis=1)
            peak_scale_idx = np.argmax(scale_energies)
            features['wavelet_peak_scale'] = scales[peak_scale_idx]
            features['wavelet_peak_frequency'] = frequencies[peak_scale_idx] if len(frequencies) > peak_scale_idx else 0
            
            # Wavelet entropy (measure of signal complexity)
            relative_energies = scale_energies / np.sum(scale_energies) if np.sum(scale_energies) > 0 else scale_energies
            relative_energies = relative_energies[relative_energies > 0]  # Remove zeros for log calculation
            features['wavelet_entropy'] = -np.sum(relative_energies * np.log2(relative_energies)) if len(relative_energies) > 0 else 0
                
        except Exception as e:
            print(f"Error in wavelet analysis: {e}")
            features = {
                'wavelet_energy': 0,
                'wavelet_max': 0,
                'wavelet_mean': 0,
                'wavelet_std': 0,
                'wavelet_low_scale_ratio': 0,
                'wavelet_mid_scale_ratio': 0,
                'wavelet_high_scale_ratio': 0,
                'wavelet_peak_scale': 0,
                'wavelet_peak_frequency': 0,
                'wavelet_entropy': 0
            }
        
        return features
    
    def extract_amplitude_features(self, trace):
        """Extract amplitude-based features"""
        data = trace.data
        features = {}
        
        # Peak-to-peak amplitude
        features['amplitude_peak_to_peak'] = np.max(data) - np.min(data)
        
        # Absolute amplitude statistics
        abs_data = np.abs(data)
        features['amplitude_mean_abs'] = np.mean(abs_data)
        features['amplitude_max_abs'] = np.max(abs_data)
        features['amplitude_std_abs'] = np.std(abs_data)
        
        # Amplitude percentiles
        features['amplitude_p90'] = np.percentile(abs_data, 90)
        features['amplitude_p95'] = np.percentile(abs_data, 95)
        features['amplitude_p99'] = np.percentile(abs_data, 99)
        
        # Amplitude ratios
        if features['amplitude_mean_abs'] > 0:
            features['amplitude_max_mean_ratio'] = features['amplitude_max_abs'] / features['amplitude_mean_abs']
        else:
            features['amplitude_max_mean_ratio'] = 0
        
        # Signal-to-noise ratio estimation
        # Assume first 10% is noise, remaining is signal+noise
        noise_length = max(1, len(data) // 10)
        noise_data = data[:noise_length]
        signal_data = data[noise_length:]
        
        noise_power = np.var(noise_data) if len(noise_data) > 1 else 0
        signal_power = np.var(signal_data) if len(signal_data) > 1 else 0
        
        if noise_power > 0:
            features['snr_estimate'] = 10 * np.log10(signal_power / noise_power)
        else:
            features['snr_estimate'] = 0
        
        return features
    
    def extract_trace_features(self, trace, trace_id):
        """Extract all features from a single trace"""
        features = {'trace_id': trace_id}
        
        # Add basic trace information
        features['sampling_rate'] = trace.stats.sampling_rate
        features['npts'] = trace.stats.npts
        features['duration'] = trace.stats.npts / trace.stats.sampling_rate
        features['channel'] = trace.stats.channel
        
        # Extract different types of features
        time_features = self.extract_time_domain_features(trace)
        freq_features = self.extract_frequency_domain_features(trace)
        trigger_features = self.extract_trigger_features(trace)
        wavelet_features = self.extract_wavelet_features(trace)
        amplitude_features = self.extract_amplitude_features(trace)
        
        # Combine all features
        features.update(time_features)
        features.update(freq_features)
        features.update(trigger_features)
        features.update(wavelet_features)
        features.update(amplitude_features)
        
        return features
    
    def process_seismic_file(self, filename):
        """Process a single seismic file and extract features"""
        filepath = os.path.join(self.seismic_data_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return None
        
        try:
            # Read seismic data
            st = read(filepath)
            
            file_features = []
            
            # Process each trace in the stream
            for i, trace in enumerate(st):
                trace_features = self.extract_trace_features(trace, f"{filename}_trace_{i}")
                
                # Add file-level metadata
                trace_features['filename'] = filename
                trace_features['trace_number'] = i
                trace_features['total_traces'] = len(st)
                
                file_features.append(trace_features)
            
            return file_features
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            return None
    
    def extract_all_features(self):
        """Extract features from all seismic files"""
        print("\n=== Extracting Advanced Seismic Features ===")
        
        if not self.metadata:
            print("No metadata available")
            return None
        
        all_features = []
        processed_count = 0
        
        # Get unique filenames from metadata
        filenames = list(set([item['filename'] for item in self.metadata]))
        print(f"Processing {len(filenames)} unique seismic files...")
        
        for filename in filenames:
            print(f"Processing: {filename}")
            
            file_features = self.process_seismic_file(filename)
            if file_features:
                # Add metadata information
                metadata_item = next((item for item in self.metadata if item['filename'] == filename), None)
                if metadata_item:
                    for feature_dict in file_features:
                        feature_dict.update({
                            'earthquake_id': metadata_item['earthquake_id'],
                            'magnitude': metadata_item['magnitude'],
                            'earthquake_time': metadata_item['earthquake_time'],
                            'network': metadata_item['network'],
                            'station': metadata_item['station'],
                            'station_name': metadata_item['station_name']
                        })
                
                all_features.extend(file_features)
                processed_count += 1
            
            if processed_count % 10 == 0:
                print(f"Processed {processed_count}/{len(filenames)} files...")
        
        if all_features:
            self.features_df = pd.DataFrame(all_features)
            print(f"✓ Extracted features from {len(all_features)} traces in {processed_count} files")
            print(f"Feature matrix shape: {self.features_df.shape}")
            
            # Save features
            output_file = os.path.join(self.output_dir, 'advanced_seismic_features.csv')
            self.features_df.to_csv(output_file, index=False)
            print(f"✓ Saved features to: {output_file}")
            
            return self.features_df
        else:
            print("⚠ No features extracted")
            return None
    
    def aggregate_features_by_earthquake(self):
        """Aggregate trace-level features to earthquake-level features"""
        if self.features_df is None or self.features_df.empty:
            print("No features to aggregate")
            return None
        
        print("\n=== Aggregating Features by Earthquake ===")
        
        # Select numeric columns for aggregation
        numeric_cols = self.features_df.select_dtypes(include=[np.number]).columns
        # Exclude ID and metadata columns
        exclude_cols = ['trace_number', 'total_traces', 'sampling_rate', 'npts', 'duration', 'magnitude']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        aggregated_features = []
        
        for earthquake_id, group in self.features_df.groupby('earthquake_id'):
            agg_dict = {
                'earthquake_id': earthquake_id,
                'magnitude': group['magnitude'].iloc[0],
                'earthquake_time': group['earthquake_time'].iloc[0],
                'num_stations': group['station'].nunique(),
                'num_traces': len(group),
                'total_duration': group['duration'].sum()
            }
            
            # Aggregate numeric features using multiple statistics
            for col in numeric_cols:
                if col in group.columns:
                    values = group[col].dropna()
                    if len(values) > 0:
                        agg_dict[f"{col}_mean"] = values.mean()
                        agg_dict[f"{col}_std"] = values.std()
                        agg_dict[f"{col}_max"] = values.max()
                        agg_dict[f"{col}_min"] = values.min()
                        agg_dict[f"{col}_median"] = values.median()
                        
                        # Additional statistics for key features
                        if any(key in col for key in ['energy', 'power', 'amplitude', 'trigger']):
                            agg_dict[f"{col}_range"] = values.max() - values.min()
                            agg_dict[f"{col}_p90"] = values.quantile(0.9)
            
            aggregated_features.append(agg_dict)
        
        aggregated_df = pd.DataFrame(aggregated_features)
        
        # Save aggregated features
        output_file = os.path.join(self.output_dir, 'aggregated_seismic_features.csv')
        aggregated_df.to_csv(output_file, index=False)
        
        print(f"✓ Aggregated features for {len(aggregated_df)} earthquakes")
        print(f"Aggregated feature matrix shape: {aggregated_df.shape}")
        print(f"✓ Saved aggregated features to: {output_file}")
        
        return aggregated_df
    
    def create_feature_summary(self):
        """Create a summary of extracted features"""
        if self.features_df is None:
            return
        
        print("\n=== Feature Summary ===")
        
        # Feature categories
        feature_categories = {
            'Time Domain': [col for col in self.features_df.columns if any(x in col for x in ['mean', 'std', 'max', 'min', 'rms', 'energy', 'skew', 'kurt'])],
            'Frequency Domain': [col for col in self.features_df.columns if 'freq_' in col],
            'Trigger/Onset': [col for col in self.features_df.columns if 'trigger' in col or 'z_detect' in col],
            'Wavelet': [col for col in self.features_df.columns if 'wavelet' in col],
            'Amplitude': [col for col in self.features_df.columns if 'amplitude' in col or 'snr' in col],
            'Metadata': [col for col in self.features_df.columns if col in ['filename', 'earthquake_id', 'magnitude', 'station', 'channel']]
        }
        
        summary = {
            'total_features': len(self.features_df.columns),
            'total_traces': len(self.features_df),
            'unique_earthquakes': self.features_df['earthquake_id'].nunique(),
            'unique_stations': self.features_df['station'].nunique(),
            'feature_categories': {}
        }
        
        for category, features in feature_categories.items():
            summary['feature_categories'][category] = len(features)
            print(f"{category}: {len(features)} features")
        
        # Save summary
        summary_file = os.path.join(self.output_dir, 'seismic_feature_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Feature summary saved to: {summary_file}")
        return summary


def main():
    """Main function to extract advanced seismic features"""
    if not OBSPY_AVAILABLE:
        print("Cannot run without ObsPy. Please install: pip install obspy scipy")
        return
    
    extractor = AdvancedSeismicFeatureExtractor()
    
    # Extract features
    features_df = extractor.extract_all_features()
    
    if features_df is not None:
        # Aggregate features
        extractor.aggregate_features_by_earthquake()
        
        # Create summary
        extractor.create_feature_summary()
        
        print("\n✓ Advanced seismic feature extraction completed successfully!")
    else:
        print("⚠ Feature extraction failed")


if __name__ == "__main__":
    main()
