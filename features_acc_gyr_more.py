import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
import pickle
import os
from neurokit2 import entropy_sample, entropy_shannon, entropy_permutation, entropy_spectral
import statsmodels.tsa.stattools as stattools
import scipy.signal as signal
from scipy.signal import correlate
from scipy import stats

# Sliding window feature extraction
class GenerateFeatures:
    def __init__(self, fs=70, window_duration=1.0, overlap=0.8):
        self.window_duration = window_duration
        self.overlap = overlap
        self.fs = fs
        
        self.results = None
        self.features = None
        
        # Validation
        if self.window_duration <= 0:
            raise ValueError("Window duration must be positive")
        if not (0 <= self.overlap < 1):
            raise ValueError("Overlap must be between 0 and 1 (exclusive)")
            
        window_samples = int(self.window_duration * self.fs)
        if window_samples < 3:
            raise ValueError("Window too small - need at least 3 samples for entropy calculation")
        
    def calculate_energy(self, signal_data):
        """
        Calculate signal energy in time domain
        """
        return np.sum(signal_data**2)
    
    def calculate_spectral_energy(self, signal_data):
        """
        Calculate spectral energy using FFT
        """
        fft_vals = np.fft.fft(signal_data)
        spectral_energy = np.sum(np.abs(fft_vals)**2)
        return spectral_energy

    def calculate_frequency_features(self, signal_data):
        """Enhanced frequency domain features"""
        try:
            freqs, psd = signal.welch(signal_data, fs=self.fs, nperseg=min(len(signal_data), 256))
            
            # Avoid division by zero
            if np.sum(psd) == 0:
                return {
                    'spectral_centroid': np.nan,
                    'spectral_rolloff': np.nan,
                    'spectral_bandwidth': np.nan,
                    'dominant_frequency': np.nan
                }
            
            # Spectral centroid (frequency "center of mass")
            spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
            
            # Spectral rolloff (frequency below which 85% of energy lies)
            cumulative_psd = np.cumsum(psd)
            rolloff_idx = np.where(cumulative_psd >= 0.85 * cumulative_psd[-1])[0]
            spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else np.nan
            
            # Spectral bandwidth
            spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd) / np.sum(psd))
            
            # Dominant frequency
            dominant_freq = freqs[np.argmax(psd)]
            
            return {
                'spectral_centroid': spectral_centroid,
                'spectral_rolloff': spectral_rolloff,
                'spectral_bandwidth': spectral_bandwidth,
                'dominant_frequency': dominant_freq
            }
        except Exception as e:
            print(f"Warning: Frequency feature calculation failed: {e}")
            return {
                'spectral_centroid': np.nan,
                'spectral_rolloff': np.nan,
                'spectral_bandwidth': np.nan,
                'dominant_frequency': np.nan
            }

    def calculate_temporal_features(self, signal_data):
        """Features that capture temporal changes"""
        try:
            # Zero crossing rate
            zero_crossings = np.where(np.diff(np.sign(signal_data)))[0]
            zero_crossing_rate = len(zero_crossings) / len(signal_data)
            
            # Signal slope/trend
            time_indices = np.arange(len(signal_data))
            slope = np.polyfit(time_indices, signal_data, 1)[0] if len(signal_data) > 1 else 0
            
            # Autocorrelation at lag 1 (measures predictability)
            if len(signal_data) > 1:
                autocorr_lag1 = np.corrcoef(signal_data[:-1], signal_data[1:])[0, 1]
                autocorr_lag1 = autocorr_lag1 if not np.isnan(autocorr_lag1) else 0
            else:
                autocorr_lag1 = 0
            
            # Peak count and prominence
            peaks, properties = signal.find_peaks(signal_data, height=np.std(signal_data))
            peak_count = len(peaks)
            
            return {
                'zero_crossing_rate': zero_crossing_rate,
                'slope': slope,
                'autocorr_lag1': autocorr_lag1,
                'peak_count': peak_count
            }
        except Exception as e:
            print(f"Warning: Temporal feature calculation failed: {e}")
            return {
                'zero_crossing_rate': np.nan,
                'slope': np.nan,
                'autocorr_lag1': np.nan,
                'peak_count': np.nan
            }

    def calculate_shape_features(self, signal_data):
        """Statistical shape descriptors"""
        try:
            # Skewness (asymmetry)
            skewness = stats.skew(signal_data)
            
            # Kurtosis (tail heaviness)
            kurtosis = stats.kurtosis(signal_data)
            
            # Interquartile range
            iqr = np.percentile(signal_data, 75) - np.percentile(signal_data, 25)
            
            # Median absolute deviation
            mad = np.median(np.abs(signal_data - np.median(signal_data)))
            
            return {
                'skewness': skewness,
                'kurtosis': kurtosis,
                'iqr': iqr,
                'mad': mad
            }
        except Exception as e:
            print(f"Warning: Shape feature calculation failed: {e}")
            return {
                'skewness': np.nan,
                'kurtosis': np.nan,
                'iqr': np.nan,
                'mad': np.nan
            }

    def analyze_signal(self, axis, labels, axis_name="axis"):
        """
        Perform sliding window entropy analysis on IMU signal
        """
        signal_data = np.array(axis)
        n_samples = len(signal_data)
        
        if n_samples == 0:
            raise ValueError("Signal is empty. Cannot perform analysis.")

        window_samples = int(self.window_duration * self.fs)
        step_samples = int(window_samples * (1 - self.overlap))
        
        if step_samples == 0:
            step_samples = 1  # Prevent infinite loop
        
        # Storage for results
        window_features = []
        window_labels = []
        
        # Sliding window analysis
        for start_idx in range(0, n_samples - window_samples + 1, step_samples):
            end_idx = start_idx + window_samples
            window_signal = signal_data[start_idx:end_idx] 
            
            try:
                # Calculate different entropy measures
                perm_ent = entropy_permutation(window_signal)[0]
                spectral_ent = entropy_spectral(window_signal)[0]
            except Exception as e:
                print(f"Warning: Entropy calculation failed for window {start_idx}-{end_idx}: {e}")
                perm_ent = np.nan
                spectral_ent = np.nan
            
            # Calculate energy features
            time_energy = self.calculate_energy(window_signal)
            spectral_energy = self.calculate_spectral_energy(window_signal)
            
            # Calculate new frequency features
            freq_features = self.calculate_frequency_features(window_signal)
            
            # Calculate temporal features
            temporal_features = self.calculate_temporal_features(window_signal)
            
            # Calculate shape features
            shape_features = self.calculate_shape_features(window_signal)

            # Assign the label with the highest occurrence to the window
            if labels is not None:
                window_label_data = labels[start_idx:end_idx]
                label = pd.Series(window_label_data).mode()[0]
                window_labels.append(label)
            else:
                label = np.nan
                window_labels.append(label)
            
            # Store results - combine all features
            features_dict = {
                'start_idx': start_idx,
                'end_idx': end_idx,
                'start_time': start_idx / self.fs,
                'end_time': end_idx / self.fs,
                'center_time': (start_idx + end_idx) / 2 / self.fs,
                'signal_name': axis_name,
                'permutation_entropy': perm_ent,
                'spectral_entropy': spectral_ent,
                'mean': np.mean(window_signal),
                'std': np.std(window_signal),
                'range': np.max(window_signal) - np.min(window_signal),
                'rms': np.sqrt(np.mean(window_signal**2)),
                'var': np.var(window_signal),
                'min': np.min(window_signal),
                'max': np.max(window_signal),
                'time_energy': time_energy,
                'spectral_energy': spectral_energy,
                'label': label
            }
            
            # Add new features
            features_dict.update(freq_features)
            features_dict.update(temporal_features)
            features_dict.update(shape_features)
            
            window_features.append(features_dict)
        
        return pd.DataFrame(window_features), window_labels

    def calculate_cross_axis_features(self, acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z):
        """Features that capture relationships between axes"""
        try:
            # Correlation between acceleration axes
            acc_xy_corr = np.corrcoef(acc_x, acc_y)[0, 1] if len(acc_x) > 1 else 0
            acc_xz_corr = np.corrcoef(acc_x, acc_z)[0, 1] if len(acc_x) > 1 else 0
            acc_yz_corr = np.corrcoef(acc_y, acc_z)[0, 1] if len(acc_y) > 1 else 0
            
            # Correlation between gyroscope axes
            gyr_xy_corr = np.corrcoef(gyr_x, gyr_y)[0, 1] if len(gyr_x) > 1 else 0
            gyr_xz_corr = np.corrcoef(gyr_x, gyr_z)[0, 1] if len(gyr_x) > 1 else 0
            gyr_yz_corr = np.corrcoef(gyr_y, gyr_z)[0, 1] if len(gyr_y) > 1 else 0
            
            # Signal vector magnitude area (SVMA)
            svma = np.mean(np.abs(acc_x) + np.abs(acc_y) + np.abs(acc_z))
            
            # Angle between acceleration vectors (if looking at orientation changes)
            acc_vector_angle = np.arctan2(acc_y, acc_x)
            angle_variance = np.var(acc_vector_angle)
            
            # Cross-correlation between acc and gyr magnitudes
            acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
            gyr_mag = np.sqrt(gyr_x**2 + gyr_y**2 + gyr_z**2)
            acc_gyr_corr = np.corrcoef(acc_mag, gyr_mag)[0, 1] if len(acc_mag) > 1 else 0
            
            # Handle NaN values
            features = {
                'acc_xy_corr': acc_xy_corr if not np.isnan(acc_xy_corr) else 0,
                'acc_xz_corr': acc_xz_corr if not np.isnan(acc_xz_corr) else 0,
                'acc_yz_corr': acc_yz_corr if not np.isnan(acc_yz_corr) else 0,
                'gyr_xy_corr': gyr_xy_corr if not np.isnan(gyr_xy_corr) else 0,
                'gyr_xz_corr': gyr_xz_corr if not np.isnan(gyr_xz_corr) else 0,
                'gyr_yz_corr': gyr_yz_corr if not np.isnan(gyr_yz_corr) else 0,
                'svma': svma,
                'angle_variance': angle_variance,
                'acc_gyr_corr': acc_gyr_corr if not np.isnan(acc_gyr_corr) else 0
            }
            
            return features
        except Exception as e:
            print(f"Warning: Cross-axis feature calculation failed: {e}")
            return {
                'acc_xy_corr': np.nan,
                'acc_xz_corr': np.nan,
                'acc_yz_corr': np.nan,
                'gyr_xy_corr': np.nan,
                'gyr_xz_corr': np.nan,
                'gyr_yz_corr': np.nan,
                'svma': np.nan,
                'angle_variance': np.nan,
                'acc_gyr_corr': np.nan
            }
            
    def analyze_multi_axis_imu(self, df):
        """
        Analyze all IMU axes and combine results
        """
        # Standardize IMU data (z-score normalization)
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input df data must be a pandas DataFrame.")
        
        required_cols = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'label']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        df = df.copy()
        df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
        df['gyr_mag'] = np.sqrt(df['gyr_x']**2 + df['gyr_y']**2 + df['gyr_z']**2)
        
        acc_x = df['acc_x'] 
        acc_y = df['acc_y']
        acc_z = df['acc_z']
        gyr_x = df['gyr_x'] 
        gyr_y = df['gyr_y']
        gyr_z = df['gyr_z']
        labels = df['label']
        
        signals = {
            'acc_x': acc_x,
            'acc_y': acc_y,
            'acc_z': acc_z,
            'acc_mag': df['acc_mag'],
            'gyr_x': gyr_x,
            'gyr_y': gyr_y,
            'gyr_z': gyr_z,
            'gyr_mag': df['gyr_mag']
        }
            
        all_results = []
        # Store labels from first signal only to avoid duplication
        self.all_labels = None
            
        for signal_name, signal_data in tqdm(signals.items(), desc='Analyzing '):
            result_df, window_labels = self.analyze_signal(signal_data, labels, signal_name)
            all_results.append(result_df)
            
            # Store labels from first signal analysis
            if self.all_labels is None:
                self.all_labels = window_labels
            
        # Combine all results
        self.features = pd.concat(all_results, ignore_index=True)
            
        # Create summary pivot table for easier analysis
        self.results = self.create_summary_table()
        
        # Add cross-axis features to summary table
        self.add_cross_axis_features_to_summary(df)
        
        return self.features, self.results

    def add_cross_axis_features_to_summary(self, df):
        """Add cross-axis features to the summary table"""
        if self.results is None:
            return
        
        # Get window parameters
        window_samples = int(self.window_duration * self.fs)
        step_samples = int(window_samples * (1 - self.overlap))
        if step_samples == 0:
            step_samples = 1
        
        cross_axis_features = []
        
        # Calculate cross-axis features for each window
        for start_idx in range(0, len(df) - window_samples + 1, step_samples):
            end_idx = start_idx + window_samples
            
            window_acc_x = df['acc_x'].iloc[start_idx:end_idx].values
            window_acc_y = df['acc_y'].iloc[start_idx:end_idx].values
            window_acc_z = df['acc_z'].iloc[start_idx:end_idx].values
            window_gyr_x = df['gyr_x'].iloc[start_idx:end_idx].values
            window_gyr_y = df['gyr_y'].iloc[start_idx:end_idx].values
            window_gyr_z = df['gyr_z'].iloc[start_idx:end_idx].values
            
            features = self.calculate_cross_axis_features(
                window_acc_x, window_acc_y, window_acc_z,
                window_gyr_x, window_gyr_y, window_gyr_z
            )
            cross_axis_features.append(features)
        
        # Add cross-axis features to results DataFrame
        if len(cross_axis_features) == len(self.results):
            for feature_name, feature_values in zip(
                cross_axis_features[0].keys(), 
                zip(*[f.values() for f in cross_axis_features])
            ):
                self.results[feature_name] = feature_values
        else:
            print(f"Warning: Cross-axis feature count mismatch. Expected {len(self.results)}, got {len(cross_axis_features)}")
    
    def create_summary_table(self):
        """
        Create a summary table with entropy measures as columns
        """
        if self.features is None:
            return None
            
        # Updated measures list with new features
        measures = ['permutation_entropy', 'spectral_entropy', 'mean', 'std', 
                    'range', 'rms', 'var', 'min', 'max', 'time_energy', 'spectral_energy',
                    'spectral_centroid', 'spectral_rolloff', 'spectral_bandwidth', 'dominant_frequency',
                    'zero_crossing_rate', 'slope', 'autocorr_lag1', 'peak_count',
                    'skewness', 'kurtosis', 'iqr', 'mad']  
        
        # Get unique time windows
        unique_times = sorted(self.features['center_time'].unique())
        
        summary_data = []
        
        for time_point in unique_times:
            time_data = self.features[self.features['center_time'] == time_point]
                
            row = {
                'center_time': time_point, 
                'start_time': time_data.iloc[0]['start_time'], 
                'end_time': time_data.iloc[0]['end_time']
            }
                
            # Add features for each signal
            for _, signal_row in time_data.iterrows():
                signal_name = signal_row['signal_name']
                for measure in measures:
                    col_name = f"{signal_name}_{measure}"
                    row[col_name] = signal_row[measure]
                
            summary_data.append(row)
        
        # Create DataFrame outside the loop
        df = pd.DataFrame(summary_data).sort_values('center_time').reset_index(drop=True)
        
        # Add labels if available
        if self.all_labels is not None and len(self.all_labels) == len(df):
            df['label'] = self.all_labels
        elif self.all_labels is not None:
            print(f"Warning: Label count mismatch. Expected {len(df)}, got {len(self.all_labels)}")
            
        return df