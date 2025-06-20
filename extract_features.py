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

# Sliding window feature extraction
class GenerateFeatures:
    def __init__(self, fs = 70, window_duration=1.0, overlap=0.8):
        self.window_duration = window_duration
        self.overlap = overlap
        self.fs = fs

        
        self.results = None
        self.features = None
        
        
        
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

    
    def analyze_signal(self, axis, labels, axis_name="axis"):
        """
        Perform sliding window entropy analysis on IMU signal
        """
        signal = np.array(axis)
        n_samples = len(signal)
        
        if n_samples == 0:
            raise ValueError("Signal is empty. Cannot perform analysis.")

        
        window_samples = int(self.window_duration * self.fs)
        step_samples = int(window_samples * (1 - self.overlap))
        
        # Storage for results
        window_features = []
        self.all_labels: list = []
        
        # Sliding window analysis
        for start_idx in range(0, n_samples - window_samples, step_samples):
            end_idx = start_idx + window_samples
            window_signal = signal[start_idx:end_idx] 
            
            # Calculate different entropy measures
            perm_ent = entropy_permutation(window_signal)[0]
            spectral_ent = entropy_spectral(window_signal)[0]
            
            # Calculate energy features
            time_energy = self.calculate_energy(window_signal)
            spectral_energy = self.calculate_spectral_energy(window_signal)

            
            # Assign the label with the highest occurrence to the window
            if labels is not None:
                window_labels = labels[start_idx:end_idx]
                label = pd.Series(window_labels).mode()[0]
                self.all_labels.append(label)
            else:
                label = np.nan
            # Store results
            window_features.append({
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
            })
        
        return (pd.DataFrame(window_features))
            
            
    def analyze_multi_axis_imu(self,df):
        """
        Analyze all IMU axes and combine results
        """
        # Standardize IMU data (z-score normalization)
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input df data must be a pandas DataFrame.")
        df = df.copy()
        df['acc_x'] = (df['acc_x'] - df['acc_x'].mean()) / df['acc_x'].std()
        df['acc_y'] = (df['acc_y'] - df['acc_y'].mean()) / df['acc_y'].std()
        df['acc_z'] = (df['acc_z'] - df['acc_z'].mean()) / df['acc_z'].std()
        df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
        
        acc_x = df['acc_x'] 
        acc_y = df['acc_y']
        acc_z = df['acc_z']
        labels = df['label']
        
        signals = {
            'acc_x': acc_x,
            'acc_y': acc_y,
            'acc_z': acc_z,
            'acc_mag': df['acc_mag']
        }
            
        all_results = []
            
        for signal_name, signal_data in tqdm(signals.items(), desc='Analyzing '):
            # print(f"Analyzing {signal_name}...")
            result_df = self.analyze_signal(signal_data, labels, signal_name)
            all_results.append(result_df)
            
        # Combine all results
        self.features = pd.concat(all_results, ignore_index=True)
            
        # Create summary pivot table for easier analysis
        self.results = self.create_summary_table()
        
            
        return self.features, self.results
    
        
    def create_summary_table(self):
        """
        Create a summary table with entropy measures as columns
        """
        if self.features is None:
            return None
            
        # Pivot table with signals as columns for each entropy measure
        # measures = ['spectral_entropy','signal_mean','signal_std','signal_range','signal_rms','signal_var', 'spectral_energy', 'time_energy']
        measures = ['permutation_entropy', 'spectral_entropy', 'mean', 'std', 
                    'range', 'rms', 'var', 'min', 'max', 'time_energy', 'spectral_energy'
                    ]  
        summary_data = []
            
        # Get unique time windows
        unique_times = self.features['center_time'].unique()
            

        for time_point in unique_times:
            time_data = self.features[self.features['center_time'] == time_point]
                
            row = {'center_time': time_point, 'start_time': time_data.iloc[0]['start_time'], 'end_time': time_data.iloc[0]['end_time']}
                
            # Add features for each signal
            for _, signal_row in time_data.iterrows():
                signal_name = signal_row['signal_name']
                for measure in measures:
                    col_name = f"{signal_name}_{measure}"
                    row[col_name] = signal_row[measure]
                
            summary_data.append(row)
            df = pd.DataFrame(summary_data).sort_values('center_time').reset_index(drop=True)
            if df.shape[0] == len(self.all_labels):
                df['label'] = self.all_labels
        return df
                