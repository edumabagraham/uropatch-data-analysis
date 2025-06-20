import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from scipy.signal import resample

    
def resample_data(data: dict, target_fs: int = 70):
    """Resamples time-series data within a dictionary of pandas DataFrames.

    This function iterates through each DataFrame in the input dictionary,
    removes specified columns, and then resamples the remaining data to a
    target sampling frequency.

    Note:
        - The function modifies the input DataFrames in place by dropping columns.
        - It assumes the `resample` function is from `scipy.signal`.

    Args:
        data (dict): A dictionary where keys are unique identifiers and values
            are pandas DataFrames. Each DataFrame must contain a 'time' column.
        target_fs (int, optional): The target sampling frequency in Hz.
            Defaults to 70.

    Returns:
        dict: A new dictionary with the same keys as the input, containing
            the resampled DataFrames.
    """
    resampled_dict = {}
    for i, void_instance in tqdm(enumerate(data.keys())):
        old_df = data[void_instance]
        old_df.drop(columns=['Real time', 'gyr_x', 'gyr_y', 'gyr_z'], axis = 1, inplace=True)
        # old_df.drop(columns=['Real time'], axis = 1, inplace=True)
        
        original_fs = 1 / old_df['time'].diff().mean()
        num_samples = int(len(old_df) * target_fs / original_fs)
        
        resampled_data = resample(old_df, num_samples)
        
        resampled_dict[void_instance] = pd.DataFrame(resampled_data, columns=old_df.columns )
    return resampled_dict


def add_labels(df: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    """Adds a 'label' column to an IMU DataFrame based on a ground truth time range.

    This function identifies the "void" period using the start and end timestamps from the ground truth DataFrame. It then
    classifies each row in the IMU DataFrame into one of three categories:
    'pre-void', 'void', or 'post-void', based on whether the IMU timestamp
    occurs before, during, or after this event period.

    Args:
        df: The DataFrame containing IMU (Inertial Measurement Unit) data.
                It must include a 'time' column with timestamp values.
        gt: The ground truth DataFrame that provides the reference time range.
            It must have a 'Time' column, where the first element marks the
            start and the last element marks the end of the "void" event.

    Returns:
        The `df` DataFrame, modified in place to include a new 'label'
        column that contains the classification for each row.
    """
    ue = [gt['Time'].iloc[0], gt['Time'].iloc[-1]]
    pre_void_labels = [f'pre-void' for i in (df[(df['time'] < ue[0])])['time']]
    void_labels = [f'void' for i in (df[(df['time'] >= ue[0]) & (df['time'] <= ue[1])])['time']]
    post_void_labels = [f'post-void' for i in (df[(df['time'] > ue[1])])['time']]
    
    labels = pre_void_labels + void_labels + post_void_labels
    df['label'] = labels
    return df