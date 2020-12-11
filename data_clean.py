# data_clean.py - Jeff Wang - Tony Ho
# cd mnt/c/users/17789/desktop/353/final

# python3 data_clean.py


import numpy as np
import pandas as pd
from scipy import signal
import re
import os
import glob


# Referenced from: https://stackoverflow.com/questions/37372603/how-to-remove-specific-substrings-from-a-set-of-strings-in-python
def getfilename(path):
    return path.replace('./raw data/','')


def trimtime(df):
    trimtime = 7
    return df.loc[(df['time'] > trimtime) & (df['time'] < (df['time'].max() - trimtime) )].reset_index(drop=True)


# Butterfly filter modified from: https://ggbaker.ca/data-science/content/filtering.html#filtering
def low_pass_fliter(noisy_signal):
    b, a = signal.butter(3, .9, btype='lowpass', analog=False)
    low_passed = signal.filtfilt(b, a, noisy_signal)
    return low_passed


def trimnoise(df):
    df['ax_f'] = low_pass_fliter(df['ax'])
    df['ay_f'] = low_pass_fliter(df['ay'])
    df['az_f'] = low_pass_fliter(df['az'])
    df['wx_f'] = low_pass_fliter(df['wx'])
    df['wy_f'] = low_pass_fliter(df['wy'])
    df['wz_f'] = low_pass_fliter(df['wz'])
    df['gFx_f'] = low_pass_fliter(df['gFx'])
    df['gFy_f'] = low_pass_fliter(df['gFy'])
    df['gFz_f'] = low_pass_fliter(df['gFz'])
    return df

def clean_data(filename):
    df = pd.read_csv(filename)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.rename(columns={'Speed (m/s)': "speed"})
    df = df.drop(df[df.Latitude == 0.0].index)
    
    df['v3'] = df.apply(lambda row: (row['ax']**2 + row['ay']**2 + row['az']**2)**0.5, axis=1)
    
    df['time'] = df['time'].diff()
    
    groupSize = 12
    grouped = df.groupby(df.index // groupSize)
    df_grouped = grouped.mean()
    
    df_grouped['time'] = grouped.sum()['time']
    
    return df_grouped
    

def get_step_count(series):
    series = series.to_numpy()
    peaks,_ = signal.find_peaks(series)
    return len(peaks)

def get_features(data, size):
    
    data['distance'] = getDistanceFromLatLon(data['Latitude'], data['Longitude'])
    
    #grouped = data.groupby(data.index // size)
    #df_features = grouped.sum()
    #df_features['steps'] = grouped['v3'].apply(get_step_count)
    #df_features['speed'] = grouped['speed'].mean()
    rollingData = data.rolling(size)
    df_features = rollingData.sum()
    df_features['steps'] = rollingData['v3'].apply(get_step_count)
    df_features['speed'] = rollingData['speed'].mean()
    
    df_features = df_features.loc[df_features['speed'] > 0.01]
    
    df_features = df_features.dropna()
    
    return df_features[['speed','distance','steps', 'time']]
    
def getDistanceFromLatLon(lat,lon):
    R = 6371; # Radius of the earth in km
    dLat = deg2rad((lat.shift(-1) - lat).dropna());  # deg2rad below
    dLon = deg2rad((lon.shift(-1) - lon).dropna()); 
    a = np.sin(dLat/2) * np.sin(dLat/2) + np.cos(deg2rad(lat)) * np.cos(deg2rad(lat.shift(-1))) * np.sin(dLon/2) * np.sin(dLon/2)
    
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a)); 
    d = R * c * 1000; # Distance in meters
    return d


def deg2rad(deg):
    return deg * (np.pi/180)
    
def createDataframe(filename, group_size, height):
    df = clean_data(filename)
    df_feat = get_features(df, group_size)
    df_feat['height'] = height
    
    return df_feat
    
def main():
    # Referenced from: https://stackoverflow.com/questions/20906474/import-multiple-csv-files-into-pandas-and-concatenate-into-one-dataframe
    files = glob.glob('./raw data/*.csv')

    os.mkdir('clean data')
    for i in files:
        df = pd.read_csv(i).dropna(axis='columns')
        df = trimtime(df)
        df = trimnoise(df)
        df.to_csv(os.path.join(f'./clean data/{getfilename(i)}'), index=False)


if __name__ == '__main__':
    main()

