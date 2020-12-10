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

