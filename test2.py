# test2.py - Jeff Wang - Tony Ho
# cd mnt/c/users/17789/desktop/353/final

# python3 test2.py Car-500m-30km.csv


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy import signal

# Butterfly filter modified from: https://ggbaker.ca/data-science/content/filtering.html#filtering
def low_pass_fliter(noisy_signal):
    b, a = signal.butter(3, .9, btype='lowpass', analog=False)
    low_passed = signal.filtfilt(b, a, noisy_signal)
    return low_passed



def main(in_dir):
    #print(in_dir)
    df = pd.read_csv('Car-500m-30km.csv')
    df.loc[(df['time'] > 0.05) & (df['time'] < 2)].reset_index(drop=True)

    # Plot ax, ay, az, at into a scatter plot
    plt.plot(df['time'], df['ax'], 'b.', alpha = 0.5)
    plt.plot(df['time'], df['ay'], 'r.', alpha = 0.5)
    plt.plot(df['time'], df['az'], 'g.', alpha = 0.5)
    plt.plot(df['time'], df['at'], 'y.', alpha = 0.5)
    plt.savefig('plot1.png')
    plt.clf()

    # Apply the Butterfly filter
    df['ax'] = low_pass_fliter(df['ax'])
    df['ay'] = low_pass_fliter(df['ay'])
    df['az'] = low_pass_fliter(df['az'])
    df['at'] = low_pass_fliter(df['at'])
    plt.plot(df['time'], df['ax'], 'b', alpha = 0.5)
    plt.plot(df['time'], df['ay'], 'r', alpha = 0.5)
    plt.plot(df['time'], df['az'], 'g', alpha = 0.5)
    plt.plot(df['time'], df['at'], 'y', alpha = 0.5)
    plt.savefig('plot2.png')
    plt.clf()
    



if __name__ == '__main__':
    main(sys.argv[1])

