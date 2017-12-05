import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftshift, fftfreq
from scipy.signal import hamming, hann, kaiser

def to_seconds(positions, fs):
	return np.multiply(positions, 1.0/fs)
	
def to_microvolts(values):
    return np.multiply(values, 0.51)

#Get data for each header
data = pd.read_csv('Records/EEG_videos_i2.csv', sep=',', header=1)
data = data.as_matrix()
data = np.delete(data, np.s_[-1:], axis=1)

#(N/fs)-second(s) domain
fs = 128
N = 1280
##10-second groups at 128Hz
groups = data.shape[0] / N
n1 = to_seconds(np.arange(0, data.shape[0]), fs)
npwr = n1[640:-N:N]
n2 = fftfreq(N, 1.0/fs)
nfreq = n2[:N/2]

plt.xlabel('Time (seconds)')
plt.ylabel('Raw data\n(one sensor)')
plt.plot(n1, data, 'k')

plt.show()
