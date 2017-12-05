import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftshift, fftfreq
from scipy.signal import hann, hamming, kaiser

def to_seconds(positions, fs):
	return np.multiply(positions, 1.0/fs)
	
def to_microvolts(values):
    return np.multiply(values, 0.51)

"""
#Get headers' names
headers = np.genfromtxt('EEGLogger.csv', dtype=None, delimiter=',')[0]
field_limit = len(headers) - 1
headers = np.delete(headers, field_limit)

#Define a confortable structure for accesing data
cont = 0
idx_dict={}
for field in headers:
	idx_dict[field] = cont
	cont += 1
"""

idx_dict = {'IED_COUNTER':0,
            'IED_INTERPOLATED':1,
            'IED_AF3':2,
            'IED_F7':3,
            'IED_F3':4,
            'IED_FC5':5,
            'IED_T7':6,
            'IED_P7':7,
            'IED_O1':8,
            'IED_O2':9,
            'IED_P8':10,
            'IED_T8':11,
            'IED_FC6':12,
            'IED_F4':13,
            'IED_F8':14,
            'IED_AF4':15,
            'IED_RAW_CQ':16}

freq_dict = {'ALPHA_MAX':12,
             'ALPHA_MIN':8,
             'BETA_MAX':40,
             'BETA_MIN':12,
             'DELTA_MAX':4,
             'DELTA_MIN':0,
             'GAMMA_MAX':100,
             'GAMMA_MIN':40,
             'THETA_MAX':8,
             'THETA_MIN':4}

#Get data for each header
data = pd.read_csv('Records/EEG_videos_i2.csv', sep=',', header=1)
data = data.as_matrix()
data = np.delete(data, np.s_[-1:], axis=1)

#(N/fs)-second(s) domain
fs = 128
N = 1280
##10-second groups at 128Hz
groups = data.shape[0] / N
n1 = to_seconds(np.arange(1280, 2560), fs)
#npwr = n1[640:-N:N]
n2 = fftfreq(1280, 1.0/fs)
nfreq = n2[:N/2]

#Highpass filter options
#  Hamming filter, [N] points
##H = hamming(N)
#  Hann filter, [N] points
H = np.concatenate([np.zeros(30), hann(N-60), np.zeros(30)])
#  Kaiser filter, [N] points, beta=[5.4]
##H = kaiser(N, 7.4)
#  Rectangle filter, [N] points, 7 freq-points deleted
##H = np.concatenate([np.zeros(4), np.ones(N-7), np.zeros(3)])

#Sample data for two seconds
s_data = to_microvolts(data[1280:2560,idx_dict['IED_AF4']])

#Transform data
t_data = fft(s_data, N)
t_data /= float(N)
t_data = np.multiply(t_data, H)

#Plotting options

plt.xlabel('Time (seconds)')
plt.ylabel('Raw data\n(one sensor)')
plt.plot(n1, s_data, 'b')

#plt.xlabel('Frequency (Hertz)')
#plt.plot(nfreq, t_data[:N/2], 'm')

plt.show()

