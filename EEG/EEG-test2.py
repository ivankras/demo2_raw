import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftshift, fftfreq
from scipy.signal import hann, hamming, kaiser

def to_volts(values):
	return np.multiply(values, 0.51 * np.power(0.1, 6))

def to_seconds(positions, fs):
	return np.multiply(positions, float(1)/fs)

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
data = pd.read_csv('Records/EEGLogger.csv', sep=',', header=1)
data = data.as_matrix()
data = np.delete(data, np.s_[-1:], axis=1)

#(N/fs)-second(s) domain
fs = 128
N = 4096
n1 = to_seconds(np.arange(0, N), fs)
n2 = fftfreq(N, float(1)/fs)
nfreq = fftshift(n2 * float(fs)/N)

#Highpass filter options
#  Hamming filter, [N] points
##H = hamming(N)
#  Hann filter, [N] points
##H = hann(N)
#  Kaiser filter, [N] points, beta=[5.4]
##H = kaiser(N, 7.4)
#  Rectangle filter, [N] points, 7 freq-points deleted
##H = np.concatenate([np.zeros(4), np.ones(N-7), np.zeros(3)])

#Sample data for two seconds
s_data = to_volts(data[1024:5120,idx_dict['IED_AF4']])

#Transform data
t_data = fft(s_data, N)

#Original power terms
t_data_conjugates = np.conjugate(t_data)
t_data_power = np.multiply(t_data, t_data_conjugates)

#Reorganize data for visualization
t_data_non_filtered = fftshift(t_data)
t_data_power_visual = fftshift(t_data_power)

#Apply highpass filter to data
t_data_filtered = np.multiply(t_data, H)

#Filtered data to power terms
t_data_conjugates = np.conjugate(t_data_filtered)
t_data_filtered_power = np.multiply(t_data_filtered, t_data_conjugates)

#Reorganize filtered data for visualization
t_data_filtered_visual = fftshift(t_data_filtered)
t_data_filtered_power_visual = fftshift(t_data_filtered_power)

#Filtered data back to time
new_s_voltage_data = ifft(t_data_filtered)

#Plotting options

plt.figure(1)

##plt.xlabel('time [s]')
##plt.xlabel('frequency [Hz]')
##plt.ylabel('voltage [V]')
##plt.ylabel('power [V*V]')

##plt.plot(n1, H, 'b')
##plt.plot(n1, s_data, 'r')
##plt.plot(n1, t_data, 'r')
##plt.plot(n2, t_data_non_filtered, 'r')
##plt.plot(n2, t_data_filtered_visual, 'k')
##plt.plot(nfreq, t_data_power_visual, 'g')

##plt.figure(2)

##plt.xlabel('time [s]')
##plt.xlabel('frequency [Hz]')
##plt.ylabel('voltage [V]')
##plt.ylabel('power [V*V]')

plt.plot(nfreq, t_data_filtered_power_visual, 'k')
##plt.plot(n1, new_s_voltage_data, 'r')

#plt.show()
