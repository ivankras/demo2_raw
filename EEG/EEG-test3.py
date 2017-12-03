import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftshift, fftfreq
from scipy.signal import hamming, hann, kaiser

def to_volts(values):
	return np.multiply(values, 0.51 * np.power(0.1, 6))

def to_seconds(positions, fs):
	return np.multiply(positions, float(1)/fs)
	
def get_bands(power_frecs, nfreqs):
    alpha = 0
    beta = 0
    gamma = 0
    delta = 0
    theta = 0
	
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
N = 1280
n1 = to_seconds(np.arange(0, N), fs)
n2 = fftfreq(N, float(1)/fs)
nfreq = n2[:N/2]

#Highpass filter
##h = ifft(kaiser(N, 5.4))
h = np.concatenate([np.zeros(30), hann(N-60), np.zeros(30)]) 

#Sample data
s_data = to_volts(data[512:1792,idx_dict['IED_AF4']])
##s_data = np.concatenate([s_data, np.zeros(N-len(s_data))])
##print s_data

#Transform data
t_data = fft(s_data, N)
t_data /= float(N)

#Apply highpass filter to data
t_data_filtered = np.multiply(t_data, h)

#Power terms
t_data_conjugates = np.conjugate(t_data_filtered)
t_data_power = np.multiply(t_data_filtered, t_data_conjugates)[:N/2]
##t_data_power = np.power(t_data_filtered, 2)

alphas = np.setdiff1d(nfreq[nfreq<12], nfreq[nfreq<8])
betas = np.setdiff1d(nfreq[nfreq<40], nfreq[nfreq<12])
gammas = np.setdiff1d(nfreq[nfreq<=100], nfreq[nfreq<40]) #Unless I get more samples (256), only until 64 Hz
deltas = nfreq[nfreq<4]
thetas = np.setdiff1d(nfreq[nfreq<8], nfreq[nfreq<4])

pwr_alpha = 0
pwr_beta = 0
pwr_gamma = 0
pwr_delta = 0
pwr_theta = 0
i = 0

for delta in deltas:
	pwr_delta += t_data_power[i]
	i += 1
pwr_delta /= (float)0.5 * N
for theta in thetas:
	pwr_theta += t_data_power[i]
	i += 1	
pwr_theta /= (float)0.5 * N
for alpha in alphas:
	pwr_alpha += t_data_power[i]
	i += 1	
pwr_alpha /= (float)0.5 * N
for beta in betas:
	pwr_beta += t_data_power[i]
	i += 1	
pwr_beta /= (float)0.5 * N
for gamma in gammas:
	pwr_gamma += t_data_power[i]
	i += 1	
pwr_gamma /= (float)0.5 * N

#plt.plot(n1, t_data, 'g')
#plt.plot(n1, h, 'k')
#plt.plot(n1, s_data, 'k')
#plt.plot(n1, t_data_filtered, 'g')
#plt.plot(nfreq, t_data_power, 'b')
plt.show()
