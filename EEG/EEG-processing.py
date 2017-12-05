import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftshift, fftfreq
from scipy.signal import hamming, hann, kaiser

def to_seconds(positions, fs):
	return np.multiply(positions, 1.0/fs)
	
def to_microvolts(values):
    return np.multiply(values, 0.51)
	
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

#Highpass filter
h = np.concatenate([np.zeros(30), hann(N-60), np.zeros(30)]) 

pwr_alpha = np.zeros(groups-1)
pwr_beta = np.zeros(groups-1)
pwr_gamma = np.zeros(groups-1)
pwr_delta = np.zeros(groups-1)
pwr_theta = np.zeros(groups-1)

for g in np.arange(groups-1):
	#Get data from a time-window
    slice_start = 640 + (g*N)
    slice_end = 640 + ((g+1)*N)
    ##Try for different sensors (look up for dictionary with available names)
    s_data = to_microvolts(data[slice_start:slice_end,idx_dict['IED_AF4']])
    
    #Transform data
    t_data = fft(s_data, N)
    t_data /= float(N)

    #Apply highpass filter to data
    t_data_filtered = np.multiply(t_data, h)

    #Power terms
    t_data_conjugates = np.conjugate(t_data_filtered)
    t_data_power = np.multiply(t_data_filtered, t_data_conjugates)[:N/2]

    alphas = np.setdiff1d(nfreq[nfreq<12], nfreq[nfreq<8])
    betas = np.setdiff1d(nfreq[nfreq<40], nfreq[nfreq<12])
    ##Actually, since EPOC+ is not working at 256 Hz, frequencies will only be 64 Hz top
    gammas = np.setdiff1d(nfreq[nfreq<=100], nfreq[nfreq<40])
    deltas = nfreq[nfreq<4]
    thetas = np.setdiff1d(nfreq[nfreq<8], nfreq[nfreq<4])

    """
    pwr_alpha[g] = 0
    pwr_beta[g] = 0
    pwr_gamma[g] = 0
    pwr_delta[g] = 0
    pwr_theta[g] = 0
    """
    i = 0
    
    for delta in deltas:
        pwr_delta[g] += t_data_power[i]
        i += 1
    pwr_delta[g] /= (float)(0.5 * N)
    for theta in thetas:
        pwr_theta[g] += t_data_power[i]
        i += 1	
    pwr_theta[g] /= (float)(0.5 * N)
    for alpha in alphas:
        pwr_alpha[g] += t_data_power[i]
        i += 1	
    pwr_alpha[g] /= (float)(0.5 * N)
    for beta in betas:
        pwr_beta[g] += t_data_power[i]
        i += 1	
    pwr_beta[g] /= (float)(0.5 * N)
    for gamma in gammas:
        pwr_gamma[g] += t_data_power[i]
        i += 1	
    pwr_gamma[g] /= (float)(0.5 * N)

#Plotting

##Change step in order to separate vertical time-marking lines: 1-value step means 10 seconds
step = 6
xcoords = npwr[::step*N/(fs*10)]

#Separate plots for each frequency range
plt.figure(1)
plt.subplot(5, 1, 1)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.ylabel('Gamma')
plt.plot(npwr, pwr_gamma, 'b')
for xc in xcoords:
    plt.axvline(x=xc, color='k', linestyle='--')
plt.subplot(5, 1, 2)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.ylabel('Beta')
plt.plot(npwr, pwr_beta, 'r')
for xc in xcoords:
    plt.axvline(x=xc, color='k', linestyle='--')
plt.subplot(5, 1, 3)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.ylabel('Alpha')
plt.plot(npwr, pwr_alpha, 'm')
for xc in xcoords:
    plt.axvline(x=xc, color='k', linestyle='--')
plt.subplot(5, 1, 4)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.ylabel('Theta')
plt.plot(npwr, pwr_theta, 'g')
for xc in xcoords:
    plt.axvline(x=xc, color='k', linestyle='--')
plt.subplot(5, 1, 5)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.ylabel('Delta')
plt.plot(npwr, pwr_delta, 'c')
plt.xlabel('Time (seconds)')
for xc in xcoords:
    plt.axvline(x=xc, color='k', linestyle='--')
plt.tight_layout()

#One plot for all frequency ranges
plt.figure(2)
plt.plot(npwr, pwr_gamma, 'b')
plt.plot(npwr, pwr_beta, 'r')
plt.plot(npwr, pwr_alpha, 'm')
plt.plot(npwr, pwr_theta, 'g')
plt.plot(npwr, pwr_delta, 'c')
for xc in xcoords:
    plt.axvline(x=xc, color='k', linestyle='--')
plt.xlabel('Time (seconds)')
plt.ylabel('Power (uV^2)')

plt.show()
