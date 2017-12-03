import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

with open('Records/EEGLogger.csv', 'r') as f:  #opens PW file
    reader = csv.reader(f)
    
""" Simple Python code
    #First line = field names
    #Next lines = one-sample values for previous fields
    for row in reader:
        print row
"""

# Code with numpy
headers = np.genfromtxt('Records/EEGLogger.csv', dtype=None, delimiter=',')[0]
data = np.genfromtxt('Records/EEGLogger.csv', dtype=None, delimiter=',', names=True) 
field_limit = len(headers) - 1
headers = np.delete(headers, field_limit)
print headers
print data.shape
#data = np.delete(a, np.s_[-1:], axis=1)
#for i in np.arange(data.shape[0]):
    #data[i] = np.delete(data[i], field_limit)
    #print data[i]



