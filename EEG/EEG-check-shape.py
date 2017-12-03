import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Get data for each header
data = pd.read_csv('Records/EEG_casicreativo_dineroyentrevista_ivan.csv', sep=',', header=1)
data = data.as_matrix()
data = np.delete(data, np.s_[-1:], axis=1)

print data.shape
#print data.shape[0] / (float)(60 * 128) #minutes at 128Hz
#print data.shape[0] / 128 #seconds at 128Hz
print data.shape[0] / (60 * 256) #minutes at 256Hz
print data.shape[0] / 256 #seconds at 256Hz
