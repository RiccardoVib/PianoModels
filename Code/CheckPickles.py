import pickle
import os
import numpy as np
import scipy.fftpack as sf
import matplotlib.pyplot as plt

data_dir = '../Files'
data = open(os.path.normpath('/'.join([data_dir, 'NotesDatasetShort.pickle'])), 'rb')
fs = 44100

Z = pickle.load(data)
sig = np.array(Z['signal'])
notes = np.array(Z['note'])
vels = np.array(Z['velocity'])

data = open(os.path.normpath('/'.join([data_dir, 'NotesDatasetShort_Sines.pickle'])), 'rb')

Z = pickle.load(data)
sine = np.array(Z['signal'])
notes_ = np.array(Z['note'])
vels_ = np.array(Z['velocity'])
#
# for i in range(notes.shape[0]):
#
#     #t = np.linspace(0, len(sig[i]) / fs, num=len(sig[i]))
#     plt.plot(sine[i])
#     plt.show()
#
#     #M = sig.shape[1]  # let M be the length of the time series
#     #Spectrum = sf.fft(sig[i], n=M*4)
#     #Spectrum = Spectrum[:len(Spectrum)//2]
#     #Spectrum_s = sf.fft(sine[i], n=M*4)
#     #Spectrum_s = Spectrum_s[:len(Spectrum_s) // 2]
#     #plt.plot(np.abs(Spectrum_s))
#     #plt.yscale('log')
#     #plt.xscale('log')
#     plt.show()


data = open(os.path.normpath('/'.join([data_dir, 'NotesDatasetPrepared_1.pickle'])), 'rb')
Z = pickle.load(data)
y_val = np.array(Z['y_val'])
x_val = np.array(Z['x_val'])
x_val = x_val[:,:,0]
t = np.linspace(0, len(x_val.reshape(-1)) / fs, num=len(x_val.reshape(-1)))
plt.plot(x_val.reshape(-1))
plt.plot(y_val.reshape(-1))
plt.show()
