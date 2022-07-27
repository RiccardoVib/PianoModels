import pickle
import os
import numpy as np
import scipy.fftpack as sf
import matplotlib.pyplot as plt
import glob
from scipy.io import wavfile
from audio_format import pcm2float

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


# data = open(os.path.normpath('/'.join([data_dir, 'NotesDatasetPrepared_1.pickle'])), 'rb')
# Z = pickle.load(data)
# y = np.array(Z['y'])
# x = np.array(Z['x'])
# x = x[:,:,0]
# t = np.linspace(0, len(x.reshape(-1)) / fs, num=len(x.reshape(-1)))
# plt.plot(x.reshape(-1))
# plt.plot(y.reshape(-1))
# plt.show()


#---------
# data = open(os.path.normpath('/'.join([data_dir, 'NotesSuperShortDatasetPrepared_16.pickle'])), 'rb')
# Z = pickle.load(data)
# kont = np.array(Z['x_test'])[:, :, 0]
# real = np.array(Z['y_test'])
#
# plt.plot(kont[:,-1].reshape(-1))
# plt.show()

data_dir = '../Files'
# file_dirs = glob.glob(os.path.normpath('/'.join([data_dir, 'inference_pred.wav'])))
# for file in file_dirs:
#
#     fs, audio_pred = wavfile.read(file)
#
# file_dirs = glob.glob(os.path.normpath('/'.join([data_dir, 'Transf_tar.wav'])))
# for file in file_dirs:
#     fs, audio_tar = wavfile.read(file)
#
# t = np.linspace(0, len(audio_tar) / fs, num=len(audio_tar))
# plt.plot(t, pcm2float(audio_pred[:-1]), t, pcm2float(audio_tar))
# plt.show()

file_dirs = glob.glob(os.path.normpath('/'.join([data_dir, 'TransformerModelGen_tar1.wav'])))
for file in file_dirs:
    fs, audio_tar = wavfile.read(file)


file_dirs = glob.glob(os.path.normpath('/'.join([data_dir, 'TransformerModelGen_pred1.wav'])))
for file in file_dirs:

    fs, audio_pred = wavfile.read(file)


t = np.linspace(0, len(audio_tar) / fs, num=len(audio_tar))
plt.plot(t, pcm2float(audio_pred), t, pcm2float(audio_tar))
plt.show()