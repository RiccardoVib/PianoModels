import numpy as np
import os
import pickle
import scipy.fftpack as sf
from scipy.signal import sawtooth, windows, convolve
import matplotlib.pyplot as plt
from audio_format import float2pcm, pcm2float

def data_create_saws(**kwargs):
    data_dir = kwargs.get('data_dir', '../Files')

    data = open(os.path.normpath('/'.join([data_dir, 'NotesDatasetSuperShort.pickle'])), 'rb')

    Z = pickle.load(data)

    signals = np.array(Z['signal'])
    notes = np.array(Z['note'])
    vels = np.array(Z['velocity'])
    fs = 44100
    Notes_collector_saw = {'signal': [], 'note': [], 'velocity': []}
    Low_cutoff = 0
    High_cutoff = 20000

    M = signals.shape[1]

    window = windows.tukey(M, alpha=0.2, sym=True)

    for i in range(notes.shape[0]):

         # let M be the length of the time series
        Spectrum = sf.rfft(signals[i], n=22050)
        Spectrum = Spectrum[:len(Spectrum)//2]
        #Spectrum = fft.fft(signals[i], 22050)
        [Low_cutoff, High_cutoff, fs] = map(float, [Low_cutoff, High_cutoff, fs])
        # Convert cutoff frequencies into points on spectrum
        [Low_point, High_point] = map(lambda F: F / fs * M, [Low_cutoff, High_cutoff])
        maximumFrequency = np.where(Spectrum == np.max(Spectrum[int(Low_point) : int(High_point)]))

        index = np.where(signals[i] == 0)
        index = index[-1][-1]
        t = np.linspace(0, M/fs, index, endpoint=False)

        vel = 1#vels[i]/(10*127)

        saw = vel*sawtooth(2*np.pi * maximumFrequency[0] * t)

        saw = np.pad(saw, (0, signals[i].shape[0] - index))

        saw = saw*window
        #saw = convolve(saw, window, mode='same')/sum(window)

        # Spectrum = sf.fft(signals[i])
        # Spectrum = Spectrum[:len(Spectrum) // 2]
        # Spectrum_s = sf.fft(saw)
        # Spectrum_s = Spectrum_s[:len(Spectrum_s) // 2]
        # plt.plot(np.abs(Spectrum_s))
        # plt.plot(np.abs(Spectrum))
        # plt.yscale('log')
        # plt.xscale('log')
        # plt.show()
        #
        #saw = float2pcm(saw)
        signal = pcm2float(signals[i])

        # plt.plot(saw)
        # #plt.plot(window)
        # plt.plot(signal)
        # plt.show()

        Notes_collector_saw['signal'].append(saw)
        Notes_collector_saw['note'].append(notes[i])
        Notes_collector_saw['velocity'].append(vels[i])

    file_data = open(os.path.normpath('/'.join([data_dir, 'NotesDatasetShort_saw.pickle'])), 'wb')
    pickle.dump(Notes_collector_saw, file_data)
    file_data.close()


if __name__ == '__main__':

    data_create_saws()

