import numpy as np
import os
import pickle
import scipy.fftpack as sf
import matplotlib.pyplot as plt


def data_create_sins(**kwargs):
    data_dir = kwargs.get('data_dir', '../Files')
    save_dir = kwargs.get('save_dir', '../Files')

    data = open(os.path.normpath('/'.join([data_dir, 'NotesDatasetShort.pickle'])), 'rb')

    Z = pickle.load(data)

    signals = np.array(Z['signal'])
    notes = np.array(Z['note'])
    vels = np.array(Z['velocity'])
    fs = 44100
    Notes_collector_sine = {'signal': [], 'note': [], 'velocity': []}
    Low_cutoff = 20
    High_cutoff = 20000

    for i in range(notes.shape[0]):

        M = signals.shape[1] # let M be the length of the time series
        Spectrum = sf.rfft(signals[i], n=M)
        Spectrum = Spectrum[:len(Spectrum)//2]
        #Spectrum = fft.fft(signals[i], 22050)
        [Low_cutoff, High_cutoff, fs] = map(float, [Low_cutoff, High_cutoff, fs])
        # Convert cutoff frequencies into points on spectrum
        [Low_point, High_point] = map(lambda F: F / fs * M, [Low_cutoff, High_cutoff])
        maximumFrequency = np.where(Spectrum == np.max(Spectrum[int(Low_point) : int(High_point)]))

        index = np.where(signals[i] == 0)
        index = index[-1][-1]
        samples = np.linspace(0, index/fs, index, endpoint=False)

        vel = vels[i]/120
        vel = vel/2
        sine = vel * np.sin(np.pi * maximumFrequency[0] * samples)
        #sine *= 32767
        #sine *= 1000
        sine = np.pad(sine, (0, signals[i].shape[0] - index))

        M = len(sine)  # let M be the length of the time series
        # Spectrum = sf.fft(signals[i], n=M * 4)
        # Spectrum = Spectrum[:len(Spectrum) // 2]
        # Spectrum_s = sf.fft(sine, n=M * 4)
        # Spectrum_s = Spectrum_s[:len(Spectrum_s) // 2]
        # plt.plot(np.abs(Spectrum_s))
        # plt.plot(np.abs(Spectrum))
        # plt.yscale('log')
        # plt.xscale('log')
        #plt.show()
        #
        # plt.plot(sine)
        # plt.show()
        #sine = np.int16(sine)
        Notes_collector_sine['signal'].append(sine)
        Notes_collector_sine['note'].append(notes[i])
        Notes_collector_sine['velocity'].append(vels[i])

    file_data = open(os.path.normpath('/'.join([save_dir, 'NotesDatasetShort_sines.pickle'])), 'wb')
    pickle.dump(Notes_collector_sine, file_data)
    file_data.close()


if __name__ == '__main__':

    data_create_sins()

