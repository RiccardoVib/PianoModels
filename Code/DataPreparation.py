import numpy as np
from scipy.io import wavfile
from scipy import signal
import os
import glob
import pickle
import audio_format

def get_keys_from_value(d, val):
    return [k for k, v in d.items() if v == val]

def data_preparation(**kwargs):
    data_dir = kwargs.get('data_dir', 'C:/Users/riccarsi/Documents/PianoAnalysisProcessed/')
    save_dir = kwargs.get('save_dir', '../Files')
    file_dirs = glob.glob(os.path.normpath('/'.join([data_dir, 'SingleDisk.wav'])))
    L = 48000

    Notes_collector, Waveform_collector = [], []
    fs = 0

    for file in file_dirs:

        #filename = os.path.split(file)[-1]
        #metadata = filename.split('_', 2)
        #ratio = metadata[1]
        #threshold = metadata[-1].replace('.wav', '')
        fs, audio_stereo = wavfile.read(file) #fs= 44,100 Hz
        signal = audio_stereo[L:2*L]
        #signal = audio_format.pcm2float(signal)
        note = 'A3'
        Notes_collector.append(note)
        Waveform_collector.append(signal)
    Notes = {'waveforms': Waveform_collector, 'notes': Notes_collector}

    file_data = open(os.path.normpath('/'.join([save_dir, 'NotesDataset.pickle'])), 'wb')
    pickle.dump(Notes, file_data)
    file_data.close()
if __name__ == '__main__':

    data_preparation()