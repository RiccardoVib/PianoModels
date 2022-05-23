import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import os
import glob
import pickle

def get_keys_from_value(d, val):
    return [k for k, v in d.items() if v == val]

def data_preparation(**kwargs):
    #data_dir = kwargs.get('data_dir', 'C:/Users/riccarsi/Documents/PianoAnalysisProcessed/Piano')
    data_dir = kwargs.get('data_dir', '../Files')

    save_dir = kwargs.get('save_dir', '../Files')
    file_dirs = glob.glob(os.path.normpath('/'.join([data_dir, 'PianoDatasetsSingleNoteDuckShort.wav'])))
    #L = 48000

    Notes_collector = {'signal': [], 'note': [], 'velocity': []}
    #fs = 44100

    for file in file_dirs:

        fs, audio = wavfile.read(file) #fs= 44,100 Hz
        end = True
        velocity_index = 0
        note = 1
        velocity = [60, 70, 80, 90, 100, 110, 120]#60-120
        while end:
            index = np.where(audio != 0)[0][0]
            note_signal = audio[index:int(fs*2.5)]
            audio = audio[index+fs*2:]
            vel = velocity[velocity_index % len(velocity)]
            limiter = 47000#75000
            t = np.linspace(0, len(note_signal)/fs, num=len(note_signal))
            plt.plot(t[:limiter], note_signal[:limiter])
            #plt.show()
            nameFig = save_dir + '/Figs/' + str(note) + '_' + str(vel) + '.png'
            plt.savefig(nameFig)
            plt.close()

            Notes_collector['signal'].append(note_signal[:limiter])
            Notes_collector['note'].append(note)
            Notes_collector['velocity'].append(vel)
            if vel == 120:
                note = note + 1
            velocity_index = velocity_index + 1

            if len(audio) < int(fs*2.5):
                end = False
        #signal = audio_format.pcm2float(signal)


    file_data = open(os.path.normpath('/'.join([save_dir, 'NotesDatasetShort.pickle'])), 'wb')
    pickle.dump(Notes_collector, file_data)
    file_data.close()

if __name__ == '__main__':

    data_preparation()



