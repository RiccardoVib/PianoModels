import pickle
import random
import os
import tensorflow as tf
import numpy as np
from Preprocess import my_scaler


def get_data(data_dir, seed=422):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # -----------------------------------------------------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------------------------------------------------
    data = open(os.path.normpath('/'.join([data_dir, 'NotesDataset.pickle'])), 'rb')

    Z = pickle.load(data)
    wave = Z['waveforms']
    note = Z['notes']

    wave = np.array(wave)

    # -----------------------------------------------------------------------------------------------------------------
    # Scale data to be within (0, 1)
    # -----------------------------------------------------------------------------------------------------------------

    scaler = my_scaler()
    scaler.fit(wave)

    wave = scaler.transform(wave)

    zero_value = (0 - scaler.min_data) / (scaler.max_data - scaler.min_data)

    # -----------------------------------------------------------------------------------------------------------------
    # Shuffle indexing matrix and and split into test, train validation
    # -----------------------------------------------------------------------------------------------------------------
    #48-71
    x = 0
    y = wave

    return x, y, scaler

if __name__ == '__main__':

    data_dir = '../Files'
    w1 = 0.000021
    w2 = 0.000045
    w4 = 0.00009
    w8 = 0.00017
    w16 = 0.00034
    x, y, scaler = get_data(data_dir=data_dir, seed=422)

    data = {'x': x, 'y': y, 'scaler': scaler}
