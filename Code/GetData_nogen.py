import pickle
import random
import os
import numpy as np
from Preprocess import my_scaler
from audio_format import pcm2float

def get_data(data_dir, window, seed=422):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # -----------------------------------------------------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------------------------------------------------
    data = open(os.path.normpath('/'.join([data_dir, 'NotesDatasetSuperShort.pickle'])), 'rb')

    Z = pickle.load(data)
    signals = np.array(Z['signal'])
    notes = np.array(Z['note'])
    vels = np.array(Z['velocity'])

    data = open(os.path.normpath('/'.join([data_dir, 'NotesDatasetSuperShort_saw.pickle'])), 'rb')

    Z = pickle.load(data)
    saw = np.array(Z['signal'])

    del Z
    # -----------------------------------------------------------------------------------------------------------------
    # Scale data to be within (0, 1)
    # -----------------------------------------------------------------------------------------------------------------
    #Z = np.array([signals, sine])

    scaler =  my_scaler(feature_range=(-1, 1))
    scaler.fit(np.array([signals, saw]))
    signals = scaler.transform(signals)
    saw = scaler.transform(saw)

    scaler_note = my_scaler()
    scaler_note.fit(notes)
    notes = scaler_note.transform(notes)

    scaler_vel = my_scaler()
    scaler_vel.fit(vels)
    vels = scaler_vel.transform(vels)

    scaler = [scaler, scaler_note, scaler_vel]

    # -----------------------------------------------------------------------------------------------------------------
    # Shuffle indexing matrix and and split into test, train validation
    # -----------------------------------------------------------------------------------------------------------------
    #signals, notes, vels = get_batches(signals, notes, vels, 1)

    N = len(notes)
    n_train = N // 100 * 70
    n_val = (N - n_train) // 2

    all_inp, all_tar = [], []

    for i in range(n_train):
        for t in range(signals.shape[1] - window):
            inp_temp = np.array([saw[i, t :t + window], np.repeat(notes[i], window), np.repeat(vels[i], window)])
            all_inp.append(inp_temp.T)
            tar_temp = np.array(signals[i, t :t + window])
            all_tar.append(tar_temp.T)

    all_inp = np.array(all_inp)
    all_tar = np.array(all_tar)

    x = all_inp
    y = all_tar

    all_inp = []
    all_tar = []

    for i in range(n_train, n_train + n_val):
        for t in range(signals.shape[1] - window):
            inp_temp = np.array(
                [saw[i, t :t + window], np.repeat(notes[i], window), np.repeat(vels[i], window)])
            all_inp.append(inp_temp.T)
            tar_temp = np.array(signals[i, t :t + window])
            all_tar.append(tar_temp.T)

    all_inp = np.array(all_inp)
    all_tar = np.array(all_tar)

    x_val = all_inp
    y_val = all_tar

    all_inp = []
    all_tar = []

    for i in range(n_train + n_val, N):
        for t in range(signals.shape[1] - window):
            inp_temp = np.array(
                [saw[i, t :t + window], np.repeat(notes[i], window), np.repeat(vels[i], window)])
            all_inp.append(inp_temp.T)
            tar_temp = np.array(signals[i, t :t + window])
            all_tar.append(tar_temp.T)

    all_inp = np.array(all_inp)
    all_tar = np.array(all_tar)

    x_test = all_inp
    y_test = all_tar

    return x, y, x_val, y_val, x_test, y_test, scaler

if __name__ == '__main__':

    data_dir = '../Files'

    x, y, x_val, y_val, x_test, y_test, scaler = get_data(data_dir=data_dir, window=16)

    data = {'x': x, 'y': y, 'x_val': x_val, 'y_val': y_val, 'x_test': x_test, 'y_test': y_test , 'scaler': scaler}

    file_data = open(os.path.normpath('/'.join([data_dir, 'NotesDatasetSuperShortPrepared_16_saw.pickle'])), 'wb')
    pickle.dump(data, file_data)
    file_data.close()
