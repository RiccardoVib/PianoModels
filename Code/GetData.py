import pickle
import random
import os
import numpy as np
from Preprocess import my_scaler, get_batches


def get_data(data_dir, seed=422):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # -----------------------------------------------------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------------------------------------------------
    data = open(os.path.normpath('/'.join([data_dir, 'NotesDatasetShort.pickle'])), 'rb')

    Z = pickle.load(data)
    signals = np.array(Z['signal'])
    notes = np.array(Z['note'])
    vels = np.array(Z['velocity'])

    # -----------------------------------------------------------------------------------------------------------------
    # Scale data to be within (0, 1)
    # -----------------------------------------------------------------------------------------------------------------

    scaler_sig = my_scaler()
    scaler_sig.fit(signals)
    signals = scaler_sig.transform(signals)

    scaler_note = my_scaler()
    scaler_note.fit(notes)
    notes = scaler_note.transform(notes)

    scaler_vel = my_scaler()
    scaler_vel.fit(vels)
    vels = scaler_vel.transform(vels)

    scaler = [scaler_sig, scaler_note, scaler_vel]
    #signals = scaler.inverse_transform(signals)

    #zero_value = (0 - scaler.min_data) / (scaler.max_data - scaler.min_data)

    # -----------------------------------------------------------------------------------------------------------------
    # Shuffle indexing matrix and and split into test, train validation
    # -----------------------------------------------------------------------------------------------------------------
    signals, notes, vels = get_batches(signals, notes, vels, 1)

    N = len(notes)

    # sigs_train, notes_train, vels_train, sigs_val, notes_val, vels_val, sigs_test, notes_test, vels_test = [], [], [], [], [], [], [], [], []
    # for i in range(N):
    #     sigs_train = signals[:N//2]
    #     notes_train = notes[:N//2]
    #     vels_train
    #signals = np.array(signals)[:, :10]
    return np.array(signals), np.array(notes), np.array(vels), scaler

if __name__ == '__main__':

    data_dir = '../Files'

    signals, notes, vels, scaler = get_data(data_dir=data_dir, seed=422)

    data = {'signals': signals, 'notes': notes, 'vels': vels, 'scaler': scaler}
