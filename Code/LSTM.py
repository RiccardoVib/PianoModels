import tensorboard
#load_ext tensorboard
#rm -rf ./logs/
import datetime
import numpy as np
import os
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.io import wavfile
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
import pickle
import Preprocess

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def trainLSTM(data_dir, epochs, seed=422, **kwargs):
    ckpt_flag = kwargs.get('ckpt_flag', False)
    b_size = kwargs.get('b_size', 32)
    learning_rate = kwargs.get('learning_rate', 0.001)
    units = kwargs.get('units', [1])
    model_save_dir = kwargs.get('model_save_dir', '../../LSTM_TrainedModels')
    save_folder = kwargs.get('save_folder', 'LSTM_testing')
    generate_wav = kwargs.get('generate_wav', None)
    drop = kwargs.get('drop', 0.)
    opt_type = kwargs.get('opt_type', 'Adam')
    inference = kwargs.get('inference', False)
    loss_type = kwargs.get('loss_type', 'mse')
    w_length = kwargs.get('w_length', 16)
    act = kwargs.get('act', 'tanh')

    file_data = open(os.path.normpath('/'.join([data_dir, 'NotesDatasetPrepared_1.pickle'])), 'rb')
    data = pickle.load(file_data)

    x = data['x']
    y = data['y']
    x_val = data['x_val']
    y_val = data['y_val']
    x_test = data['x_test']
    y_test = data['y_test']
    scaler = data['scaler']
      
    layers = len(units)
    n_units = ''
    for unit in units:
        n_units += str(unit)+', '

    n_units = n_units[:-2]
    
    #T past values used to predict the next value
    T = x.shape[1] #time window
    D = x.shape[2] #conditioning

    inputs = Input(shape=(T,D), name='enc_input')
    first_unit_encoder = units.pop(0)
    if len(units) > 0:
        last_unit_encoder = units.pop()
        outputs = LSTM(first_unit_encoder, return_sequences=True, name='LSTM_En0')(inputs)
        for i, unit in enumerate(units):
            outputs,  state_h, state_c = LSTM(unit, return_sequences=True, name='LSTM_En' + str(i + 1))(outputs)
        outputs = LSTM(last_unit_encoder, name='LSTM_EnFin')(outputs)
    else:
        outputs = LSTM(first_unit_encoder, name='LSTM_En')(inputs)

    if drop != 0.:
        outputs = tf.keras.layers.Dropout(drop, name='DropLayer')(outputs)
    outputs = Dense(1, activation=act, name='DenseLay')(outputs)
    model = Model(inputs, outputs)
    model.summary()

    if opt_type == 'Adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif opt_type == 'SGD':
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        raise ValueError('Please pass opt_type as either Adam or SGD')

    if loss_type == 'mae':
        model.compile(loss='mae', metrics=['mae'], optimizer=opt)
    elif loss_type == 'mse':
        model.compile(loss='mse', metrics=['mse'], optimizer=opt)
    else:
        raise ValueError('Please pass loss_type as either MAE or MSE')

    callbacks = []
    if ckpt_flag:
        ckpt_path = os.path.normpath(os.path.join(model_save_dir, save_folder, 'Checkpoints', 'best', 'best.ckpt'))
        ckpt_path_latest = os.path.normpath(
            os.path.join(model_save_dir, save_folder, 'Checkpoints', 'latest', 'latest.ckpt'))
        ckpt_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'Checkpoints', 'best'))
        ckpt_dir_latest = os.path.normpath(os.path.join(model_save_dir, save_folder, 'Checkpoints', 'latest'))

        if not os.path.exists(os.path.dirname(ckpt_dir)):
            os.makedirs(os.path.dirname(ckpt_dir))
        if not os.path.exists(os.path.dirname(ckpt_dir_latest)):
            os.makedirs(os.path.dirname(ckpt_dir_latest))

        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path, monitor='val_loss', mode='min',
                                                           save_best_only=True, save_weights_only=True, verbose=1)
        ckpt_callback_latest = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path_latest, monitor='val_loss',
                                                                  mode='min',
                                                                  save_best_only=False, save_weights_only=True,
                                                                  verbose=1)
        callbacks += [ckpt_callback, ckpt_callback_latest]
        latest = tf.train.latest_checkpoint(ckpt_dir_latest)
        if latest is not None:
            print("Restored weights from {}".format(ckpt_dir))
            model.load_weights(latest)
            # start_epoch = int(latest.split('-')[-1].split('.')[0])
            # print('Starting from epoch: ', start_epoch + 1)
        else:
            print("Initializing random weights.")

    #log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.00000001, patience=20, restore_best_weights=True, verbose=0)
    callbacks += [early_stopping_callback]

    #train the RNN
    if not inference:
        results = model.fit(x, y, batch_size=b_size, epochs=epochs, verbose=0,
                        validation_data=(x_val, y_val),
                        callbacks=callbacks)


    predictions_test = model.predict(x_test, batch_size=b_size)

    if ckpt_flag:
        best = tf.train.latest_checkpoint(ckpt_dir)
        if best is not None:
            print("Restored weights from {}".format(ckpt_dir))
            model.load_weights(best)

    test_loss = model.evaluate(x_test, y_test, batch_size=b_size, verbose=0)
    print('Test Loss: ', test_loss)
    if inference:
        results = {}
    else:
        results = {
            'Test_Loss': test_loss,
            'Min_val_loss': np.min(results.history['val_loss']),
            'Min_train_loss': np.min(results.history['loss']),
            'b_size': b_size,
            'learning_rate': learning_rate,
            'drop': drop,
            'opt_type': opt_type,
            'loss_type': loss_type,
            'layers': layers,
            'units': n_units,
            'w_length': w_length,
            #'Train_loss': results.history['loss'],
            'Val_loss': results.history['val_loss']
        }
        print(results)
    if not inference:
        if ckpt_flag:
            with open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results.txt'])), 'w') as f:
                for key, value in results.items():
                    print('\n', key, '  : ', value, file=f)
                pickle.dump(results, open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results.pkl'])), 'wb'))

 

    if generate_wav is not None:
        np.random.seed(seed)
        x_gen = x_test
        y_gen = y_test
        predictions = model.predict(x_gen)
        print('GenerateWavLoss: ', model.evaluate(x_gen, y_gen, batch_size=b_size, verbose=0))

        predictions = predictions.reshape(-1)
        x_gen = x_gen[:,:,0].reshape(-1)
        y_gen = y_gen.reshape(-1)

        # Define directories
        pred_name = 'LSTM_pred.wav'
        inp_name = 'LSTM_inp.wav'
        tar_name = 'LSTM_tar.wav'

        pred_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', pred_name))
        inp_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', inp_name))
        tar_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', tar_name))

        if not os.path.exists(os.path.dirname(pred_dir)):
            os.makedirs(os.path.dirname(pred_dir))

        # Save Wav files
        predictions = predictions.astype('int16')
        x_gen = x_gen.astype('int16')
        y_gen = y_gen.astype('int16')
        wavfile.write(pred_dir, 44100, predictions)
        wavfile.write(inp_dir, 44100, x_gen)
        wavfile.write(tar_dir, 44100, y_gen)

    return results

if __name__ == '__main__':
    data_dir = '../Files'
  
    trainLSTM(data_dir=data_dir,
              model_save_dir='../../TrainedModels',
              save_folder='LSTM_piano',
              ckpt_flag=True,
              b_size=128,
              units=[64, 64],
              learning_rate=0.001,
              epochs=100,
              loss_type='mse',
              generate_wav=2,
              w_length=1,
              act='tanh',
              inference=False)