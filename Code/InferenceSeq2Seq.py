import numpy as np
import os
import time
import tensorflow as tf
from GetData import get_data
from scipy.io import wavfile
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
import pickle



#from tensorflow.compat.v1 import ConfigProto
#from tensorflow.compat.v1 import InteractiveSession
#config = ConfigProto()
#config.gpu_options.allow_growth = True
#session = InteractiveSession(config=config)


# generate target given source sequence
def predict_sequence(encoder_model, decoder_model, input_seq, n_steps, output_dim, last_pred, window):
    # encode
    input_seq = input_seq.reshape(1, 2, window)
    state = encoder_model.predict(input_seq)
    # start of sequence input
    target_seq = np.zeros((1, output_dim, 1))  # .reshape(1, 1, output_dim)
    last_prediction = last_pred
    target_seq[0, 0, 0] = last_prediction
    # collect predictions
    output = []
    for t in range(10):#n_steps):
        # predict next char
        yhat, h, c = decoder_model.predict([target_seq] + state)
        # store prediction
        output.append(yhat[0, 0, :])
        # update state
        state = [h, c]
        # update target sequence
        target_seq = yhat
        last_prediction = yhat[0, 0, :]

    output = np.array(output)
    return output, last_prediction


def inferenceLSTM(data_dir, seed=422, **kwargs):
    ckpt_flag = kwargs.get('ckpt_flag', False)
    b_size = kwargs.get('b_size', 16)
    learning_rate = kwargs.get('learning_rate', 0.001)
    encoder_units = kwargs.get('encoder_units', [64, 64])
    decoder_units = kwargs.get('decoder_units', [64, 64])
    if encoder_units[-1] != decoder_units[0]:
        raise ValueError('Final encoder layer must same units as first decoder layer!')
    model_save_dir = kwargs.get('model_save_dir', '../../LSTM_TrainedModels')
    save_folder = kwargs.get('save_folder', 'LSTM_enc_dec_Testing')
    drop = kwargs.get('drop', 0.)
    inference = kwargs.get('inference', False)

    sigs, notes, vels, scaler = get_data(data_dir=data_dir, seed=seed)

    sigs = sigs.reshape(sigs.shape[0], sigs.shape[2])
    notes = notes.reshape(notes.shape[0])
    vels = vels.reshape(vels.shape[0])
    # T past values used to predict the next value
    T = sigs.shape[1]  # time window
    D = 2  # conditioning

    unit_encoder = 64
    unit_decoder = 64
    num_decoder_tokens = 1
    #TRAINING
    
    #encoder
    encoder_inputs = Input(shape=(D,1), name='enc_input')

    outputs, state_h, state_c = LSTM(unit_encoder, return_state=True, name='LSTM_En')(encoder_inputs)

    encoder_states = [state_h, state_c]   
    
    #decoder
    decoder_inputs = Input(shape=(T-1,1), name='dec_input')

    decoder_lstm = LSTM(unit_decoder, return_sequences=True, return_state=True, name='LSTM_De', dropout=drop)
    outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    
    if drop != 0.:
        outputs = tf.keras.layers.Dropout(drop, name='DropLayer')(outputs)
    decoder_dense = Dense(num_decoder_tokens, activation='sigmoid', name='DenseLay')
    decoder_outputs = decoder_dense(outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.summary()
    
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='mse', metrics=['mse'], optimizer=opt)
    
    callbacks = []
    if ckpt_flag:
        ckpt_path = os.path.normpath(os.path.join(model_save_dir, save_folder, 'Checkpoints', 'best', 'best.ckpt'))
        ckpt_path_latest = os.path.normpath(os.path.join(model_save_dir, save_folder, 'Checkpoints', 'latest', 'latest.ckpt'))
        ckpt_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'Checkpoints', 'best'))
        ckpt_dir_latest = os.path.normpath(os.path.join(model_save_dir, save_folder, 'Checkpoints', 'latest'))
        
        if not os.path.exists(os.path.dirname(ckpt_dir)):
            os.makedirs(os.path.dirname(ckpt_dir))
        if not os.path.exists(os.path.dirname(ckpt_dir_latest)):
            os.makedirs(os.path.dirname(ckpt_dir_latest))

        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path, monitor='val_loss', mode='min',
                                                           save_best_only=True, save_weights_only=True, verbose=1)
        ckpt_callback_latest = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path_latest, monitor='val_loss', mode='min',
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


    N = len(notes)
    cond = np.array([notes, vels]).reshape(168,2)

    if ckpt_flag:
        best = tf.train.latest_checkpoint(ckpt_dir)
        if best is not None:
            print("Restored weights from {}".format(ckpt_dir))
            model.load_weights(best)
    #test_loss = model.evaluate([cond[(2*N)//3+1:], sigs[(2*N)//3+1:, :-1]], sigs[(2*N)//3+1:, 1:], batch_size=b_size, verbose=0)
    #print('Test Loss: ', test_loss)

    
    #INFERENCE
    # define inference encoder
    encoder_model = Model(encoder_inputs, encoder_states)
    #define inference decoder
    decoder_state_input_h = Input(shape=(unit_decoder,))
    decoder_state_input_c = Input(shape=(unit_decoder,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    x_gen = cond[(2*N)//3+2]
    y_gen = sigs[(2*N)//3+2]

    if inference:
        #start = time.time()
        last_prediction = y_gen[0]
        predictions = [last_prediction]
        output_dim = y_gen.shape[0] - 1

        out, last_prediction = predict_sequence(encoder_model, decoder_model, x_gen, y_gen.shape[0],
                                                    output_dim, last_prediction, 1)
        #predictions.append(out)
        #end = time.time()
        #print(end - start)
        #out = np.array(out)
        predictions = np.concatenate((np.array(predictions).reshape(-1), out.reshape(-1)), axis=0)
        predictions = np.array(predictions)

        predictions = scaler[0].inverse_transform(predictions)
        y_gen = scaler[0].inverse_transform(y_gen)

        predictions = predictions.reshape(-1)
        y_gen = y_gen.reshape(-1)

        # Define directories
        pred_name = 'LSTM_pred.wav'
        tar_name = 'LSTM_tar.wav'

        pred_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', pred_name))
        tar_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', tar_name))

        if not os.path.exists(os.path.dirname(pred_dir)):
            os.makedirs(os.path.dirname(pred_dir))

        # Save Wav files
        predictions = predictions.astype('int16')
        y_gen = y_gen.astype('int16')
        wavfile.write(pred_dir, 44100, predictions)
        wavfile.write(tar_dir, 44100, y_gen)



if __name__ == '__main__':
    data_dir = '../Files'
    seed = 422
    #start = time.time()
    inferenceLSTM(data_dir=data_dir,
              model_save_dir='../../TrainedModels',
              save_folder='Seq2Seq_Testing',
              ckpt_flag=True,
              b_size=128,
              learning_rate=0.0001,
              encoder_units=[64],
              decoder_units=[64],
              inference=True)
    #end = time.time()
    #print(end - start)