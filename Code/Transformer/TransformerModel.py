import tensorflow as tf
from keras import layers, Sequential, Model
#from tensorflow.python.keras.layers import MultiHeadAttention
from Preprocess import positional_encoding
from tensorflow.keras import layers
import numpy as np
import os
import pickle
from scipy.io import wavfile
from GetDataPiano import get_data
from TrainFunctionality import combinedLoss

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()

        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training, generative):
        #https://stackoverflow.com/questions/67805117/multiheadattention-attention-mask-keras-tensorflow-example
        #embedding = tf.keras.layers.Embedding(input_dim=inputs.shape[1], output_dim=inputs.shape[1], mask_zero=True)
        #mask = embedding.compute_mask(inputs)
        #mask = mask[:, tf.newaxis, tf.newaxis, :]

        masking_layer = layers.Masking(mask_value=0.0, input_shape=(inputs.shape[1], inputs.shape[1]))
        #masked_att = Attention(casual=True, dropout=0.5)
        attn_output = self.att(inputs, inputs)#, attention_mask=mask)
        if generative:
            attn_output = masking_layer(attn_output)
        #attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


def trainMultiAttention(data_dir, epochs, seed=422, **kwargs):

    ckpt_flag = kwargs.get('ckpt_flag', False)
    b_size = kwargs.get('b_size', 32)
    learning_rate = kwargs.get('learning_rate', 0.001)
    ff_dim = kwargs.get('ff_dim', 512)
    num_heads = kwargs.get('num_heads', 8)
    d_model = kwargs.get('d_model', 512)
    model_save_dir = kwargs.get('model_save_dir', '/scratch/users/riccarsi/TrainedModels')
    save_folder = kwargs.get('save_folder', 'ED_Testing')
    drop = kwargs.get('drop', 0.)
    opt_type = kwargs.get('opt_type', 'Adam')
    loss_type = kwargs.get('loss_type', 'mse')
    w_length = kwargs.get('w_length', 16)
    generate_wav = kwargs.get('generate_wav', None)
    generative = kwargs.get('generative', True)
    all = kwargs.get('all', True)

    if generative:

        sigs, notes, vels, scaler = get_data(data_dir=data_dir, seed=seed)

        sigs = sigs.reshape(sigs.shape[0], sigs.shape[2])
        notes = notes.reshape(notes.shape[0])
        vels = vels.reshape(vels.shape[0])
        all_inp = []

        for i in range(sigs.shape[0]):
            inp_temp = np.array([sigs[i, :], np.repeat(notes[i], sigs.shape[1]), np.repeat(vels[i], sigs.shape[1])])
            all_inp.append(inp_temp.T)

        all_inp = np.array(all_inp)
        N = len(notes)
        #cond = np.array([notes, vels]).reshape(168, 2)
        sigs = all_inp

        T = sigs.shape[1] - 1  # time window
        D = sigs.shape[2]  # conditioning
        out_dim = T

    elif not generative and not all:
        file_data = open(os.path.normpath('/'.join([data_dir, 'NotesSuperShortDatasetPrepared_16.pickle'])), 'rb')
        data = pickle.load(file_data)

        x = data['x']
        y = data['y']
        x_val = data['x_val']
        y_val = data['y_val']
        x_test = data['x_test']
        y_test = data['y_test']
        scaler = data['scaler']

        T = x.shape[1]
        D = x.shape[2]
        out_dim = 1
    elif not generative and all:

        file_data = open(os.path.normpath('/'.join([data_dir, 'NotesDatasetPrepared_allinp.pickle'])), 'rb')
        data = pickle.load(file_data)

        x = data['x']
        y = data['y']
        x_val = data['x_val']
        y_val = data['y_val']
        x_test = data['x_test']
        y_test = data['y_test']
        scaler = data['scaler']

        T = x.shape[1]
        D = x.shape[2]
        out_dim = T
    else:
        raise ValueError('Somethig wrong')

    #inputs layers
    inp_enc = tf.keras.Input(shape=(T, D))
    #inp_dec = tf.keras.Input(shape=[None, T, D])
    positional_encoding_enc = positional_encoding(T, d_model)
    #positional_encoding_dec = positional_encoding(T, d_model)
    inp_ = tf.keras.layers.Dense(d_model)(inp_enc) #embedding
    #inp_dec = tf.keras.layers.Dense(d_model)(inp_dec) #embedding
    outputs_enc = inp_ + positional_encoding_enc
    #outputs_dec = inp_dec + positional_encoding_dec
    outputs_enc = TransformerBlock(d_model, num_heads, ff_dim)(outputs_enc, generative=True)
    outputs_enc = tf.keras.layers.Dense(out_dim, activation='sigmoid')(outputs_enc)

    model = Model(inputs=inp_enc, outputs=outputs_enc)
    model.summary()

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=['mse'], metrics=['mse'], optimizer=opt)

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

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0000001, patience=20,
                                                               restore_best_weights=True, verbose=0)
    callbacks += [early_stopping_callback]

    # train
    if generative:
        #results = model.fit(sigs[:N//2, :-1, :], sigs[:N//2, 1:, 0], batch_size=b_size, epochs=epochs, verbose=0,
                        #validation_data = (sigs[N // 2 + 1:(2 * N) // 3, :-1, :], sigs[N // 2 + 1:(2 * N) // 3, 1:, 0]),
                        #callbacks = callbacks)
        for e in range(epochs):
            for i in range(N//2):

                results = model.fit(sigs[i:i+1, :-1, :], sigs[i:i+1, 1:, 0], batch_size=b_size, epochs=1, verbose=0,
                        validation_data=(sigs[N//2+i:N//2+i+1, :-1, :], sigs[N//2+i:N//2+i+1, 1:, 0]),
                        callbacks=callbacks)
    elif not generative and not all:
        results = model.fit(x, y[:, -1], batch_size=b_size, epochs=epochs, verbose=0,
                        validation_data=(x_val, y_val[:, -1]),
                        callbacks=callbacks)

    elif not generative and all:
        results = model.fit(x, y, batch_size=b_size, epochs=epochs, verbose=0,
                        validation_data=(x_val, y_val),
                        callbacks=callbacks)

    if generative:
        #test_loss = model.evaluate(sigs[(2*N)//3+1:, :-1, :], sigs[(2*N)//3+1:, 1:, 0], batch_size=b_size, verbose=0)
        test_loss = model.evaluate(sigs[N-2:N-1, :-1, :], sigs[N-2:N-1, 1:, 0], batch_size=b_size, verbose=0)
    elif not generative:
        test_loss = model.evaluate(x_test, y_test[:, -1], batch_size=b_size, verbose=0)

    print('Test Loss: ', test_loss)

    results = {
        'Test_Loss': test_loss,
        'Min_val_loss': np.min(results.history['val_loss']),
        'Min_train_loss': np.min(results.history['loss']),
        'b_size': b_size,
        'learning_rate': learning_rate,
        'drop': drop,
        'opt_type': opt_type,
        'loss_type': loss_type,
        'd_model': d_model,
        'ff_dim': ff_dim,
        'num_heads': num_heads,
        'w_length': w_length,
        # 'Train_loss': results.history['loss']
        'Val_loss': results.history['val_loss']
        }
    # print(results)
    if ckpt_flag:
        with open(os.path.normpath(
                '/'.join([model_save_dir, save_folder, 'result.txt'])), 'w') as f:
            for key, value in results.items():
                print('\n', key, '  : ', value, file=f)
            pickle.dump(results,
                        open(os.path.normpath(
                            '/'.join([model_save_dir, save_folder, 'results.pkl'])),
                                'wb'))


    if ckpt_flag:
        best = tf.train.latest_checkpoint(ckpt_dir)
        if best is not None:
            print("Restored weights from {}".format(ckpt_dir))
            model.load_weights(best)




    if generate_wav is not None:

        if generative:
            predictions = model.predict(sigs[N-2:N-1, :-1, :], batch_size=b_size)
            predictions = scaler[0].inverse_transform(predictions)
            y_test = scaler[0].inverse_transform(sigs[N-2:N-1, 1:, 0])

        if not generative:
            predictions = model.predict(x_test, batch_size=b_size)
            predictions = (scaler[0].inverse_transform(predictions)).reshape(-1)
            x_test = (scaler[0].inverse_transform(x_test[:, :, 0])).reshape(-1)
            y_test = (scaler[0].inverse_transform(y_test)).reshape(-1)
            x_test = x_test.astype('int16')

            inp_name = '_inp.wav'
            inp_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', inp_name))
            if not os.path.exists(os.path.dirname(inp_dir)):
                os.makedirs(os.path.dirname(inp_dir))
            wavfile.write(inp_dir, 44100, x_test)

        # Define directories
        pred_name = '_pred.wav'
        tar_name = '_tar.wav'

        pred_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', pred_name))
        tar_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', tar_name))

        if not os.path.exists(os.path.dirname(pred_dir)):
            os.makedirs(os.path.dirname(pred_dir))

        # Save Wav files
        predictions = predictions.astype('int16')
        y_test = y_test.astype('int16')
        wavfile.write(pred_dir, 44100, predictions[0][0].T)
        wavfile.write(tar_dir, 44100, y_test.T)

    return results

if __name__ == '__main__':
    data_dir = '../../Files'
    #data_dir = '/scratch/users/riccarsi/Files'
    seed = 422
    #start = time.time()
    trainMultiAttention(data_dir=data_dir,
              model_save_dir='../../TrainedModels',
              save_folder='MultiAttention_gen2',
              ckpt_flag=True,
              b_size=128,
              learning_rate=0.001,
              d_model=512,
              ff_dim=512,
              num_heads=8,
              epochs=100,
              loss_type='mse',
              generate_wav=10,
              w_length=16,
              generative=True,
              all=False)