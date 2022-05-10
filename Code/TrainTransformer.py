import tensorflow as tf
from keras.utils.generic_utils import Progbar
from tensorflow.keras.optimizers import Adam
import numpy as np
from Transformer_2 import Transformer, Schedule, MaskHandler
from tqdm import tqdm
from GetData import get_data

# Initialize parameters
num_layers = 4
num_neurons = 128
num_hidden_layers = 512
num_heads = 8
epochs = 1
b_size = 1
x, y, scaler = get_data(data_dir='../Files', seed=422)

max_length = y.shape[1]

# Initialize learning rate
learning_rate = Schedule(num_neurons)
optimizer = Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

# Initialize transformer
transformer = Transformer(num_layers, num_neurons, num_hidden_layers, num_heads, max_length, max_length)


train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int16),
    tf.TensorSpec(shape=(None, None), dtype=tf.int16),
]

loss_fn = tf.keras.losses.MeanSquaredError()

train_loss = tf.keras.metrics.Mean(name='train_loss')
val_loss = tf.keras.metrics.Mean(name='val_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')



#@tf.function(input_signature=train_step_signature)
def train_step(input_language, target_language):
    target_input = target_language[:, :-1]
    target_output = target_language[:, 1:]


    look_ahead_mask = MaskHandler.look_ahead_mask(size=tf.shape(target_language)[1])
    #decoder_target_padding_mask = MaskHandler.padding_mask(target_language)
    #combined_mask = tf.maximum(decoder_target_padding_mask, look_ahead_mask)

    # Run training step
    with tf.GradientTape() as tape:
        predictions, _ = transformer(input_language, target_input, True)
        total_loss = loss_fn(target_output, predictions[:, :, 0])

    gradients = tape.gradient(total_loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    train_loss.update_state(total_loss)

summary_res = 1
for epoch in tqdm(range(20)):
    train_loss.reset_states()
    val_loss.reset_states()

    # Get batches
    x_batches = x
    y_batches = y
    # Set-up training progress bar
    n_batch = len(y_batches)
    print("\nepoch {}/{}".format(epoch + 1, epochs))
    pb_i = Progbar(n_batch * b_size, stateful_metrics=['Loss: '])

    for batch_num in range(len(y_batches)):
        x_batch = x_batches#[batch_num * b_size:batch_num * b_size + b_size]
        y_batch = y_batches#[batch_num * b_size:batch_num * b_size + b_size]

    x_batch = tf.constant(x_batch, dtype='float32')
    y_batch = tf.constant(y_batch, dtype='float32')

    train_step(input_language=x_batch, target_language=y_batch)

    # Print progbar
    if batch_num % summary_res == 0:
        values = [('Loss: ', train_loss.result())]
        pb_i.add(b_size * summary_res, values=values)
