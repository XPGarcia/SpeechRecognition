import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras

# The set of characters accepted in the transcription.
characters = [x for x in "abcdefghijklmnñopqrstuvwxyz¿?¡!áéíóú "]
# Mapping characters to integers
char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
# Mapping integers back to original characters
num_to_char = keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

# An integer scalar Tensor. The window length in samples.
frame_length = 768
# An integer scalar Tensor. The number of samples to step.
frame_step = 384
# An integer scalar Tensor. The size of the FFT to apply.
# If not provided, uses the smallest power of 2 enclosing frame_length.
fft_length = 768


def encode_single_sample(wav_file):
    file = tf.io.read_file(wav_file)
    audio, audioSR = tf.audio.decode_wav(file)
    audio = tf.squeeze(audio, axis=-1)
    input_len = audioSR * 6
    audio = audio[:input_len]
    zero_padding = tf.zeros(
        [input_len] - tf.shape(audio),
        dtype=tf.float32)
    audio = tf.cast(audio, tf.float32)
    equal_length = tf.concat([audio, zero_padding], 0)
    stft = tf.signal.stft(
        equal_length, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length
    )
    spectrogram = tf.abs(stft)
    spectrogram = tf.math.pow(spectrogram, 0.5)
    means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)
    return spectrogram


# A utility function to decode the output of the network
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    # Iterate over the results and get back the text
    output_text = []
    for result in results:
        result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
        output_text.append(result)
    return output_text


def CTCLoss(y_true, y_pred):
    # Compute the training-time loss value
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss


def load_model():
    model = tf.keras.models.load_model("model_speech", custom_objects={'CTCLoss': CTCLoss})
    return model


def predict(filename, model):
    predictions = []
    spectrogram = encode_single_sample(filename)
    expanded = tf.expand_dims(spectrogram, axis=0)
    batch_predictions = model.predict(expanded)
    batch_predictions = decode_batch_predictions(batch_predictions)
    predictions.extend(batch_predictions)
    return predictions
