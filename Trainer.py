import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from jiwer import wer


def train(n_train, model_name, previous_model_name, build=False):
    data_path = "C:\\Users\\Xavier\\datasetWav\\"
    wavs_path = data_path + "clips\\"

    metadata = pd.read_csv(data_path + 'train.tsv', sep='\t')
    metadata.columns = ["client_id", "path", "sentence", "up_votes", "down_votes", "age", "gender", "accent", "locale", "segment"]
    metadata = metadata[["path", "sentence"]]
    metadata["path"] = metadata["path"].str.replace(".mp3", '.wav')
    metadata = metadata.sample(frac=1).reset_index(drop=True)

    # Length of train records per epoch
    max_train_data = 10000
    max_test_data = 1000

    # Train and validation dataset
    df_train = metadata[max_train_data*(n_train-1):max_train_data*n_train]
    df_val = metadata[max_train_data*n_train:max_train_data*n_train + max_test_data]
    df_test = metadata[15000:16000]

    # The set of characters accepted in the transcription.
    characters = [x for x in "abcdefghijklmnñopqrstuvwxyz¿?¡!áéíóú "]
    # Mapping characters to integers
    char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
    # Mapping integers back to original characters
    num_to_char = keras.layers.StringLookup(
        vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
    )

    # Variables for stft transformation
    frame_length = 768
    frame_step = 384
    fft_length = 768

    def encode_single_sample(wav_file, label):
        ###########################################
        # Process the audio
        ##########################################
        file = tf.io.read_file(wavs_path + wav_file)
        audio, audioSR = tf.audio.decode_wav(file)
        audio = tf.squeeze(audio, axis=-1)
        input_len = audioSR * 6
        audio = audio[:input_len]
        zero_padding = tf.zeros(
            [input_len] - tf.shape(audio),
            dtype=tf.float32)
        audio = tf.cast(audio, tf.float32)
        # Same length
        equal_length = tf.concat([audio, zero_padding], 0)
        # Get Spectrogram
        stft = tf.signal.stft(
            equal_length, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length
        )
        # Magnitude of spectrogram
        spectrogram = tf.abs(stft)
        spectrogram = tf.math.pow(spectrogram, 0.5)
        # Normalisation
        means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
        stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
        spectrogram = (spectrogram - means) / (stddevs + 1e-10)
        ###########################################
        # Process the label
        ##########################################
        label = tf.strings.lower(label)
        label = tf.strings.unicode_split(label, input_encoding="UTF-8")
        # Map characters to numbers
        label = char_to_num(label)

        return spectrogram, label

    # Variable of batch
    batch_size = 32
    # Define the training dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (list(df_train["path"]), list(df_train["sentence"]))
    )

    train_dataset = (
        train_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
        .padded_batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    # Define the validation dataset
    val_dataset = tf.data.Dataset.from_tensor_slices(
        (list(df_val["path"]), list(df_val["sentence"]))
    )
    val_dataset = (
        val_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
        .padded_batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    # Define the test dataset
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (list(df_test["path"]), list(df_test["sentence"]))
    )
    test_dataset = (
        test_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
        .padded_batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    def CTCLoss(y_true, y_pred):
        # Compute the training-time loss value
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
        return loss

    def build_model(input_dim, output_dim, rnn_layers=2, rnn_units=128):
        # Model's input
        input_spectrogram = layers.Input((None, input_dim), name="input")
        # Expand the dimension
        x = layers.Reshape((-1, input_dim, 1), name="expand_dim")(input_spectrogram)
        # Convolution layer 1
        x = layers.Conv2D(
            filters=32,
            kernel_size=[11, 41],
            strides=[2, 2],
            padding="same",
            use_bias=False,
            name="conv_1",
        )(x)
        x = layers.BatchNormalization(name="conv_1_bn")(x)
        x = layers.ReLU(name="conv_1_relu")(x)
        # x = layers.MaxPool2D(pool_size=[3, 3], strides=[2, 2])(x)
        # Convolution layer 2
        x = layers.Conv2D(
            filters=32,
            kernel_size=[11, 21],
            strides=[1, 2],
            padding="same",
            use_bias=False,
            name="conv_2",
        )(x)
        x = layers.BatchNormalization(name="conv_2_bn")(x)
        x = layers.ReLU(name="conv_2_relu")(x)
        # x = layers.MaxPool2D(pool_size=[3, 3], strides=[2, 2])(x)
        # Reshape the resulted volume to feed the RNNs layers
        x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)
        # RNN layers
        for i in range(1, rnn_layers + 1):
            recurrent = layers.GRU(
                units=rnn_units,
                activation="tanh",
                recurrent_activation="sigmoid",
                use_bias=True,
                return_sequences=True,
                reset_after=True,
                name=f"gru_{i}",
            )
            x = layers.Bidirectional(
                recurrent, name=f"bidirectional_{i}", merge_mode="concat"
            )(x)
            if i < rnn_layers:
                x = layers.Dropout(rate=0.2)(x)
        # Dense layer
        x = layers.Dense(units=rnn_units * 2, name="dense_1")(x)
        x = layers.ReLU(name="dense_1_relu")(x)
        x = layers.Dropout(rate=0.2)(x)
        # Classification layer
        output = layers.Dense(units=output_dim + 1, activation="softmax")(x)
        # Model
        model = keras.Model(input_spectrogram, output, name="SpeechRecognition")
        # Optimizer
        opt = keras.optimizers.Adam(learning_rate=1e-4)
        # Compile the model and return
        model.compile(optimizer=opt, loss=CTCLoss)
        return model

    if build:
        # Build model
        model = build_model(
            input_dim=(fft_length // 2 + 1),
            output_dim=char_to_num.vocabulary_size(),
            rnn_units=512,
        )
        model.summary(line_length=110)
    else:
        # Load model
        model = model = tf.keras.models.load_model(previous_model_name, custom_objects={'CTCLoss': CTCLoss})
        model.summary(line_length=110)

    # A utility function to decode the output of the network
    def decode_batch_predictions(pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
        output_text = []
        for result in results:
            result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
            output_text.append(result)
        return output_text

    # A callback class to output a few transcriptions during training
    class CallbackEval(keras.callbacks.Callback):
        def __init__(self, dataset):
            super().__init__()
            self.dataset = dataset

        def on_epoch_end(self, epoch: int, logs=None):
            predictions = []
            targets = []
            for batch in self.dataset:
                X, y = batch
                batch_predictions = model.predict(X)
                batch_predictions = decode_batch_predictions(batch_predictions)
                predictions.extend(batch_predictions)
                for label in y:
                    label = (
                        tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
                    )
                    targets.append(label)
            wer_score = wer(targets, predictions)
            print("-" * 100)
            print(f"Word Error Rate: {wer_score:.4f}")
            print("-" * 100)
            for i in np.random.randint(0, len(predictions), 2):
                print(f"Target    : {targets[i]}")
                print(f"Prediction: {predictions[i]}")
                print("-" * 100)

    # Define the number of epochs.
    epochs = 20
    # Callback function to check transcription on the val set.
    test_callback = CallbackEval(val_dataset)
    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[test_callback],
    )
    # Save model
    model.save(model_name)
    return history

