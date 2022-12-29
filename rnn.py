class RecurrentNeuralNetwork():
    def recurrent_neural_network() -> None:
        from tensorflow.keras.datasets import imdb
        (train_input, train_target), (
            test_input, test_target) = imdb.load_data(num_words=500)
        # print(train_input.shape, test_input.shape)
        from sklearn.model_selection import train_test_split
        train_input, val_input, train_target, val_target = train_test_split(
            train_input, train_target, test_size=0.2, random_state=42)
        import numpy as np
        lengths = np.array([len(x) for x in train_input])

        # print(np.mean(lengths), np.median(lengths))
        import matplotlib.pyplot as plt
        # plt.hist(lengths)
        # plt.xlabel('length')
        # plt.ylabel('frequency')
        # plt.show()

        from tensorflow.keras.preprocessing.sequence import pad_sequences
        train_seq = pad_sequences(train_input, maxlen=100)
        # print(train_seq[0][-10:])
        val_seq = pad_sequences(val_input, maxlen=100)

        from tensorflow import keras
        model = keras.Sequential()
        model.add(keras.layers.SimpleRNN(8, input_shape=(100, 500)))
        model.add(keras.layers.Dense(1, activation='sigmoid'))

        train_oh = keras.utils.to_categorical(train_seq)
        # print(train_oh.shape)
        # print(train_oh[0][0][:12])
        # print(np.sum(train_oh[0][0]))
        val_oh = keras.utils.to_categorical(val_seq)
        # print(val_oh.shape)
        model.summary()

        import os
        BEST_SIMPLERNN_MODEL_H5 = 'best-simplernn-model.h5'
        if not BEST_SIMPLERNN_MODEL_H5 in os.listdir():
            rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
            model.compile(optimizer=rmsprop,
                          loss='binary_crossentropy', metrics=['accuracy'])
            checkpoint_cb = keras.callbacks.ModelCheckpoint(
                BEST_SIMPLERNN_MODEL_H5, save_best_only=True)
            early_stopping_cb = keras.callbacks.EarlyStopping(
                patience=3, restore_best_weights=True)
            history = model.fit(train_oh, train_target, epochs=100, batch_size=64,
                                validation_data=(val_oh, val_target),
                                callbacks=[checkpoint_cb, early_stopping_cb])
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend(['train', 'val'])
            plt.show()
        else:
            model = keras.models.load_model(BEST_SIMPLERNN_MODEL_H5)
            # print(train_seq.nbytes, train_oh.nbytes)

        model2 = keras.Sequential()
        model2.add(keras.layers.Embedding(500, 16, input_length=100))
        model2.add(keras.layers.SimpleRNN(8))
        model2.add(keras.layers.Dense(1, activation='sigmoid'))
        model2.summary()

        BEST_EMBEDDING_MODEL_H5 = 'best-embedding-model.h5'
        if not BEST_EMBEDDING_MODEL_H5 in os.listdir():
            rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
            model2.compile(optimizer=rmsprop,
                           loss='binary_crossentropy', metrics=['accuracy'])
            checkpoint_cb = keras.callbacks.ModelCheckpoint(
                BEST_EMBEDDING_MODEL_H5, save_best_only=True)
            early_stopping_cb = keras.callbacks.EarlyStopping(
                patience=3, restore_best_weights=True)
            history = model2.fit(train_seq, train_target, epochs=100,
                                 batch_size=64,
                                 validation_data=(val_seq, val_target),
                                 callbacks=[checkpoint_cb, early_stopping_cb])
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend(['train', 'val'])
            plt.show()
        else:
            model2 = keras.models.load_model(BEST_EMBEDDING_MODEL_H5)

    def lstm() -> None:
        from tensorflow.keras.datasets import imdb
        from sklearn.model_selection import train_test_split
        (train_input, train_target), (
            test_input, test_target) = imdb.load_data(num_words=500)
        train_input, val_input, train_target, val_target = train_test_split(
            train_input, train_target, test_size=0.2, random_state=42)
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        train_seq = pad_sequences(train_input, maxlen=100)
        val_seq = pad_sequences(val_input, maxlen=100)

        from tensorflow import keras
        import os
        BEST_LSTM_MODEL_H5 = 'best-lstm-model.h5'
        if not BEST_LSTM_MODEL_H5 in os.listdir():
            model = keras.Sequential()
            model.add(keras.layers.Embedding(500, 16, input_length=100))
            model.add(keras.layers.LSTM(8))
            model.add(keras.layers.Dense(1, activation='sigmoid'))
            model.summary()

            rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
            model.compile(optimizer=rmsprop,
                          loss='binary_crossentropy', metrics=['accuracy'])
            checkpoint_cb = keras.callbacks.ModelCheckpoint(
                BEST_LSTM_MODEL_H5, save_best_only=True)
            early_stopping_cb = keras.callbacks.EarlyStopping(
                patience=3, restore_best_weights=True)
            history = model.fit(train_seq, train_target, epochs=100,
                                batch_size=64,
                                validation_data=(val_seq, val_target),
                                callbacks=[checkpoint_cb, early_stopping_cb])

            import matplotlib.pyplot as plt
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend(['train', 'val'])
            plt.show()

        BEST_DROPOUT_MODEL_H5 = 'best-dropout-model.h5'
        if not BEST_DROPOUT_MODEL_H5 in os.listdir():
            model2 = keras.Sequential()
            model2.add(keras.layers.Embedding(500, 16, input_length=100))
            model2.add(keras.layers.LSTM(8, dropout=0.3))
            model2.add(keras.layers.Dense(1, activation='sigmoid'))

            rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
            model2.compile(optimizer=rmsprop,
                           loss='binary_crossentropy', metrics=['accuracy'])
            checkpoint_cb = keras.callbacks.ModelCheckpoint(
                BEST_DROPOUT_MODEL_H5, save_best_only=True)
            early_stopping_cb = keras.callbacks.EarlyStopping(
                patience=3, restore_best_weights=True)
            history = model2.fit(train_seq, train_target, epochs=100, batch_size=64,
                                 validation_data=(val_seq, val_target),
                                 callbacks=[checkpoint_cb, early_stopping_cb])

            import matplotlib.pyplot as plt
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend(['train', 'val'])
            plt.show()

        BEST_2RNN_MODEL_H5 = 'best-2rnn-model.h5'
        if not BEST_2RNN_MODEL_H5 in os.listdir():
            model3 = keras.Sequential()
            model3.add(keras.layers.Embedding(500, 16, input_length=100))
            model3.add(keras.layers.LSTM(
                8, dropout=0.3, return_sequences=True))
            model3.add(keras.layers.LSTM(8, dropout=0.3))
            model3.add(keras.layers.Dense(1, activation='sigmoid'))
            # model3.summary()

            rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
            model3.compile(optimizer=rmsprop,
                           loss='binary_crossentropy', metrics=['accuracy'])
            checkpoint_cb = keras.callbacks.ModelCheckpoint(
                BEST_2RNN_MODEL_H5, save_best_only=True)
            early_stopping_cb = keras.callbacks.EarlyStopping(
                patience=3, restore_best_weights=True)
            history = model3.fit(train_seq, train_target,
                                 epochs=100, batch_size=64,
                                 validation_data=(val_seq, val_target),
                                 callbacks=[checkpoint_cb, early_stopping_cb])
            import matplotlib.pyplot as plt
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend(['train', 'val'])
            plt.show()
