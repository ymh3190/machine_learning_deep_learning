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
        plt.hist(lengths)
        plt.xlabel('length')
        plt.ylabel('frequency')
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
        print(np.sum(train_oh[0][0]))
